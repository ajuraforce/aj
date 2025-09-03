"""
Portfolio Optimization Engine

Implements modern portfolio theory for optimal asset allocation with signal integration,
dynamic risk adjustments, and multiple optimization strategies (MVO, Risk Parity, HRP).
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import logging
import openai
import json
import os
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.covariance import LedoitWolf
except ImportError:
    # Fallback if sklearn not available
    AgglomerativeClustering = None
    LedoitWolf = None

logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    weight: float = 0.0
    unrealized_pnl: float = 0.0
    sector: str = ""

class PortfolioOptimizer:
    def __init__(self, config: dict, db_path: str):
        self.config = config
        self.db_path = db_path
        portfolio_config = config.get('portfolio_optimization', {})
        self.base_max_sector_exposure = portfolio_config.get('max_sector_exposure', 0.3)
        self.base_max_single_asset = portfolio_config.get('max_single_asset', 0.15)
        self.rebalance_frequency = portfolio_config.get('rebalance_frequency', 'daily')
        self.risk_target = portfolio_config.get('risk_target', 0.65)
        self.correlation_threshold = portfolio_config.get('correlation_threshold', 0.7)
        self.volatility_window = portfolio_config.get('volatility_window', 30)
        self.optimization_method = portfolio_config.get('optimization_method', 'mvo')  # 'mvo', 'risk_parity', 'hrp'
        # Portfolio state
        self.positions = {}
        self.portfolio_value = 100000.0  # Default 100k portfolio (ensure float)
        self.signal_data = {}  # To store signal confidence: {symbol: confidence}
        self.historical_performance = {}  # For feedback loop: {symbol: {'hit_rate': float, 'realized_return': float}}
        
        # Initialize GPT-5 client for portfolio commentary
        try:
            self.gpt5_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.gpt5_enabled = True
            logger.info("GPT-5 portfolio advisor initialized successfully")
        except Exception as e:
            logger.warning(f"GPT-5 portfolio advisor initialization failed: {e}")
            self.gpt5_client = None
            self.gpt5_enabled = False
        
        # Enhancement settings
        self.backtesting_enabled = config.get('enhancements', {}).get('backtesting', {}).get('enabled', False)
        self.backtesting_periods = config.get('enhancements', {}).get('backtesting', {}).get('historical_periods', 252)
        
    def update_signals(self, signal_data: Dict[str, float]):
        """Update signal confidence data for allocation weighting."""
        self.signal_data = signal_data

    def update_historical_performance(self, symbol: str, hit_rate: float, realized_return: float):
        """Update historical performance for feedback loop."""
        self.historical_performance[symbol] = {'hit_rate': hit_rate, 'realized_return': realized_return}
        
    def get_returns_data(self, symbols: List[str], periods: int = 252) -> pd.DataFrame:
        """Fetch return data for portfolio optimization. Use log returns and shrinkage covariance."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if market_data table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
            if not cursor.fetchone():
                logger.warning("market_data table not found, returning empty dataframe")
                conn.close()
                return pd.DataFrame()
            
            returns_data = {}
            for symbol in symbols:
                query = """
                SELECT timestamp, close
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=[symbol, periods])
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df = df.sort_values('timestamp')
                    df['returns'] = np.log(df['close'] / df['close'].shift(1)).dropna()
                    if len(df['returns'].dropna()) > 0:
                        returns_data[symbol] = df['returns'].dropna()
            conn.close()
            if returns_data:
                returns_df = pd.DataFrame(returns_data).dropna()  # Align timestamps and drop NaNs
                return returns_df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching returns data: {e}")
            return pd.DataFrame()
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if market_data table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
            if not cursor.fetchone():
                logger.warning("market_data table not found, returning empty prices dict")
                conn.close()
                return {}
            
            prices = {}
            for symbol in symbols:
                query = """
                SELECT close
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """
                result = conn.execute(query, (symbol,)).fetchone()
                if result:
                    prices[symbol] = result[0]
            
            conn.close()
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return {}
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, returns_df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio using shrinkage covariance."""
        try:
            if returns_df.empty or len(weights) != len(returns_df.columns):
                return 0.0, 1.0, 0.0
                
            # Use shrinkage covariance if available
            if LedoitWolf is not None:
                cov_matrix = LedoitWolf().fit(returns_df).covariance_
            else:
                cov_matrix = returns_df.cov().values
                
            # Safe calculation with proper error handling
            mean_returns = returns_df.mean()
            if isinstance(mean_returns, pd.Series) and len(mean_returns) == 0:
                return 0.0, 1.0, 0.0
            if np.any(np.isnan(mean_returns)):
                return 0.0, 1.0, 0.0
                
            portfolio_return = np.dot(weights, mean_returns) * 252
            try:
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix * 252, weights))
                if np.isnan(portfolio_variance) or portfolio_variance < 0:
                    portfolio_variance = 0.0001
            except (ZeroDivisionError, ValueError):
                portfolio_variance = 0.0001
            
            # Ensure non-negative variance and avoid division by zero
            portfolio_variance = max(portfolio_variance, 0.0001)
            portfolio_volatility = max(np.sqrt(portfolio_variance), 0.0001)
            
            risk_free_rate = 0.02
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            return portfolio_return, portfolio_volatility, sharpe_ratio
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return 0.0, 1.0, 0.0
    
    def detect_regime(self, returns_df: pd.DataFrame) -> str:
        """Detect market regime: 'crisis', 'bull', 'neutral' based on volatility and correlations."""
        if returns_df.empty:
            return 'neutral'
        vol_series = returns_df.std()
        if isinstance(vol_series, pd.Series) and len(vol_series) == 0:
            return 'neutral'
        if isinstance(vol_series, pd.Series):
            vol = vol_series.mean() * np.sqrt(252)
        else:
            vol = float(vol_series) * np.sqrt(252)
        corr_matrix = returns_df.corr()
        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)]
        avg_corr = np.nanmean(upper_triangle) if len(upper_triangle) > 0 else 0.0
        if vol > self.risk_target * 1.5 or avg_corr > self.correlation_threshold:
            return 'crisis'
        elif vol < self.risk_target * 0.75 and avg_corr < self.correlation_threshold / 2:
            return 'bull'
        return 'neutral'

    def compute_sector_correlations(self, returns_df: pd.DataFrame, sectors_config: dict) -> Dict[str, float]:
        """Compute average intra-sector correlations."""
        sector_corrs = {}
        for sector_name, sector_data in sectors_config.get('sectors', {}).items():
            sector_symbols = [s for s in sector_data.get('assets', []) if s in returns_df.columns]
            if len(sector_symbols) > 1:
                sector_df = returns_df[sector_symbols]
                try:
                    corr_matrix = sector_df.corr()
                    if corr_matrix.empty or corr_matrix.isna().all().all():
                        sector_corrs[sector_name] = 0.0
                        continue
                except Exception:
                    sector_corrs[sector_name] = 0.0
                    continue
                upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)]
                sector_corrs[sector_name] = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
            else:
                sector_corrs[sector_name] = 0.0
        return sector_corrs

    def backtest_strategy(self, symbols: List[str], historical_periods: int) -> Dict:
        """
        Walk-forward backtesting using historical market_data.
        """
        try:
            df = self.get_returns_data(symbols, periods=historical_periods)
            if df.empty:
                return {'error': 'No historical return data'}
            
            returns_list, sharpe_list = [], []
            window = self.config.get('portfolio_optimization', {}).get('walk_forward_window', 50)
            
            for start in range(0, len(df) - 2 * window, window):
                train = df.iloc[start:start+window]
                test = df.iloc[start+window:start+2*window]
                
                # Optimize on train
                weights_dict = self.optimize_portfolio_weights(symbols, self.config.get('sectors', {}))
                weights = np.array(list(weights_dict.values()))
                
                # Calculate returns on test
                if len(test) > 0 and len(weights) == len(test.columns):
                    port_ret = np.dot(test, weights).sum(axis=1) if hasattr(test, 'sum') else test.sum(axis=1)
                    if hasattr(port_ret, 'mean') and hasattr(port_ret, 'std'):
                        returns_list.append(port_ret.mean())
                        sharpe_list.append(port_ret.mean() / (port_ret.std() + 1e-8))
            
            return {
                'avg_return': float(np.mean(returns_list)) if returns_list else 0.0,
                'avg_sharpe': float(np.mean(sharpe_list)) if sharpe_list else 0.0,
                'periods_tested': len(returns_list)
            }
        except Exception as e:
            logger.error(f"Error in backtesting strategy: {e}")
            return {'error': str(e), 'avg_return': 0.0, 'avg_sharpe': 0.0, 'periods_tested': 0}

    def adjust_dynamic_limits(self, regime: str, sector_correlations: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Dynamically adjust single-asset and sector limits based on regime and correlations."""
        max_single = self.base_max_single_asset
        max_sectors = {sector: self.base_max_sector_exposure for sector in sector_correlations}
        if regime == 'crisis':
            max_single *= 0.8
            for sector in max_sectors:
                if sector_correlations[sector] > self.correlation_threshold:
                    max_sectors[sector] *= 0.7
        elif regime == 'bull':
            max_single *= 1.2
            for sector in max_sectors:
                if sector_correlations[sector] < self.correlation_threshold / 2:
                    max_sectors[sector] *= 1.1
        return max_single, max_sectors

    def black_litterman_expected_returns(self, returns_df: pd.DataFrame, prior_returns: np.ndarray) -> np.ndarray:
        """Simplified Black-Litterman to incorporate signal views."""
        expected_returns = prior_returns.copy()
        for symbol, idx in zip(returns_df.columns, range(len(returns_df.columns))):
            confidence = self.signal_data.get(symbol, 0.5)
            hit_rate = self.historical_performance.get(symbol, {'hit_rate': 0.5})['hit_rate']
            view_return = confidence * hit_rate * (returns_df[symbol].mean() * 252 + 0.05)  # Bias upward for positive signals
            expected_returns[idx] = 0.7 * expected_returns[idx] + 0.3 * view_return  # Blend with prior
        return expected_returns

    def optimize_mvo(self, returns_df: pd.DataFrame, bounds: list, constraints: list) -> np.ndarray:
        """Mean-Variance Optimization to maximize Sharpe."""
        prior_returns = returns_df.mean() * 252
        expected_returns = self.black_litterman_expected_returns(returns_df, prior_returns)
        
        if LedoitWolf is not None:
            cov_matrix = LedoitWolf().fit(returns_df).covariance_ * 252
        else:
            cov_matrix = returns_df.cov().values * 252

        def negative_sharpe(weights):
            try:
                port_return = np.dot(weights, expected_returns)
                port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                port_vol = np.sqrt(max(port_variance, 0.0001))
                if port_vol <= 0 or np.isnan(port_vol):
                    port_vol = 0.0001
                sharpe = (port_return - 0.02) / port_vol
                return -sharpe if not np.isnan(sharpe) else 0
            except (ZeroDivisionError, ValueError, TypeError):
                return 0

        n_assets = len(returns_df.columns)
        if n_assets == 0:
            return np.array([])
        # Safely calculate initial weights avoiding division by zero
        if n_assets > 0:
            initial_weights = np.array([1.0 / n_assets] * n_assets)
        else:
            initial_weights = np.array([])
        result = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else initial_weights
    
    def optimize_risk_parity(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Risk Parity optimization - equal risk contribution from each asset"""
        try:
            if LedoitWolf is not None:
                cov_matrix = LedoitWolf().fit(returns_df).covariance_ * 252
            else:
                cov_matrix = returns_df.cov().values * 252
            
            n_assets = len(returns_df.columns)
            
            def risk_budget_objective(weights):
                """Minimize the sum of squared differences between risk contributions"""
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
                contrib = np.multiply(marginal_contrib, weights)
                contrib = contrib / np.sum(contrib)  # Normalize to sum to 1
                target = np.ones(n_assets) / n_assets  # Equal risk target
                return np.sum((contrib - target) ** 2)
            
            # Constraints and bounds
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0.01, 0.4) for _ in range(n_assets)]  # Min 1%, max 40% per asset
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            result = minimize(risk_budget_objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints, options={'maxiter': 1000})
            
            return result.x if result.success else initial_weights
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            n_assets = len(returns_df.columns)
            return np.array([1.0 / n_assets] * n_assets)
    
    def optimize_hrp(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Hierarchical Risk Parity optimization"""
        try:
            if LedoitWolf is not None:
                cov_matrix = LedoitWolf().fit(returns_df).covariance_ * 252
            else:
                cov_matrix = returns_df.cov().values * 252
            
            corr_matrix = returns_df.corr()
            
            # Calculate distance matrix
            dist = np.sqrt((1 - corr_matrix) / 2)
            
            # Hierarchical clustering (simplified version without sklearn if not available)
            if AgglomerativeClustering is not None:
                cluster = AgglomerativeClustering(n_clusters=None, linkage='single', 
                                                distance_threshold=0, affinity='precomputed')
                cluster.fit(dist)
                linkage_matrix = cluster.children_
            else:
                # Simple fallback clustering
                linkage_matrix = self._simple_linkage_clustering(dist.values)
            
            # Get quasi-diagonal order
            sort_ix = self._get_quasi_diag(linkage_matrix)
            
            # Recursive bisection
            weights = self._get_rec_bipart(pd.DataFrame(cov_matrix, 
                                                      index=returns_df.columns, 
                                                      columns=returns_df.columns), sort_ix)
            
            return weights.values / weights.sum()
            
        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            n_assets = len(returns_df.columns)
            return np.array([1.0 / n_assets] * n_assets)
    
    def _simple_linkage_clustering(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Simple linkage clustering fallback when sklearn not available"""
        n = len(dist_matrix)
        linkage_matrix = []
        clusters = {i: [i] for i in range(n)}
        
        for _ in range(n - 1):
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            # Find closest clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if i in clusters and j in clusters:
                        # Calculate average linkage distance
                        dist_sum = 0
                        count = 0
                        for idx_i in clusters[i]:
                            for idx_j in clusters[j]:
                                dist_sum += dist_matrix[idx_i][idx_j]
                                count += 1
                        
                        if count > 0:
                            avg_dist = dist_sum / count
                            if avg_dist < min_dist:
                                min_dist = avg_dist
                                merge_i, merge_j = i, j
            
            # Merge clusters
            if merge_i >= 0 and merge_j >= 0:
                new_cluster = clusters[merge_i] + clusters[merge_j]
                linkage_matrix.append([merge_i, merge_j, min_dist, len(new_cluster)])
                clusters[max(clusters.keys()) + 1] = new_cluster
                del clusters[merge_i]
                del clusters[merge_j]
        
        return np.array(linkage_matrix)
    
    def _get_quasi_diag(self, linkage_matrix: np.ndarray) -> list:
        """Get quasi-diagonal order from linkage matrix"""
        try:
            if len(linkage_matrix) == 0:
                return list(range(linkage_matrix.shape[0] + 1))
            
            # Simple implementation - return indices in reverse order
            n_original = len(linkage_matrix) + 1
            return list(range(n_original))
            
        except Exception as e:
            logger.error(f"Error getting quasi-diagonal order: {e}")
            return list(range(len(linkage_matrix) + 1))
    
    def _get_rec_bipart(self, cov: pd.DataFrame, sort_ix: list) -> pd.Series:
        """Recursively assign weights using bisection"""
        try:
            w = pd.Series(1, index=sort_ix)
            c_items = [sort_ix]
            
            while len(c_items) > 0:
                c_items = [i[j:k] for i in c_items 
                          for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
                
                for i in range(0, len(c_items), 2):
                    if i + 1 < len(c_items):
                        c_items0 = c_items[i]
                        c_items1 = c_items[i + 1]
                        
                        # Calculate variances
                        c_var0 = np.diag(cov.loc[c_items0, c_items0]).sum()
                        c_var1 = np.diag(cov.loc[c_items1, c_items1]).sum()
                        
                        # Calculate allocation
                        alpha = 1 - c_var0 / (c_var0 + c_var1)
                        w[c_items0] *= alpha
                        w[c_items1] *= 1 - alpha
            
            return w
            
        except Exception as e:
            logger.error(f"Error in recursive bisection: {e}")
            return pd.Series([1.0 / len(sort_ix)] * len(sort_ix), index=sort_ix)
    
    async def generate_gpt5_rebalancing_insights(self, recommendations: List[Dict], regime: str, portfolio_metrics: Dict) -> Dict:
        """Enhanced portfolio analysis with GPT-5 reasoning"""
        if not self.gpt5_enabled or not recommendations:
            return {'insights': 'GPT-5 portfolio analysis not available.', 'confidence': 0.0}
            
        try:
            # Format recommendations for analysis
            rec_text = "\n".join([f"- {rec['symbol']}: {rec['action']} {rec['change_pct']:.1%} (reason: {rec.get('reason', 'rebalancing')})" 
                                for rec in recommendations[:5]])
            
            prompt = f"""
            As an expert portfolio manager, analyze this rebalancing recommendation:
            
            Current Portfolio State:
            - Total Value: ${portfolio_metrics.get('total_value', 100000):,.2f}
            - Position Count: {portfolio_metrics.get('position_count', 0)}
            - Risk Score: {portfolio_metrics.get('risk_score', 5)}/10
            - Market Regime: {regime}
            - Current Volatility: {portfolio_metrics.get('volatility', 0):.2%}
            - Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}
            
            Recommended Changes:
            {rec_text}
            
            Provide detailed analysis including:
            1. Strategic rationale for rebalancing
            2. Risk-reward assessment  
            3. Market timing considerations
            4. Alternative approaches
            5. Implementation timeline recommendations
            6. Potential risks and mitigations
            
            Format response as JSON with keys: strategic_rationale, risk_assessment, timing_analysis, alternatives, implementation_plan, risk_mitigations
            """
            
            response = self.gpt5_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1200
            )
            
            analysis = json.loads(response.choices[0].message.content or '{}')
            
            return {
                'gpt5_analysis': analysis,
                'confidence': 0.85,
                'insights': analysis.get('strategic_rationale', 'No insights available'),
                'risk_assessment': analysis.get('risk_assessment', 'Risk analysis not available'),
                'implementation_plan': analysis.get('implementation_plan', 'No implementation plan available')
            }
            
        except Exception as e:
            logger.error(f"Error generating GPT-5 rebalancing insights: {e}")
            return {'insights': 'Portfolio analysis failed.', 'confidence': 0.0}
    
    async def generate_market_commentary(self, portfolio_metrics: Dict, market_data: Dict) -> str:
        """Generate daily market commentary using GPT-5"""
        if not self.gpt5_enabled:
            return "Market commentary not available - GPT-5 not configured."
            
        try:
            prompt = f"""
            Write a professional market commentary for institutional clients:
            
            Portfolio Performance:
            - Current allocation efficiency: {portfolio_metrics.get('efficiency', 'N/A')}
            - Risk-adjusted returns: {portfolio_metrics.get('sharpe_ratio', 'N/A')}
            - Current volatility: {portfolio_metrics.get('volatility', 0):.2%}
            - Sector exposures: {portfolio_metrics.get('sector_exposures', {})}
            - Portfolio value: ${portfolio_metrics.get('total_value', 0):,.2f}
            
            Market Conditions:
            - Market regime: {market_data.get('regime', 'unknown')}
            - Key economic events: {market_data.get('events', [])}
            - Sector rotations observed: {market_data.get('sector_rotations', [])}
            
            Create actionable insights for traders and portfolio managers.
            Style: Professional, data-driven, actionable insights
            Length: 150-250 words
            """
            
            response = self.gpt5_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=400
            )
            
            return response.choices[0].message.content or 'Market commentary not available'
            
        except Exception as e:
            logger.error(f"Error generating market commentary: {e}")
            return "Market commentary generation failed."

    def optimize_portfolio_weights(self, symbols: List[str], sectors_config: dict) -> Dict[str, float]:
        """Optimize portfolio weights using selected method with dynamic constraints and signals."""
        try:
            returns_df = self.get_returns_data(symbols)
            if returns_df.empty or len(returns_df.columns) < 2:
                # Handle empty symbols list to avoid division by zero
                if len(symbols) == 0:
                    return {}
                equal_weight = 1.0 / len(symbols)
                return {symbol: equal_weight for symbol in symbols}

            regime = self.detect_regime(returns_df)
            sector_corrs = self.compute_sector_correlations(returns_df, sectors_config)
            max_single, max_sectors = self.adjust_dynamic_limits(regime, sector_corrs)

            n_assets = len(returns_df.columns)
            bounds = [(0, max_single) for _ in range(n_assets)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

            for sector_name, max_exp in max_sectors.items():
                if sector_name in sectors_config.get('sectors', {}):
                    sector_symbols = [s for s in sectors_config['sectors'][sector_name].get('assets', []) if s in returns_df.columns]
                    if sector_symbols:
                        sector_indices = [returns_df.columns.get_loc(s) for s in sector_symbols]
                        constraints.append({'type': 'ineq', 'fun': lambda w, idx=sector_indices, max_e=max_exp: max_e - np.sum(w[idx])})

            if self.optimization_method == 'mvo':
                optimal_weights = self.optimize_mvo(returns_df, bounds, constraints)
            elif self.optimization_method == 'risk_parity':
                optimal_weights = self.optimize_risk_parity(returns_df)
            elif self.optimization_method == 'hrp':
                optimal_weights = self.optimize_hrp(returns_df)
            else:
                # Default to equal weights for unknown methods
                optimal_weights = np.array([1.0 / n_assets] * n_assets)

            weights_dict = {symbol: optimal_weights[i] for i, symbol in enumerate(returns_df.columns)}
            return weights_dict
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            # Handle empty symbols list to avoid division by zero
            if len(symbols) == 0:
                return {}
            equal_weight = 1.0 / len(symbols)
            return {symbol: equal_weight for symbol in symbols}
    
    def calculate_current_portfolio_state(self, sectors_config: dict) -> Dict:
        """Calculate current portfolio state"""
        try:
            current_prices = self.get_current_prices(list(self.positions.keys()))
            
            total_value = 0
            sector_exposures = {}
            position_details = []
            
            for symbol, position in self.positions.items():
                current_price = current_prices.get(symbol, position.entry_price)
                position_value = position.quantity * current_price
                total_value += position_value
                
                # Update position
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                # Safely calculate position weight avoiding division by zero
                if self.portfolio_value > 0:
                    position.weight = position_value / self.portfolio_value
                else:
                    position.weight = 0.0
                
                # Sector exposure tracking
                sector = position.sector
                sector_exposures[sector] = sector_exposures.get(sector, 0) + position.weight
                
                position_details.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'weight': position.weight,
                    'unrealized_pnl': position.unrealized_pnl,
                    'sector': sector
                })
            
            # Update portfolio value with minimum threshold
            self.portfolio_value = max(total_value, 1.0)
            
            return {
                'total_value': total_value,
                'positions': position_details,
                'sector_exposures': sector_exposures,
                'position_count': len(self.positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio state: {e}")
            return {
                'total_value': 0,
                'positions': [],
                'sector_exposures': {},
                'position_count': 0
            }
    
    def generate_rebalancing_recommendations(self, symbols: List[str], sectors_config: dict) -> Dict:
        """Generate portfolio rebalancing recommendations"""
        try:
            # Get optimal weights
            optimal_weights = self.optimize_portfolio_weights(symbols, sectors_config)
            
            # Get current portfolio state
            current_state = self.calculate_current_portfolio_state(sectors_config)
            
            # Calculate current weights
            current_weights = {}
            for position in current_state['positions']:
                current_weights[position['symbol']] = position['weight']
            
            # Generate recommendations
            recommendations = []
            total_trades_value = 0
            
            for symbol in symbols:
                current_weight = current_weights.get(symbol, 0)
                optimal_weight = optimal_weights.get(symbol, 0)
                weight_diff = optimal_weight - current_weight
                
                if abs(weight_diff) > 0.05:  # 5% threshold for rebalancing
                    # Safely calculate trade value avoiding division by zero
                    if self.portfolio_value > 0:
                        trade_value = weight_diff * self.portfolio_value
                    else:
                        trade_value = 0
                    
                    if weight_diff > 0:
                        action = "INCREASE"
                        target_change = f"+{weight_diff:.1%}"
                    else:
                        action = "REDUCE"
                        target_change = f"{weight_diff:.1%}"
                    
                    recommendations.append({
                        'symbol': symbol,
                        'action': action,
                        'current_weight': current_weight,
                        'target_weight': optimal_weight,
                        'weight_change': weight_diff,
                        'trade_value': abs(trade_value),
                        'recommendation': f"{symbol}: {current_weight:.1%} → {optimal_weight:.1%} ({target_change})"
                    })
                    
                    total_trades_value += abs(trade_value)
            
            # Calculate portfolio risk score
            returns_df = self.get_returns_data(symbols)
            if not returns_df.empty:
                current_portfolio_weights = np.array([current_weights.get(s, 0) for s in returns_df.columns])
                _, current_volatility, _ = self.calculate_portfolio_metrics(current_portfolio_weights, returns_df)
                
                optimal_portfolio_weights = np.array([optimal_weights.get(s, 0) for s in returns_df.columns])
                _, optimal_volatility, _ = self.calculate_portfolio_metrics(optimal_portfolio_weights, returns_df)
                
                # Safely calculate risk scores avoiding division by zero
                if current_volatility > 0:
                    current_risk_score = min(10, current_volatility * 50)  # Scale to 0-10
                else:
                    current_risk_score = 0.0
                    
                if optimal_volatility > 0:
                    optimal_risk_score = min(10, optimal_volatility * 50)
                else:
                    optimal_risk_score = 0.0
            else:
                current_risk_score = 5.0
                optimal_risk_score = 5.0
            
            # Check sector exposure violations
            sector_violations = []
            for sector, exposure in current_state['sector_exposures'].items():
                if exposure > self.base_max_sector_exposure:
                    sector_violations.append({
                        'sector': sector,
                        'current_exposure': exposure,
                        'max_allowed': self.base_max_sector_exposure,
                        'violation': exposure - self.base_max_sector_exposure
                    })
            
            result = {
                'rebalancing_needed': len(recommendations) > 0,
                'recommendations': recommendations,
                'total_trades_value': total_trades_value,
                'current_portfolio': current_state,
                'risk_analysis': {
                    'current_risk_score': current_risk_score,
                    'optimal_risk_score': optimal_risk_score,
                    'target_risk_score': self.risk_target * 10,
                    'regime': self.detect_regime(returns_df) if not returns_df.empty else 'neutral'
                },
                'sector_violations': sector_violations,
                'portfolio_metrics': {
                    'diversification_ratio': len([w for w in current_weights.values() if w > 0.01]),
                    'concentration_risk': max(current_weights.values()) if current_weights and current_weights.values() else 0,
                    'cash_allocation': max(0, 1 - sum(current_weights.values())) if current_weights else 1.0,
                    'total_value': current_state['total_value'],
                    'position_count': current_state['position_count'],
                    'volatility': optimal_volatility if not returns_df.empty else 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating rebalancing recommendations: {e}")
            return {
                'rebalancing_needed': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_portfolio_optimization(self, symbols: List[str], sectors_config: dict) -> Dict:
        """Run complete portfolio optimization process"""
        try:
            # Generate rebalancing recommendations
            rebalancing_results = self.generate_rebalancing_recommendations(symbols, sectors_config)
            
            # Add GPT-5 insights if enabled
            if self.gpt5_enabled and rebalancing_results.get('rebalancing_needed', False):
                regime = rebalancing_results['risk_analysis']['regime']
                gpt5_insights = await self.generate_gpt5_rebalancing_insights(
                    rebalancing_results['recommendations'], 
                    regime, 
                    rebalancing_results['portfolio_metrics']
                )
                rebalancing_results['gpt5_insights'] = gpt5_insights
            
            # Create formatted message for alerts
            if rebalancing_results['rebalancing_needed']:
                message_lines = ["📊 Portfolio Rebalancing Recommendation"]
                message_lines.append("Current Allocation:")
                
                for rec in rebalancing_results['recommendations'][:5]:  # Top 5 recommendations
                    message_lines.append(f"├── {rec['recommendation']}")
                
                risk_score = rebalancing_results['risk_analysis']['current_risk_score']
                target_risk = rebalancing_results['risk_analysis']['target_risk_score']
                message_lines.append(f"└── Risk Score: {risk_score:.1f}/10 (target: {target_risk:.1f}/10)")
                
                rebalancing_results['alert_message'] = '\n'.join(message_lines)
            
            return rebalancing_results
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {
                'error': str(e),
                'rebalancing_needed': False,
                'timestamp': datetime.now().isoformat()
            }