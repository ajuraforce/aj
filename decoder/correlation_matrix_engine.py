"""
Advanced Correlation Matrix Engine
Detects correlation breaks and cross-asset relationships
"""

import numpy as np
import pandas as pd
from scipy import stats
import sqlite3
from typing import Dict, List, Tuple, Optional
import asyncio
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class CorrelationMatrixEngine:
    def __init__(self, config: dict, db_path: str):
        self.config = config
        self.db_path = db_path
        correlation_config = config.get('correlation_analysis', {})
        self.window_size = correlation_config.get('window_size', 100)
        self.break_threshold = correlation_config.get('break_threshold', 0.3)
        self.significance_level = correlation_config.get('significance_level', 0.95)
        self.cross_market_pairs = correlation_config.get('cross_market_pairs', [])
        
        # Historical correlation storage
        self.correlation_history = {}
        
    def fetch_returns_data(self, symbols: List[str], lookback_periods: int = None) -> pd.DataFrame:
        """Fetch return data for correlation analysis"""
        try:
            if lookback_periods is None:
                lookback_periods = self.window_size * 2
                
            conn = sqlite3.connect(self.db_path)
            
            all_data = []
            for symbol in symbols:
                query = """
                SELECT timestamp, close, symbol
                FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(symbol, lookback_periods))
                if not df.empty:
                    df['returns'] = df['close'].pct_change()
                    df = df.dropna()
                    if not df.empty:
                        all_data.append(df[['timestamp', 'returns', 'symbol']])
            
            conn.close()
            
            if not all_data:
                return pd.DataFrame()
            
            # Merge all return series
            combined_df = pd.concat(all_data, ignore_index=True)
            pivot_df = combined_df.pivot(index='timestamp', columns='symbol', values='returns')
            pivot_df = pivot_df.fillna(0).sort_index()
            
            return pivot_df
            
        except Exception as e:
            logger.error(f"Error fetching returns data: {e}")
            return pd.DataFrame()
    
    def calculate_rolling_correlation(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlation matrix"""
        try:
            if returns_df.empty or len(returns_df) < self.window_size:
                return pd.DataFrame()
            
            # Calculate rolling correlation
            rolling_corr = returns_df.rolling(window=self.window_size).corr()
            
            # Get the most recent correlation matrix
            if not rolling_corr.empty:
                latest_corr = rolling_corr.iloc[-len(returns_df.columns):]
                return latest_corr
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {e}")
            return pd.DataFrame()
    
    def detect_correlation_breaks(self, symbol1: str, symbol2: str, 
                                current_corr: float, lookback_periods: int = 200) -> Dict:
        """Detect significant correlation breaks"""
        try:
            # Fetch historical correlation data
            returns_df = self.fetch_returns_data([symbol1, symbol2], lookback_periods)
            
            if returns_df.empty or len(returns_df) < self.window_size * 2:
                return {'break_detected': False, 'significance': 0.0}
            
            # Calculate historical rolling correlations
            historical_corrs = []
            for i in range(self.window_size, len(returns_df) - self.window_size):
                try:
                    window_data = returns_df.iloc[i-self.window_size:i]
                    if (len(window_data) >= self.window_size and 
                        symbol1 in window_data.columns and 
                        symbol2 in window_data.columns):
                        corr = window_data[symbol1].corr(window_data[symbol2])
                        if not np.isnan(corr):
                            historical_corrs.append(corr)
                except Exception:
                    continue
            
            if len(historical_corrs) < 10:
                return {'break_detected': False, 'significance': 0.0}
            
            # Statistical significance test
            historical_mean = np.mean(historical_corrs)
            historical_std = np.std(historical_corrs)
            
            if historical_std == 0:
                return {'break_detected': False, 'significance': 0.0}
            
            # Z-score of current correlation vs historical
            z_score = abs(current_corr - historical_mean) / historical_std
            
            # Convert to p-value
            p_value = 2 * (1 - stats.norm.cdf(z_score))
            significance = 1 - p_value
            
            # Check if break threshold is exceeded
            correlation_change = abs(current_corr - historical_mean)
            break_detected = (correlation_change > self.break_threshold and 
                             significance > self.significance_level)
            
            return {
                'break_detected': break_detected,
                'significance': significance,
                'historical_mean': historical_mean,
                'current_correlation': current_corr,
                'correlation_change': correlation_change,
                'z_score': z_score,
                'p_value': p_value
            }
            
        except Exception as e:
            logger.error(f"Error detecting correlation breaks: {e}")
            return {'break_detected': False, 'significance': 0.0}
    
    def analyze_sectoral_correlations(self, sectors_config: dict) -> Dict:
        """Analyze sector-wide correlation patterns"""
        try:
            sector_correlations = {}
            
            sectors = sectors_config.get('sectors', {})
            for sector_name, sector_data in sectors.items():
                sector_symbols = sector_data.get('assets', [])
                
                # Get returns for sector assets
                returns_df = self.fetch_returns_data(sector_symbols)
                
                if returns_df.empty or len(returns_df.columns) < 2:
                    continue
                    
                # Calculate sector correlation matrix
                corr_matrix = returns_df.corr()
                
                # Calculate average intra-sector correlation
                n_assets = len(corr_matrix.columns)
                total_corr = 0
                pair_count = 0
                
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        corr_val = corr_matrix.iloc[i, j]
                        if not np.isnan(corr_val):
                            total_corr += corr_val
                            pair_count += 1
                
                avg_correlation = total_corr / pair_count if pair_count > 0 else 0
                
                # Detect sector momentum
                sector_momentum = 'NEUTRAL'
                if avg_correlation > 0.7:
                    sector_momentum = 'HIGH_CORRELATION'
                elif avg_correlation < 0.3:
                    sector_momentum = 'LOW_CORRELATION'
                
                sector_correlations[sector_name] = {
                    'average_correlation': avg_correlation,
                    'momentum': sector_momentum,
                    'asset_count': n_assets,
                    'correlation_matrix': corr_matrix.to_dict()
                }
            
            return sector_correlations
            
        except Exception as e:
            logger.error(f"Error analyzing sectoral correlations: {e}")
            return {}
    
    def detect_market_regime_changes(self, returns_df: pd.DataFrame) -> Dict:
        """Detect market regime changes through correlation analysis"""
        try:
            if returns_df.empty or len(returns_df.columns) < 3:
                return {'regime': 'UNKNOWN', 'confidence': 0.0}
            
            # Calculate average correlation across all pairs
            window_data = returns_df.tail(self.window_size)
            corr_matrix = window_data.corr()
            
            # Get upper triangle correlations (excluding diagonal)
            n_assets = len(corr_matrix.columns)
            correlations = []
            
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        correlations.append(abs(corr_val))
            
            if not correlations:
                return {'regime': 'UNKNOWN', 'confidence': 0.0}
            
            avg_abs_correlation = np.mean(correlations)
            correlation_std = np.std(correlations)
            
            # Determine market regime
            if avg_abs_correlation > 0.8:
                regime = 'CRISIS_MODE'  # High correlation suggests crisis
                confidence = min(0.9, avg_abs_correlation)
            elif avg_abs_correlation > 0.6:
                regime = 'TRENDING_MARKET'  # Moderate correlation
                confidence = min(0.7, avg_abs_correlation)
            elif avg_abs_correlation < 0.3:
                regime = 'STOCK_PICKING_MARKET'  # Low correlation, good for individual stock selection
                confidence = min(0.8, 1 - avg_abs_correlation)
            else:
                regime = 'NORMAL_MARKET'
                confidence = 0.5
            
            return {
                'regime': regime,
                'confidence': confidence,
                'average_correlation': avg_abs_correlation,
                'correlation_std': correlation_std,
                'correlation_distribution': {
                    'min': min(correlations),
                    'max': max(correlations),
                    'median': np.median(correlations),
                    'q25': np.percentile(correlations, 25),
                    'q75': np.percentile(correlations, 75)
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime changes: {e}")
            return {'regime': 'UNKNOWN', 'confidence': 0.0}
    
    async def generate_correlation_alerts(self, symbols: List[str]) -> List[Dict]:
        """Generate correlation break alerts"""
        try:
            alerts = []
            returns_df = self.fetch_returns_data(symbols)
            
            if returns_df.empty:
                return alerts
            
            # Check all pairs for correlation breaks
            current_corr = returns_df.tail(self.window_size).corr()
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol1 in current_corr.columns and symbol2 in current_corr.columns:
                        current_correlation = current_corr.loc[symbol1, symbol2]
                        
                        if not np.isnan(current_correlation):
                            break_analysis = self.detect_correlation_breaks(
                                symbol1, symbol2, current_correlation
                            )
                            
                            if break_analysis['break_detected']:
                                alerts.append({
                                    'type': 'CORRELATION_BREAK',
                                    'symbol1': symbol1,
                                    'symbol2': symbol2,
                                    'current_correlation': current_correlation,
                                    'historical_mean': break_analysis['historical_mean'],
                                    'significance': break_analysis['significance'],
                                    'change_magnitude': break_analysis['correlation_change'],
                                    'message': f"🚨 Correlation Break Alert\\n"
                                              f"{symbol1}-{symbol2} correlation: "
                                              f"{break_analysis['historical_mean']:.2f} → {current_correlation:.2f}\\n"
                                              f"Significance: {break_analysis['significance']:.0%}\\n"
                                              f"Potential Impact: Market divergence signal",
                                    'timestamp': datetime.now().isoformat()
                                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating correlation alerts: {e}")
            return []
    
    async def run_correlation_analysis(self, symbols: List[str], sectors_config: dict) -> Dict:
        """Run complete correlation analysis"""
        try:
            # Fetch data
            returns_df = self.fetch_returns_data(symbols)
            
            if returns_df.empty:
                return {'error': 'Insufficient data for correlation analysis'}
            
            # Calculate current correlation matrix
            current_correlations = self.calculate_rolling_correlation(returns_df)
            
            # Detect correlation breaks
            correlation_alerts = await self.generate_correlation_alerts(symbols)
            
            # Analyze sectoral patterns
            sector_analysis = self.analyze_sectoral_correlations(sectors_config)
            
            # Detect market regime
            regime_analysis = self.detect_market_regime_changes(returns_df)
            
            # Cross-market analysis
            cross_market_correlations = {}
            for pair in self.cross_market_pairs:
                symbols_pair = pair.split('/')
                if len(symbols_pair) == 2 and all(s in returns_df.columns for s in symbols_pair):
                    cross_corr = returns_df[symbols_pair].corr().iloc[0, 1]
                    cross_market_correlations[pair] = cross_corr
            
            return {
                'correlation_matrix': current_correlations.to_dict() if not current_correlations.empty else {},
                'correlation_alerts': correlation_alerts,
                'sector_analysis': sector_analysis,
                'market_regime': regime_analysis,
                'cross_market_correlations': cross_market_correlations,
                'analysis_timestamp': datetime.now().isoformat(),
                'symbols_analyzed': len(symbols),
                'data_quality': {
                    'total_observations': len(returns_df),
                    'missing_data_ratio': returns_df.isnull().sum().sum() / (len(returns_df) * len(returns_df.columns))
                }
            }
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {'error': str(e), 'analysis_timestamp': datetime.now().isoformat()}