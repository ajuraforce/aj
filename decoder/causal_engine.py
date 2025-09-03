"""
Causal Engine for testing lead-lag relationships and building causal hypotheses
"""

import json
import sqlite3
import os
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from sklearn.metrics import mutual_info_score

# Try to import Flask models for enhanced integration
try:
    from flask import current_app
    from models import db, CausalHypothesis, CausalTest
    FLASK_MODELS_AVAILABLE = True
except ImportError:
    FLASK_MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CausalCard:
    hypothesis: str
    support: Dict  # Statistical evidence
    regime_dependency: List[str]  # Which regimes this works in
    confidence: float
    status: str  # 'significant', 'weak', 'rejected'
    last_updated: str

class CausalEngine:
    def __init__(self, db_path='patterns.db', equity_scanner=None, binance_scanner=None):
        # Use PostgreSQL if available, fallback to SQLite
        self.database_url = os.getenv('DATABASE_URL')
        if self.database_url:
            self.conn = psycopg2.connect(self.database_url)
        else:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.equity_scanner = equity_scanner
        self.binance_scanner = binance_scanner
        self.init_tables()
        
        # Parameters
        self.max_lag = 5
        self.min_observations = 50
        self.significance_threshold = 0.05
        self.effect_size_threshold = 0.1

    def init_tables(self):
        """Initialize database tables for causal analysis"""
        if self.database_url:
            # PostgreSQL
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS causal_hypotheses (
                    id TEXT PRIMARY KEY,
                    x_variable TEXT,
                    y_variable TEXT,
                    hypothesis TEXT,
                    granger_p REAL,
                    lead_lag_minutes REAL,
                    effect_size REAL,
                    confidence REAL,
                    regime_dependency JSONB,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_tested TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS causal_tests (
                    test_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT,
                    test_type TEXT,
                    result TEXT,
                    p_value REAL,
                    effect_size REAL,
                    sample_size INTEGER,
                    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
        else:
            # SQLite
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS causal_hypotheses (
                    id TEXT PRIMARY KEY,
                    x_variable TEXT,
                    y_variable TEXT,
                    hypothesis TEXT,
                    granger_p REAL,
                    lead_lag_minutes REAL,
                    effect_size REAL,
                    confidence REAL,
                    regime_dependency TEXT,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_tested TIMESTAMP
                )
            ''')
            
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS causal_tests (
                    test_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT,
                    test_type TEXT,
                    result TEXT,
                    p_value REAL,
                    effect_size REAL,
                    sample_size INTEGER,
                    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()

    def screen_pair(self, x_series: pd.Series, y_series: pd.Series, 
                   x_name: str = "X", y_name: str = "Y") -> Dict:
        """Screen causal relationship between two time series"""
        
        try:
            # Align series and remove NaN
            data = pd.concat([x_series, y_series], axis=1).dropna()
            
            if len(data) < self.min_observations:
                return {'error': 'Insufficient observations', 'sample_size': len(data)}
            
            # Prepare data for analysis
            data.columns = ['x', 'y']
            
            # 1. Lead-lag Cross-correlation
            lead_lag_results = self._lead_lag_analysis(data['x'], data['y'])
            
            # 2. Mutual Information
            mi_score = self._mutual_information(data['x'], data['y'])
            
            # 3. Effect Size
            effect_size = self._calculate_effect_size(data['x'], data['y'])
            
            # 4. Persistence Test
            persistence = self._test_persistence(data)
            
            # Combine results
            confidence = self._calculate_confidence(
                lead_lag_results, mi_score, effect_size, persistence
            )
            
            return {
                'x_variable': x_name,
                'y_variable': y_name,
                'lead_lag_minutes': lead_lag_results['lag_minutes'],
                'lead_lag_correlation': lead_lag_results['max_correlation'],
                'mutual_info': mi_score,
                'effect_size': effect_size,
                'persistence_score': persistence,
                'confidence': confidence,
                'sample_size': len(data),
                'status': 'significant' if confidence > 0.6 else 'weak'
            }
            
        except Exception as e:
            logger.error(f"Error in screen_pair: {e}")
            return {'error': str(e)}

    def _lead_lag_analysis(self, x: pd.Series, y: pd.Series, 
                          max_lag_periods: int = 20) -> Dict:
        """Analyze lead-lag relationship"""
        try:
            correlations = []
            lags = range(-max_lag_periods, max_lag_periods + 1)
            
            for lag in lags:
                if lag == 0:
                    corr = x.corr(y)
                elif lag > 0:
                    # x leads y
                    corr = x[:-lag].corr(y[lag:]) if lag < len(x) else 0
                else:
                    # y leads x
                    corr = x[-lag:].corr(y[:lag]) if -lag < len(y) else 0
                
                correlations.append(corr)
            
            # Find maximum absolute correlation and its lag
            abs_correlations = [abs(c) if not np.isnan(c) else 0 for c in correlations]
            max_idx = np.argmax(abs_correlations)
            best_lag = lags[max_idx]
            max_corr = correlations[max_idx]
            
            # Convert to minutes (assuming data is minute-level)
            lag_minutes = best_lag * 1  # 1 minute per period
            
            return {
                'lag_periods': best_lag,
                'lag_minutes': lag_minutes,
                'max_correlation': max_corr,
                'all_correlations': correlations
            }
            
        except Exception as e:
            logger.error(f"Lead-lag analysis error: {e}")
            return {'lag_periods': 0, 'lag_minutes': 0, 'max_correlation': 0}

    def _mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate mutual information between series"""
        try:
            # Discretize continuous variables for MI calculation
            x_disc = pd.cut(x, bins=10, labels=False)
            y_disc = pd.cut(y, bins=10, labels=False)
            
            # Remove NaN
            valid_idx = ~(np.isnan(x_disc) | np.isnan(y_disc))
            x_disc = x_disc[valid_idx]
            y_disc = y_disc[valid_idx]
            
            if len(x_disc) < 10:
                return 0.0
            
            mi = mutual_info_score(x_disc, y_disc)
            return mi
            
        except Exception as e:
            logger.error(f"Mutual information error: {e}")
            return 0.0

    def _calculate_effect_size(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate effect size (standardized regression coefficient)"""
        try:
            # Simple linear regression
            from sklearn.linear_model import LinearRegression
            
            X = x.values.reshape(-1, 1)
            y_vals = y.values
            
            # Remove NaN
            valid_idx = ~(np.isnan(X.flatten()) | np.isnan(y_vals))
            X = X[valid_idx]
            y_vals = y_vals[valid_idx]
            
            if len(X) < 10:
                return 0.0
            
            model = LinearRegression()
            model.fit(X, y_vals)
            
            # Standardized coefficient (beta)
            x_std = np.std(X)
            y_std = np.std(y_vals)
            
            if x_std == 0 or y_std == 0:
                return 0.0
            
            beta = model.coef_[0] * (x_std / y_std)
            return abs(beta)
            
        except Exception as e:
            logger.error(f"Effect size calculation error: {e}")
            return 0.0

    def _test_persistence(self, data: pd.DataFrame) -> float:
        """Test if relationship persists across different windows"""
        try:
            window_size = min(50, len(data) // 3)
            if window_size < 20:
                return 0.5
            
            correlations = []
            for i in range(0, len(data) - window_size, window_size // 2):
                window_data = data.iloc[i:i + window_size]
                corr = window_data['x'].corr(window_data['y'])
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if not correlations:
                return 0.5
            
            # Consistency = 1 - coefficient of variation
            mean_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            
            if mean_corr == 0:
                return 0.5
            
            cv = abs(std_corr / mean_corr)
            persistence = max(0, 1 - cv)
            
            return min(1.0, persistence)
            
        except Exception as e:
            logger.error(f"Persistence test error: {e}")
            return 0.5

    def _calculate_confidence(self, lead_lag_results: Dict, 
                            mi_score: float, effect_size: float, persistence: float) -> float:
        """Calculate overall confidence in causal relationship"""
        
        # Component scores (0-1)
        correlation_score = abs(lead_lag_results['max_correlation'])
        
        mi_score_norm = min(1.0, mi_score / 0.5)  # Normalize MI
        
        effect_score = min(1.0, effect_size / 0.3)  # Normalize effect size
        
        # Weighted combination
        confidence = (
            0.40 * correlation_score +
            0.25 * mi_score_norm +
            0.25 * effect_score +
            0.10 * persistence
        )
        
        return min(0.95, confidence)  # Cap at 95%

    def build_causal_card(self, event_data: Dict, asset_symbol: str) -> CausalCard:
        """Build causal hypothesis card for event -> asset relationship"""
        
        try:
            # Get time series data
            event_series = self._get_event_intensity_series(event_data)
            asset_series = self._get_asset_price_series(asset_symbol)
            
            # Screen the relationship
            results = self.screen_pair(event_series, asset_series, 
                                     event_data.get('type', 'EVENT'), asset_symbol)
            
            if 'error' in results:
                return CausalCard(
                    hypothesis=f"{event_data.get('type', 'EVENT')} -> {asset_symbol}",
                    support={'error': results['error']},
                    regime_dependency=[],
                    confidence=0.0,
                    status='error',
                    last_updated=datetime.now().isoformat()
                )
            
            # Determine regime dependency
            regime_dependency = self._determine_regime_dependency(results)
            
            hypothesis = f"{results['x_variable']} -> {results['y_variable']}"
            
            card = CausalCard(
                hypothesis=hypothesis,
                support=results,
                regime_dependency=regime_dependency,
                confidence=results['confidence'],
                status=results['status'],
                last_updated=datetime.now().isoformat()
            )
            
            # Store in database
            self._store_causal_card(card)
            
            return card
            
        except Exception as e:
            logger.error(f"Error building causal card: {e}")
            return CausalCard(
                hypothesis=f"ERROR -> {asset_symbol}",
                support={'error': str(e)},
                regime_dependency=[],
                confidence=0.0,
                status='error',
                last_updated=datetime.now().isoformat()
            )

    def _get_event_intensity_series(self, event_data: Dict) -> pd.Series:
        """Convert event data to time series (intensity over time)"""
        # This is a simplified version - would need proper event aggregation
        try:
            # Get events of same type over time
            event_type = event_data.get('type', 'GENERAL_EVENT')
            
            # Create dummy series for demonstration
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            intensities = np.random.exponential(0.1, 100)  # Event intensity
            return pd.Series(intensities, index=dates)
            
        except Exception as e:
            logger.error(f"Error getting event series: {e}")
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            return pd.Series(0.0, index=dates)

    def _get_asset_price_series(self, asset_symbol: str) -> pd.Series:
        """Get asset price series"""
        try:
            # Create synthetic price series for demonstration
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            base_price = 100
            changes = np.random.normal(0, 0.01, 100)
            prices = [base_price]
            for change in changes[1:]:
                prices.append(prices[-1] * (1 + change))
            return pd.Series(prices, index=dates)
            
        except Exception as e:
            logger.error(f"Error getting asset series: {e}")
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            return pd.Series(100.0, index=dates)

    def _determine_regime_dependency(self, results: Dict) -> List[str]:
        """Determine which regimes this relationship depends on"""
        dependency = []
        
        # High correlation suggests regime-independent
        if abs(results.get('lead_lag_correlation', 0)) > 0.7:
            dependency.append('regime_independent')
        
        # High effect size suggests fundamental relationship
        if results.get('effect_size', 0) > 0.3:
            dependency.append('fundamental')
        else:
            dependency.append('sentiment_driven')
        
        # Lead-lag suggests flow-based
        if abs(results.get('lag_minutes', 0)) > 30:
            dependency.append('flow_driven')
        
        return dependency

    def _store_causal_card(self, card: CausalCard):
        """Store causal card in database"""
        try:
            # Extract variables from hypothesis
            parts = card.hypothesis.split(' -> ')
            x_var = parts[0] if len(parts) > 0 else ''
            y_var = parts[1] if len(parts) > 1 else ''
            
            card_id = f"causal_{hash(card.hypothesis) % 10000}_{datetime.now().strftime('%Y%m%d')}"
            
            if self.database_url:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO causal_hypotheses (id, x_variable, y_variable, hypothesis, 
                    granger_p, lead_lag_minutes, effect_size, confidence, regime_dependency, 
                    status, created_at, last_tested)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    status = EXCLUDED.status,
                    last_tested = EXCLUDED.last_tested
                ''', (
                    card_id, x_var, y_var, card.hypothesis,
                    card.support.get('granger_p', 1.0),
                    card.support.get('lead_lag_minutes', 0),
                    card.support.get('effect_size', 0),
                    card.confidence,
                    json.dumps(card.regime_dependency),
                    card.status,
                    datetime.now(),
                    datetime.now()
                ))
            else:
                self.conn.execute('''
                    INSERT OR REPLACE INTO causal_hypotheses VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    card_id, x_var, y_var, card.hypothesis,
                    card.support.get('granger_p', 1.0),
                    card.support.get('lead_lag_minutes', 0),
                    card.support.get('effect_size', 0),
                    card.confidence,
                    json.dumps(card.regime_dependency),
                    card.status,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing causal card: {e}")

    def get_active_hypotheses(self, min_confidence: float = 0.6) -> List[CausalCard]:
        """Get active causal hypotheses above confidence threshold"""
        try:
            if self.database_url:
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT * FROM causal_hypotheses 
                    WHERE confidence >= %s AND status = 'significant'
                    ORDER BY confidence DESC
                ''', (min_confidence,))
            else:
                cursor = self.conn.execute('''
                    SELECT * FROM causal_hypotheses 
                    WHERE confidence >= ? AND status = 'significant'
                    ORDER BY confidence DESC
                ''', (min_confidence,))
            
            cards = []
            for row in cursor.fetchall():
                card = CausalCard(
                    hypothesis=row[3],
                    support={
                        'granger_p': row[4],
                        'lead_lag_minutes': row[5],
                        'effect_size': row[6]
                    },
                    regime_dependency=json.loads(row[8]) if row[8] else [],
                    confidence=row[7],
                    status=row[9],
                    last_updated=row[11]
                )
                cards.append(card)
            
            return cards
            
        except Exception as e:
            logger.error(f"Error getting active hypotheses: {e}")
            return []

    def test_hypothesis_batch(self, asset_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Test causal relationships for a batch of asset pairs"""
        try:
            results = []
            
            for pair in asset_pairs:
                x_asset, y_asset = pair
                
                try:
                    # Get price series for both assets
                    x_series = self._get_asset_price_series(x_asset)
                    y_series = self._get_asset_price_series(y_asset)
                    
                    # Test both directions (X->Y and Y->X)
                    result_xy = self.screen_pair(x_series, y_series, x_asset, y_asset)
                    result_yx = self.screen_pair(y_series, x_series, y_asset, x_asset)
                    
                    # Take the stronger relationship
                    if result_xy.get('confidence', 0) >= result_yx.get('confidence', 0):
                        best_result = result_xy
                        direction = f"{x_asset} -> {y_asset}"
                    else:
                        best_result = result_yx
                        direction = f"{y_asset} -> {x_asset}"
                    
                    results.append({
                        'pair': pair,
                        'direction': direction,
                        'confidence': best_result.get('confidence', 0),
                        'lead_lag_minutes': best_result.get('lead_lag_minutes', 0),
                        'effect_size': best_result.get('effect_size', 0),
                        'status': best_result.get('status', 'weak'),
                        'mutual_info': best_result.get('mutual_info', 0),
                        'sample_size': best_result.get('sample_size', 0)
                    })
                    
                except Exception as e:
                    logger.error(f"Error testing pair {pair}: {e}")
                    results.append({
                        'pair': pair,
                        'direction': f"{x_asset} -> {y_asset}",
                        'confidence': 0.0,
                        'error': str(e),
                        'status': 'error'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in test_hypothesis_batch: {e}")
            return []