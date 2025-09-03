"""
Advanced Models: SVAR and State-Space for macro and multi-asset analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class AdvancedModels:
    def __init__(self):
        self.var_available = False
        self.kalman_available = False
        
        # Try to import statsmodels components
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            self.VAR = VAR
            self.var_available = True
        except ImportError:
            logger.warning("statsmodels VAR not available. Install with: pip install statsmodels")
        
        try:
            from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
            self.KalmanFilter = KalmanFilter
            self.kalman_available = True
        except ImportError:
            logger.warning("statsmodels KalmanFilter not available. Install with: pip install statsmodels")

    def svar_effect(self, df: pd.DataFrame) -> Dict:
        """Fit a VAR model and compute effect sizes."""
        if not self.var_available:
            logger.warning("VAR model not available - returning mock results")
            return {
                'effect_size': 0.1,
                'significant': False,
                'error': 'VAR model not available'
            }
        
        try:
            # Ensure we have at least 2 columns for VAR
            if df.shape[1] < 2:
                logger.error("VAR requires at least 2 time series")
                return {'effect_size': 0, 'significant': False, 'error': 'insufficient_data'}
            
            # Drop any NaN values
            df_clean = df.dropna()
            
            if len(df_clean) < 10:  # Need minimum observations
                logger.error("Insufficient observations for VAR model")
                return {'effect_size': 0, 'significant': False, 'error': 'insufficient_observations'}
            
            model = self.VAR(df_clean)
            res = model.fit(maxlags=min(5, len(df_clean)//4))
            irf = res.irf(periods=10)
            
            # Use IRF to measure impact of shock in first variable on second
            effect = float(irf.irfs[:,1,0].mean()) if irf.irfs.shape[1] > 1 else 0
            
            return {
                'effect_size': effect, 
                'significant': abs(effect) > 0.1,
                'model_aic': res.aic,
                'periods_analyzed': len(df_clean)
            }
        except Exception as e:
            logger.error(f"SVAR error: {e}")
            return {'effect_size': 0, 'significant': False, 'error': str(e)}

    def kalman_estimate(self, series: pd.Series) -> pd.Series:
        """Apply Kalman filter to estimate latent state (e.g., liquidity)"""
        if not self.kalman_available:
            logger.warning("Kalman filter not available - returning original series")
            return series
        
        try:
            # Simple Kalman filter setup for univariate time series
            kf = self.KalmanFilter(k_endog=1, k_states=1)
            kf.bind(series.values.reshape(-1,1))
            res = kf.filter()
            
            return pd.Series(res.filtered_state[0], index=series.index)
        except Exception as e:
            logger.error(f"Kalman filter error: {e}")
            return series

    def detect_regime_changes(self, series: pd.Series, window: int = 20) -> Dict:
        """
        Detect regime changes using rolling statistics
        """
        try:
            # Calculate rolling mean and std
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            
            # Detect significant changes in volatility (regime changes)
            std_changes = rolling_std.pct_change().abs()
            threshold = std_changes.quantile(0.9)  # Top 10% of changes
            
            regime_changes = std_changes > threshold
            change_points = series.index[regime_changes].tolist()
            
            return {
                'regime_changes_detected': len(change_points),
                'change_points': [str(cp) for cp in change_points[-5:]],  # Last 5 changes
                'current_volatility': float(rolling_std.iloc[-1]) if not rolling_std.empty else 0,
                'volatility_trend': 'increasing' if len(rolling_std) > 1 and rolling_std.iloc[-1] > rolling_std.iloc[-2] else 'stable'
            }
        except Exception as e:
            logger.error(f"Regime change detection error: {e}")
            return {
                'regime_changes_detected': 0,
                'change_points': [],
                'current_volatility': 0,
                'volatility_trend': 'unknown'
            }

    def correlation_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Advanced correlation analysis with time-varying correlations
        """
        try:
            if df.shape[1] < 2:
                return {'error': 'Need at least 2 assets for correlation analysis'}
            
            # Static correlation matrix
            corr_matrix = df.corr()
            
            # Rolling correlation (30-period window)
            window = min(30, len(df) // 4)
            rolling_corr = df.rolling(window=window).corr()
            
            # Extract key metrics
            asset_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    static_corr = corr_matrix.iloc[i, j]
                    
                    asset_pairs.append({
                        'pair': f"{col1}-{col2}",
                        'static_correlation': float(static_corr),
                        'correlation_strength': 'strong' if abs(static_corr) > 0.7 else 'moderate' if abs(static_corr) > 0.3 else 'weak'
                    })
            
            return {
                'asset_pairs': asset_pairs,
                'max_correlation': float(corr_matrix.max().max()),
                'min_correlation': float(corr_matrix.min().min()),
                'analysis_period': len(df)
            }
        except Exception as e:
            logger.error(f"Correlation analysis error: {e}")
            return {'error': str(e)}