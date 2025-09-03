"""
Regime Engine v1
Detects market regimes (risk-on/off, liquidity, volatility) with hysteresis
"""

import json
import sqlite3
import os
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RegimeState:
    risk: str  # 'on', 'off', 'neutral'
    liquidity: str  # 'abundant', 'tight', 'neutral'
    volatility: str  # 'low', 'high', 'normal'
    trend: str  # 'bull', 'bear', 'sideways'
    confidence: float
    last_updated: str

class RegimeEngine:
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
        
        # Regime detection parameters
        self.thresholds = {
            'vix_low': 16,
            'vix_high': 20,
            'vol_low': 0.15,
            'vol_high': 0.25,
            'dxy_momentum': 0.02,
            'funding_rate_threshold': 0.0001
        }
        
        # Hysteresis parameters (prevent rapid switching)
        self.hysteresis_periods = 2  # Require 2 cycles before regime change
        self.regime_cache = {}
        
        # Load current state
        self.current_regime = self.load_state() or RegimeState(
            risk='neutral', liquidity='neutral', volatility='normal',
            trend='sideways', confidence=0.5, last_updated=datetime.now().isoformat()
        )

    def init_tables(self):
        """Initialize database tables for regime tracking"""
        if self.database_url:
            # PostgreSQL
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regimes (
                    timestamp TIMESTAMP PRIMARY KEY,
                    risk TEXT,
                    liquidity TEXT,
                    volatility TEXT,
                    trend TEXT,
                    confidence REAL,
                    indicators JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_indicators (
                    timestamp TIMESTAMP,
                    indicator_name TEXT,
                    value REAL,
                    normalized_value REAL,
                    regime_signal TEXT,
                    PRIMARY KEY (timestamp, indicator_name)
                )
            ''')
            self.conn.commit()
        else:
            # SQLite
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS regimes (
                    timestamp TEXT PRIMARY KEY,
                    risk TEXT,
                    liquidity TEXT,
                    volatility TEXT,
                    trend TEXT,
                    confidence REAL,
                    indicators TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS regime_indicators (
                    timestamp TEXT,
                    indicator_name TEXT,
                    value REAL,
                    normalized_value REAL,
                    regime_signal TEXT,
                    PRIMARY KEY (timestamp, indicator_name)
                )
            ''')
            self.conn.commit()

    def detect_current_regime(self) -> RegimeState:
        """Detect current market regime using multiple indicators"""
        try:
            indicators = self._gather_indicators()
            
            # Risk appetite detection
            risk_state = self._detect_risk_regime(indicators)
            
            # Liquidity conditions
            liquidity_state = self._detect_liquidity_regime(indicators)
            
            # Volatility regime
            volatility_state = self._detect_volatility_regime(indicators)
            
            # Trend regime
            trend_state = self._detect_trend_regime(indicators)
            
            # Calculate overall confidence
            confidence = self._calculate_regime_confidence(indicators)
            
            new_regime = RegimeState(
                risk=risk_state,
                liquidity=liquidity_state,
                volatility=volatility_state,
                trend=trend_state,
                confidence=confidence,
                last_updated=datetime.now().isoformat()
            )
            
            # Apply hysteresis
            new_regime = self._apply_hysteresis(new_regime)
            
            # Store regime
            self._store_regime(new_regime, indicators)
            
            self.current_regime = new_regime
            logger.info(f"Regime detected: Risk={risk_state}, Liquidity={liquidity_state}, Vol={volatility_state}, Trend={trend_state}")
            
            return new_regime
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return self.current_regime

    def _gather_indicators(self) -> Dict:
        """Gather indicators from various sources"""
        indicators = {}
        
        try:
            # VIX proxy (crypto volatility)
            if self.binance_scanner:
                btc_data = self.binance_scanner.get_market_data()
                if 'BTC' in btc_data:
                    indicators['btc_volatility'] = btc_data['BTC'].get('volume_change_24h', 0) / 100
                
            # Indian market indicators
            if self.equity_scanner:
                # Use Nifty as risk proxy
                nifty_data = self.equity_scanner.get_market_data(symbols=['^NSEI'], periods=5)
                if not nifty_data.empty:
                    indicators['nifty_change'] = nifty_data['close'].pct_change().iloc[-1] if len(nifty_data) > 1 else 0
                    indicators['nifty_volatility'] = nifty_data['close'].pct_change().std() * np.sqrt(252)
            
            # Fallback synthetic indicators for demonstration
            if not indicators:
                indicators = self._generate_synthetic_indicators()
                
        except Exception as e:
            logger.error(f"Error gathering indicators: {e}")
            indicators = self._generate_synthetic_indicators()
        
        return indicators

    def _generate_synthetic_indicators(self) -> Dict:
        """Generate synthetic indicators for demonstration"""
        return {
            'vix_proxy': np.random.normal(18, 3),
            'credit_spreads': np.random.normal(150, 30),
            'funding_rates': np.random.normal(0.0001, 0.0002),
            'flow_momentum': np.random.normal(0, 0.02),
            'cross_asset_correlation': np.random.uniform(0.3, 0.8)
        }

    def _detect_risk_regime(self, indicators: Dict) -> str:
        """Detect risk-on/off regime"""
        try:
            risk_signals = []
            
            # VIX-like indicator
            vix_proxy = indicators.get('vix_proxy', 18)
            if vix_proxy < self.thresholds['vix_low']:
                risk_signals.append('on')
            elif vix_proxy > self.thresholds['vix_high']:
                risk_signals.append('off')
            else:
                risk_signals.append('neutral')
            
            # Credit spreads
            credit_spreads = indicators.get('credit_spreads', 150)
            if credit_spreads < 120:
                risk_signals.append('on')
            elif credit_spreads > 200:
                risk_signals.append('off')
            else:
                risk_signals.append('neutral')
            
            # Cross-asset correlation
            correlation = indicators.get('cross_asset_correlation', 0.5)
            if correlation < 0.4:
                risk_signals.append('on')  # Low correlation = risk-on
            elif correlation > 0.7:
                risk_signals.append('off')  # High correlation = risk-off
            else:
                risk_signals.append('neutral')
            
            # Majority vote
            risk_on_count = risk_signals.count('on')
            risk_off_count = risk_signals.count('off')
            
            if risk_on_count > risk_off_count:
                return 'on'
            elif risk_off_count > risk_on_count:
                return 'off'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error detecting risk regime: {e}")
            return 'neutral'

    def _detect_liquidity_regime(self, indicators: Dict) -> str:
        """Detect liquidity conditions"""
        try:
            # Funding rates
            funding_rate = indicators.get('funding_rates', 0)
            
            if abs(funding_rate) < self.thresholds['funding_rate_threshold']:
                return 'abundant'
            elif funding_rate > self.thresholds['funding_rate_threshold'] * 3:
                return 'tight'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error detecting liquidity regime: {e}")
            return 'neutral'

    def _detect_volatility_regime(self, indicators: Dict) -> str:
        """Detect volatility regime"""
        try:
            volatility_indicators = []
            
            # BTC volatility proxy
            btc_vol = indicators.get('btc_volatility', 0.2)
            volatility_indicators.append(btc_vol)
            
            # Nifty volatility
            nifty_vol = indicators.get('nifty_volatility', 0.2)
            volatility_indicators.append(nifty_vol)
            
            avg_vol = np.mean(volatility_indicators)
            
            if avg_vol < self.thresholds['vol_low']:
                return 'low'
            elif avg_vol > self.thresholds['vol_high']:
                return 'high'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return 'normal'

    def _detect_trend_regime(self, indicators: Dict) -> str:
        """Detect trend regime"""
        try:
            # Use momentum indicators
            flow_momentum = indicators.get('flow_momentum', 0)
            nifty_change = indicators.get('nifty_change', 0)
            
            avg_momentum = np.mean([flow_momentum, nifty_change])
            
            if avg_momentum > 0.02:  # 2% positive momentum
                return 'bull'
            elif avg_momentum < -0.02:  # 2% negative momentum
                return 'bear'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Error detecting trend regime: {e}")
            return 'sideways'

    def _calculate_regime_confidence(self, indicators: Dict) -> float:
        """Calculate confidence in regime detection"""
        try:
            # Base confidence on data quality and consistency
            data_completeness = len(indicators) / 5  # Expected 5 indicators
            
            # Add noise penalty
            noise_penalty = 0
            for value in indicators.values():
                if isinstance(value, (int, float)) and abs(value) > 10:  # Extreme values
                    noise_penalty += 0.1
            
            confidence = min(0.95, max(0.3, data_completeness - noise_penalty))
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _apply_hysteresis(self, new_regime: RegimeState) -> RegimeState:
        """Apply hysteresis to prevent rapid regime switching"""
        try:
            if not hasattr(self, 'regime_history'):
                self.regime_history = []
            
            # Add to history
            self.regime_history.append({
                'regime': (new_regime.risk, new_regime.liquidity, new_regime.volatility, new_regime.trend),
                'timestamp': datetime.now()
            })
            
            # Keep only recent history
            cutoff = datetime.now() - timedelta(hours=2)
            self.regime_history = [r for r in self.regime_history if r['timestamp'] > cutoff]
            
            # Check for consistency
            if len(self.regime_history) >= self.hysteresis_periods:
                recent_regimes = [r['regime'] for r in self.regime_history[-self.hysteresis_periods:]]
                
                # If not consistent, return current regime
                if len(set(recent_regimes)) > 1:
                    return self.current_regime
            
            return new_regime
            
        except Exception as e:
            logger.error(f"Error applying hysteresis: {e}")
            return new_regime

    def _store_regime(self, regime: RegimeState, indicators: Dict):
        """Store regime state in database"""
        try:
            if self.database_url:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO regimes (timestamp, risk, liquidity, volatility, trend, confidence, indicators)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (timestamp) DO UPDATE SET
                    risk = EXCLUDED.risk,
                    liquidity = EXCLUDED.liquidity,
                    volatility = EXCLUDED.volatility,
                    trend = EXCLUDED.trend,
                    confidence = EXCLUDED.confidence
                ''', (
                    regime.last_updated, regime.risk, regime.liquidity,
                    regime.volatility, regime.trend, regime.confidence,
                    json.dumps(indicators)
                ))
            else:
                self.conn.execute('''
                    INSERT OR REPLACE INTO regimes (timestamp, risk, liquidity, volatility, trend, confidence, indicators)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    regime.last_updated, regime.risk, regime.liquidity,
                    regime.volatility, regime.trend, regime.confidence,
                    json.dumps(indicators)
                ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing regime: {e}")

    def load_state(self) -> Optional[RegimeState]:
        """Load latest regime state"""
        try:
            if self.database_url:
                cursor = self.conn.cursor()
                cursor.execute('SELECT * FROM regimes ORDER BY timestamp DESC LIMIT 1')
            else:
                cursor = self.conn.execute('SELECT * FROM regimes ORDER BY timestamp DESC LIMIT 1')
            
            row = cursor.fetchone()
            if row:
                return RegimeState(
                    risk=row[1],
                    liquidity=row[2],
                    volatility=row[3],
                    trend=row[4],
                    confidence=row[5],
                    last_updated=row[0]
                )
            return None
            
        except Exception as e:
            logger.error(f"Error loading regime state: {e}")
            return None

    def get_regime_history(self, hours_back: int = 24) -> List[Dict]:
        """Get regime history"""
        try:
            cutoff = datetime.now() - timedelta(hours=hours_back)
            
            if self.database_url:
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT * FROM regimes WHERE timestamp >= %s ORDER BY timestamp DESC
                ''', (cutoff,))
            else:
                cursor = self.conn.execute('''
                    SELECT * FROM regimes WHERE timestamp >= ? ORDER BY timestamp DESC
                ''', (cutoff.isoformat(),))
            
            return [dict(zip(['timestamp', 'risk', 'liquidity', 'volatility', 'trend', 'confidence', 'indicators'], row))
                    for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting regime history: {e}")
            return []

    def get_current_regime(self) -> RegimeState:
        """Get current regime state"""
        return self.current_regime

    def is_regime_stable(self, hours_back: int = 4) -> bool:
        """Check if regime has been stable recently"""
        try:
            history = self.get_regime_history(hours_back)
            if len(history) < 2:
                return True
            
            # Check consistency across different dimensions
            risks = set(r['risk'] for r in history)
            liquidity = set(r['liquidity'] for r in history)
            volatility = set(r['volatility'] for r in history)
            
            # Stable if no more than 1 regime change per dimension
            return len(risks) <= 2 and len(liquidity) <= 2 and len(volatility) <= 2
            
        except Exception as e:
            logger.error(f"Error checking regime stability: {e}")
            return True