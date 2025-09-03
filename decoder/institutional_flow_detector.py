"""
Institutional Flow Detection Engine
Detects large player movements and institutional patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

class InstitutionalFlowDetector:
    def __init__(self, config: dict, db_path: str):
        self.config = config
        self.db_path = db_path
        
        # Institutional detection thresholds
        self.volume_spike_threshold = 3.0  # 3x average volume
        self.price_impact_threshold = 0.02  # 2% price impact
        self.block_size_threshold = 10000  # Minimum block size in USD
        self.accumulation_period = 24  # Hours for accumulation detection
        
    def cross_market_confirmation(self, signals: Dict) -> Dict:
        """Cross-market confirmation of institutional signals"""
        try:
            # Calculate composite confirmation score
            signal_values = list(signals.values())
            confirmation_score = np.mean(signal_values) if signal_values else 0.0
            
            # Check signal alignment
            strong_signals = sum(1 for v in signal_values if v > 0.7)
            weak_signals = sum(1 for v in signal_values if v < 0.3)
            
            confirmation_strength = 'LOW'
            if strong_signals >= 3:
                confirmation_strength = 'HIGH'
            elif strong_signals >= 2:
                confirmation_strength = 'MEDIUM'
            
            return {
                'confirmation_score': confirmation_score,
                'confirmation_strength': confirmation_strength,
                'signal_alignment': strong_signals - weak_signals,
                'consensus': confirmation_score > 0.6
            }
        except Exception as e:
            logger.error(f"Error in cross market confirmation: {e}")
            return {'confirmation_score': 0.0, 'confirmation_strength': 'LOW', 'consensus': False}
        
    def get_volume_data(self, symbol: str, periods: int = 200) -> pd.DataFrame:
        """Fetch volume and price data for institutional analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT timestamp, close, volume, open, high, low
            FROM market_data
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=[symbol, periods])
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')
            
            # Calculate additional metrics
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3  # Simplified VWAP
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching volume data for {symbol}: {e}")
            return pd.DataFrame()
    
    def detect_volume_anomalies(self, symbol: str) -> Dict:
        """Detect unusual volume patterns indicating institutional activity"""
        try:
            data = self.get_volume_data(symbol)
            
            if data.empty or len(data) < 50:
                return {'anomaly_detected': False, 'volume_score': 0.0}
            
            # Calculate volume statistics
            recent_volume = data['volume'].tail(10).mean()
            historical_volume = data['volume'].head(-10).mean()
            volume_std = data['volume'].std()
            
            # Z-score calculation
            current_volume = data['volume'].iloc[-1]
            volume_zscore = (current_volume - historical_volume) / volume_std if volume_std > 0 else 0
            
            # Detect sustained volume increase
            volume_trend = data['volume'].tail(5).mean() / data['volume'].head(-5).mean() if len(data) > 10 else 1
            
            # Large block detection
            price = data['close'].iloc[-1]
            estimated_trade_value = current_volume * price
            large_block_detected = estimated_trade_value > self.block_size_threshold
            
            # Institutional activity score
            volume_score = min(1.0, (abs(volume_zscore) / 3.0) + (volume_trend - 1) * 2)
            
            anomaly_detected = (volume_zscore > self.volume_spike_threshold or 
                               volume_trend > 2.0 or large_block_detected)
            
            return {
                'anomaly_detected': anomaly_detected,
                'volume_score': volume_score,
                'volume_zscore': volume_zscore,
                'volume_trend': volume_trend,
                'estimated_trade_value': estimated_trade_value,
                'large_block_detected': large_block_detected
            }
            
        except Exception as e:
            logger.error(f"Error detecting volume anomalies for {symbol}: {e}")
            return {'anomaly_detected': False, 'volume_score': 0.0}
    
    def analyze_price_impact(self, symbol: str) -> Dict:
        """Analyze price impact patterns indicating institutional trades"""
        try:
            data = self.get_volume_data(symbol, 100)
            
            if data.empty or len(data) < 20:
                return {'high_impact_detected': False, 'impact_score': 0.0}
            
            # Calculate price impact metrics
            volume_weighted_price_change = []
            
            for i in range(10, len(data)):
                window = data.iloc[i-10:i+1]
                if len(window) > 1:
                    volume_weights = window['volume'] / window['volume'].sum()
                    weighted_change = (window['price_change'] * volume_weights).sum()
                    volume_weighted_price_change.append(abs(weighted_change))
            
            if not volume_weighted_price_change:
                return {'high_impact_detected': False, 'impact_score': 0.0}
            
            # Current vs historical impact
            current_impact = volume_weighted_price_change[-1] if volume_weighted_price_change else 0
            avg_impact = np.mean(volume_weighted_price_change[:-5]) if len(volume_weighted_price_change) > 5 else 0
            
            # Impact score
            impact_ratio = current_impact / avg_impact if avg_impact > 0 else 1
            impact_score = min(1.0, impact_ratio / 3.0)
            
            high_impact_detected = (current_impact > self.price_impact_threshold and impact_ratio > 2.0)
            
            return {
                'high_impact_detected': high_impact_detected,
                'impact_score': impact_score,
                'current_impact': current_impact,
                'average_impact': avg_impact,
                'impact_ratio': impact_ratio
            }
            
        except Exception as e:
            logger.error(f"Error analyzing price impact for {symbol}: {e}")
            return {'high_impact_detected': False, 'impact_score': 0.0}
    
    def detect_accumulation_distribution(self, symbol: str) -> Dict:
        """Detect accumulation or distribution patterns"""
        try:
            data = self.get_volume_data(symbol, 50)
            
            if data.empty or len(data) < 20:
                return {'pattern': 'NEUTRAL', 'strength': 0.0}
            
            # Calculate On-Balance Volume (OBV)
            obv = []
            obv_current = 0
            
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv_current += data['volume'].iloc[i]
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv_current -= data['volume'].iloc[i]
                obv.append(obv_current)
            
            if len(obv) < 10:
                return {'pattern': 'NEUTRAL', 'strength': 0.0}
            
            # Trend analysis
            recent_obv = np.mean(obv[-5:])
            historical_obv = np.mean(obv[:-5])
            
            obv_trend = (recent_obv - historical_obv) / abs(historical_obv) if historical_obv != 0 else 0
            
            # Pattern classification
            if obv_trend > 0.1:
                pattern = 'ACCUMULATION'
                strength = min(1.0, obv_trend)
            elif obv_trend < -0.1:
                pattern = 'DISTRIBUTION'
                strength = min(1.0, abs(obv_trend))
            else:
                pattern = 'NEUTRAL'
                strength = 0.3
            
            return {
                'pattern': pattern,
                'strength': strength,
                'obv_trend': obv_trend,
                'recent_obv': recent_obv,
                'historical_obv': historical_obv
            }
            
        except Exception as e:
            logger.error(f"Error detecting accumulation/distribution for {symbol}: {e}")
            return {'pattern': 'NEUTRAL', 'strength': 0.0}
    
    def analyze_dark_pool_activity(self, symbol: str) -> Dict:
        """Estimate dark pool activity from market microstructure"""
        try:
            data = self.get_volume_data(symbol, 100)
            
            if data.empty or len(data) < 20:
                return {'dark_pool_activity': 'LOW', 'confidence': 0.0}
            
            # Volume-price analysis
            price_efficiency = []
            
            for i in range(10, len(data)):
                window = data.iloc[i-10:i+1]
                if len(window) > 1:
                    # Calculate price-volume correlation
                    corr = window['volume'].corr(abs(window['price_change']))
                    price_efficiency.append(corr if not pd.isna(corr) else 0)
            
            if not price_efficiency:
                return {'dark_pool_activity': 'LOW', 'confidence': 0.0}
            
            # Low correlation suggests off-exchange trading
            avg_correlation = np.mean(price_efficiency)
            
            # Spread analysis (simplified)
            data['spread_proxy'] = (data['high'] - data['low']) / data['close']
            avg_spread = data['spread_proxy'].tail(10).mean()
            historical_spread = data['spread_proxy'].head(-10).mean()
            
            spread_ratio = avg_spread / historical_spread if historical_spread > 0 else 1
            
            # Dark pool probability
            if avg_correlation < 0.3 and spread_ratio < 0.8:
                activity_level = 'HIGH'
                confidence = 0.8
            elif avg_correlation < 0.5 or spread_ratio < 0.9:
                activity_level = 'MEDIUM'
                confidence = 0.6
            else:
                activity_level = 'LOW'
                confidence = 0.4
            
            return {
                'dark_pool_activity': activity_level,
                'confidence': confidence,
                'price_volume_correlation': avg_correlation,
                'spread_compression': 1 - spread_ratio
            }
            
        except Exception as e:
            logger.error(f"Error analyzing dark pool activity for {symbol}: {e}")
            return {'dark_pool_activity': 'LOW', 'confidence': 0.0}
    
    async def analyze_institutional_flow(self, symbol: str) -> Dict:
        """Comprehensive institutional flow analysis"""
        try:
            # Run all detection methods
            volume_analysis = self.detect_volume_anomalies(symbol)
            impact_analysis = self.analyze_price_impact(symbol)
            accumulation_analysis = self.detect_accumulation_distribution(symbol)
            dark_pool_analysis = self.analyze_dark_pool_activity(symbol)
            
            # Composite institutional activity score
            scores = [
                volume_analysis.get('volume_score', 0),
                impact_analysis.get('impact_score', 0),
                accumulation_analysis.get('strength', 0),
                dark_pool_analysis.get('confidence', 0)
            ]
            
            institutional_score = np.mean([s for s in scores if s > 0])
            
            # After computing institutional_score
            signals = {
                'volume': volume_analysis.get('volume_score',0),
                'impact': impact_analysis.get('impact_score',0),
                'accumulation': accumulation_analysis.get('strength',0),
                'dark_pool': dark_pool_analysis.get('confidence',0)
            }
            confirmation = self.cross_market_confirmation(signals)
            
            # Generate alert if high institutional activity
            alert_generated = institutional_score > 0.7
            
            if alert_generated:
                alert_message = f"""🏛️ Institutional Flow Alert: {symbol}
├── Volume Anomaly: {volume_analysis.get('volume_zscore', 0):.1f}σ spike
├── Price Impact: {impact_analysis.get('impact_ratio', 1):.1f}x normal
├── Pattern: {accumulation_analysis.get('pattern', 'N/A')} ({accumulation_analysis.get('strength', 0):.0%})
└── Dark Pool Activity: {dark_pool_analysis.get('dark_pool_activity', 'N/A')} confidence"""
            else:
                alert_message = ""
            
            result = {
                'symbol': symbol,
                'institutional_score': institutional_score,
                'volume_analysis': volume_analysis,
                'price_impact': impact_analysis,
                'accumulation_pattern': accumulation_analysis,
                'dark_pool_analysis': dark_pool_analysis,
                'alert_generated': alert_generated,
                'alert_message': alert_message,
                'timestamp': datetime.now().isoformat()
            }
            result.update({
                'cross_market_confirmation': confirmation
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in institutional flow analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'institutional_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_institutional_detection(self, symbols: List[str]) -> Dict:
        """Run institutional detection across multiple symbols"""
        try:
            institutional_results = {}
            high_activity_symbols = []
            
            for symbol in symbols:
                result = await self.analyze_institutional_flow(symbol)
                institutional_results[symbol] = result
                
                if result.get('institutional_score', 0) > 0.6:
                    high_activity_symbols.append(symbol)
            
            return {
                'institutional_flow_analysis': institutional_results,
                'high_activity_symbols': high_activity_symbols,
                'total_symbols_analyzed': len(symbols),
                'detection_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in institutional detection: {e}")
            return {
                'error': str(e),
                'institutional_flow_analysis': {},
                'high_activity_symbols': [],
                'detection_timestamp': datetime.now().isoformat()
            }
    
    def detect_order_book_imbalance(self, bids_volume: float, asks_volume: float) -> Dict:
        """Detect order book imbalance indicating buy/sell side pressure"""
        bid_ask_ratio = bids_volume / max(asks_volume, 1)
        signal = "neutral"
        if bid_ask_ratio > 2.5:
            signal = "strong_buy_side_pressure"
        elif bid_ask_ratio < 0.4:
            signal = "strong_sell_side_pressure"
        return {"bid_ask_ratio": bid_ask_ratio, "signal": signal}

    def detect_time_sliced_accumulation(self, volume_slices: List[float], threshold: float = 0.08) -> Dict:
        """Detect time sliced accumulation pattern from volume slices."""
        if not volume_slices or len(volume_slices) < 6:
            return {"accumulation_detected": False, "average_buy_pressure": 0}
        avg_buy_pressure = sum(volume_slices) / len(volume_slices)
        accumulation_detected = avg_buy_pressure > threshold
        return {"accumulation_detected": accumulation_detected, "average_buy_pressure": avg_buy_pressure}

    def cross_market_confirmation(self, signals: Dict[str, float]) -> Dict:
        """Aggregate cross market signals and give confirmation level."""
        if not signals:
            return {"confirmation": "NONE", "confidence": 0.0}
        total_strength = sum(signals.values())
        avg_strength = total_strength / len(signals)
        if avg_strength > 0.7:
            confirmation = "STRONG"
            confidence = 0.85
        elif avg_strength > 0.4:
            confirmation = "MEDIUM"
            confidence = 0.6
        else:
            confirmation = "WEAK"
            confidence = 0.3
        return {"confirmation": confirmation, "confidence": confidence, "avg_strength": avg_strength}

    def approximate_block_trade_detection(self, end_of_day_volume: float, usual_volume: float) -> Dict:
        """Detect possible block trade by spike in end-of-day volume."""
        if usual_volume <= 0:
            return {"block_trade": False, "confidence": 0}
        spike_ratio = end_of_day_volume / usual_volume
        block_trade = spike_ratio > 2.5
        confidence = min(1.0, spike_ratio / 5)
        return {"block_trade": block_trade, "confidence": confidence, "spike_ratio": spike_ratio}