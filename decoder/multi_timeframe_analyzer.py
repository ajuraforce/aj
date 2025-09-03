"""
Multi-Timeframe Analysis Engine
Analyzes patterns across multiple timeframes for enhanced signal reliability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple
import asyncio
import random
import logging

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    def __init__(self, config: dict, db_path: str):
        self.config = config
        self.db_path = db_path
        self.windows = config.get('multi_timeframe', {}).get('windows', ['1h', '4h', '1d'])
        self.confirmation_threshold = config.get('multi_timeframe', {}).get('confirmation_threshold', 0.7)
        self.weight_multiplier = config.get('multi_timeframe', {}).get('weight_multiplier', 1.5)
        
        # Add tracking for prediction accuracy
        self.prediction_history = {}
        
    def calculate_z_score(self, data: pd.Series, window: int = 20) -> float:
        """Calculate z-score for current value against rolling window"""
        try:
            if len(data) < window or data.isna().all():
                return 0.0
                
            rolling_series = data.rolling(window=window).mean()
            rolling_mean = float(rolling_series.iloc[-1])
            rolling_std_series = data.rolling(window=window).std()
            rolling_std = float(rolling_std_series.iloc[-1])
            current_value = float(data.iloc[-1])
            
            if pd.isna(rolling_std) or rolling_std == 0:
                return 0.0
            return (current_value - rolling_mean) / rolling_std
        except Exception as e:
            logger.warning(f"Error calculating z-score: {e}")
            return 0.0
    
    def get_timeframe_data(self, symbol: str, timeframe: str, periods: int = 100) -> pd.DataFrame:
        """Fetch data for specific timeframe"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Convert timeframe to minutes for SQL query
            timeframe_minutes = {
                '1h': 60, '4h': 240, '1d': 1440,
                '15m': 15, '30m': 30, '2h': 120
            }
            
            minutes = timeframe_minutes.get(timeframe, 60)
            
            query = f"""
            SELECT 
                timestamp,
                close as price,
                volume,
                COALESCE(sentiment_score, 0) as sentiment_score,
                COALESCE(mention_count, 0) as mention_count
            FROM (
                SELECT timestamp, close, volume, NULL as sentiment_score, NULL as mention_count
                FROM market_data 
                WHERE symbol = ? 
                AND timestamp >= strftime('%s', datetime('now', '-{periods * minutes} minutes'))
                UNION ALL
                SELECT timestamp, price as close, 0 as volume, sentiment_score, mention_count
                FROM patterns 
                WHERE asset = ? 
                AND timestamp >= datetime('now', '-{periods * minutes} minutes')
            )
            ORDER BY timestamp DESC
            LIMIT {periods}
            """
            
            df = pd.read_sql_query(query, conn, params=[symbol, symbol])
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Fill missing values with forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching timeframe data for {symbol} ({timeframe}): {e}")
            return pd.DataFrame()
    
    def analyze_timeframe_signals(self, symbol: str, timeframe: str) -> Dict:
        """Analyze signals for a specific timeframe"""
        try:
            data = self.get_timeframe_data(symbol, timeframe)
            
            if data.empty or len(data) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'details': {}}
            
            # Price momentum analysis
            if len(data) >= 2:
                price_change = (data['price'].iloc[-1] - data['price'].iloc[-2]) / data['price'].iloc[-2]
            else:
                price_change = 0.0
                
            price_returns = data['price'].pct_change().dropna()
            price_z_score = self.calculate_z_score(pd.Series(price_returns))
            
            # Volume analysis
            volume_z_score = self.calculate_z_score(pd.Series(data['volume']))
            volume_spike = volume_z_score > 2.0
            
            # Sentiment analysis
            sentiment_momentum = float(data['sentiment_score'].rolling(window=min(5, len(data))).mean().iloc[-1])
            sentiment_z_score = self.calculate_z_score(pd.Series(data['sentiment_score']))
            
            # Technical indicators
            sma_window = min(20, len(data) - 1)
            if sma_window > 0:
                sma_20 = float(data['price'].rolling(window=sma_window).mean().iloc[-1])
                current_price = data['price'].iloc[-1]
                trend_direction = 'BULLISH' if current_price > sma_20 else 'BEARISH'
            else:
                trend_direction = 'NEUTRAL'
            
            # Calculate RSI
            delta = data['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=min(14, len(data))).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, len(data))).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
            
            # Signal determination
            signal_score = 0
            
            # Price momentum weight
            if price_z_score > 1.5: 
                signal_score += 2
            elif price_z_score > 0.5: 
                signal_score += 1
            elif price_z_score < -1.5: 
                signal_score -= 2
            elif price_z_score < -0.5: 
                signal_score -= 1
            
            # Volume confirmation
            if volume_spike and price_z_score > 0: 
                signal_score += 1
            elif volume_spike and price_z_score < 0: 
                signal_score -= 1
            
            # Sentiment weight
            if sentiment_z_score > 1.0: 
                signal_score += 1
            elif sentiment_z_score < -1.0: 
                signal_score -= 1
            
            # RSI consideration
            if current_rsi < 30 and price_z_score > 0: 
                signal_score += 1  # Oversold bounce
            elif current_rsi > 70 and price_z_score < 0: 
                signal_score -= 1  # Overbought decline
            
            # Determine final signal
            if signal_score >= 3:
                signal = 'STRONG_BULLISH'
                confidence = min(0.9, abs(signal_score) / 5.0)
            elif signal_score >= 1:
                signal = 'BULLISH'
                confidence = min(0.7, abs(signal_score) / 3.0)
            elif signal_score <= -3:
                signal = 'STRONG_BEARISH'
                confidence = min(0.9, abs(signal_score) / 5.0)
            elif signal_score <= -1:
                signal = 'BEARISH'
                confidence = min(0.7, abs(signal_score) / 3.0)
            else:
                signal = 'NEUTRAL'
                confidence = 0.3
            
            return {
                'signal': signal,
                'confidence': confidence,
                'timeframe': timeframe,
                'details': {
                    'price_z_score': price_z_score,
                    'volume_z_score': volume_z_score,
                    'sentiment_z_score': sentiment_z_score,
                    'rsi': current_rsi,
                    'trend_direction': trend_direction,
                    'signal_score': signal_score,
                    'volume_spike': volume_spike
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe signals for {symbol} ({timeframe}): {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0, 'details': {}}
    
    def track_prediction_accuracy(self, symbol: str, signal: str, confidence: float) -> Dict:
        """Track prediction accuracy for adaptive learning"""
        try:
            if symbol not in self.prediction_history:
                self.prediction_history[symbol] = {'predictions': [], 'accuracy': 0.5}
            
            # Add prediction to history (simplified tracking)
            self.prediction_history[symbol]['predictions'].append({
                'signal': signal,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            # Keep only recent predictions (last 100)
            if len(self.prediction_history[symbol]['predictions']) > 100:
                self.prediction_history[symbol]['predictions'] = self.prediction_history[symbol]['predictions'][-100:]
            
            # Calculate simple accuracy estimate (simplified for now)
            accuracy = 0.5 + (confidence - 0.5) * 0.2  # Basic confidence-based accuracy estimate
            self.prediction_history[symbol]['accuracy'] = accuracy
            
            return {'accuracy': accuracy, 'prediction_count': len(self.prediction_history[symbol]['predictions'])}
        except Exception as e:
            logger.error(f"Error tracking prediction accuracy for {symbol}: {e}")
            return {'accuracy': 0.5, 'prediction_count': 0}

    async def analyze_multi_timeframe(self, symbol: str) -> Dict:
        """Analyze signals across multiple timeframes"""
        try:
            # Adjust windows based on regime if enabled
            if self.config.get('multi_timeframe', {}).get('regime_detection', False):
                # Example: use only '1h','4h' in volatile regimes
                regime = self.config.get('regime', 'neutral')
                if regime == 'crisis':
                    self.windows = ['1h','2h']
                elif regime == 'bull':
                    self.windows = ['4h','1d']
            
            timeframe_results = {}
            
            # Analyze each timeframe
            for timeframe in self.windows:
                result = self.analyze_timeframe_signals(symbol, timeframe)
                timeframe_results[timeframe] = result
            
            # Cross-timeframe confirmation
            confirmations = 0
            total_confidence = 0
            bullish_signals = 0
            bearish_signals = 0
            
            for tf, result in timeframe_results.items():
                if result['signal'] in ['BULLISH', 'STRONG_BULLISH']:
                    bullish_signals += 1
                elif result['signal'] in ['BEARISH', 'STRONG_BEARISH']:
                    bearish_signals += 1
                
                if result['confidence'] > 0.5:
                    confirmations += 1
                    total_confidence += result['confidence']
            
            # Calculate overall confidence
            if confirmations > 0:
                avg_confidence = total_confidence / confirmations
                confirmation_ratio = confirmations / len(self.windows)
                
                if confirmation_ratio >= self.confirmation_threshold:
                    # Monte Carlo-style confidence boost
                    mc_samples = [random.gauss(avg_confidence, 0.1) for _ in range(100)]
                    boosted = np.mean([min(1.0, s * self.weight_multiplier) for s in mc_samples])
                    final_confidence = float(boosted)
                else:
                    final_confidence = avg_confidence * 0.7
            else:
                final_confidence = 0.3
                confirmation_ratio = 0
            
            # Determine overall signal
            if bullish_signals >= 2 and bullish_signals > bearish_signals:
                overall_signal = 'CONFIRMED_BULLISH'
            elif bearish_signals >= 2 and bearish_signals > bullish_signals:
                overall_signal = 'CONFIRMED_BEARISH'
            elif bullish_signals == bearish_signals and bullish_signals > 0:
                overall_signal = 'MIXED'
            else:
                overall_signal = 'NEUTRAL'
            
            return {
                'symbol': symbol,
                'overall_signal': overall_signal,
                'overall_confidence': final_confidence,
                'confirmation_ratio': confirmation_ratio,
                'timeframe_results': timeframe_results,
                'summary': {
                    'bullish_timeframes': bullish_signals,
                    'bearish_timeframes': bearish_signals,
                    'neutral_timeframes': len(self.windows) - bullish_signals - bearish_signals
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'overall_signal': 'NEUTRAL',
                'overall_confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def track_prediction_accuracy(self, symbol: str, predicted_signal: str, confidence: float) -> Dict:
        """Track accuracy of previous predictions for learning."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Store prediction
            prediction_id = f"{symbol}_{int(time.time())}"
            conn.execute("""
                INSERT INTO pattern_outcomes (pattern_id, symbol, predicted_outcome, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (prediction_id, symbol, predicted_signal, confidence, int(time.time())))
            
            # Get recent actual outcomes for accuracy tracking
            query = """
                SELECT predicted_outcome, actual_outcome, confidence
                FROM pattern_outcomes
                WHERE symbol = ? AND actual_outcome IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 50
            """
            results = conn.execute(query, (symbol,)).fetchall()
            
            if results:
                correct_predictions = sum(1 for pred, actual, _ in results if pred == actual)
                total_predictions = len(results)
                accuracy = correct_predictions / total_predictions
                weighted_accuracy = sum(conf for pred, actual, conf in results if pred == actual) / total_predictions
            else:
                accuracy = 0.5
                weighted_accuracy = 0.5
            
            conn.close()
            return {"accuracy": accuracy, "weighted_accuracy": weighted_accuracy, "sample_size": len(results)}
            
        except Exception as e:
            logger.error(f"Error tracking prediction accuracy: {e}")
            return {"accuracy": 0.5, "weighted_accuracy": 0.5, "sample_size": 0}

    def calculate_signal_probabilities(self, signal_strength: float, historical_accuracy: float) -> Dict:
        """Calculate probability of different outcomes based on signal strength and historical accuracy."""
        base_prob = 0.5  # Neutral baseline
        
        # Adjust probability based on signal strength
        signal_adjustment = signal_strength * 0.3  # Max 30% adjustment from signal
        
        # Adjust based on historical accuracy
        accuracy_adjustment = (historical_accuracy - 0.5) * 0.2  # Max 20% adjustment from track record
        
        # Calculate directional probabilities
        if signal_strength > 0:
            bullish_prob = min(0.85, base_prob + signal_adjustment + accuracy_adjustment)
            bearish_prob = max(0.15, 1 - bullish_prob)
        else:
            bearish_prob = min(0.85, base_prob - signal_adjustment + accuracy_adjustment)
            bullish_prob = max(0.15, 1 - bearish_prob)
        
        neutral_prob = max(0.1, 1 - bullish_prob - bearish_prob)
        
        return {
            "bullish_probability": bullish_prob,
            "bearish_probability": bearish_prob,
            "neutral_probability": neutral_prob,
            "signal_strength": signal_strength,
            "historical_accuracy": historical_accuracy
        }

    async def enhanced_multi_timeframe_analysis(self, symbol: str) -> Dict:
        """Enhanced multi-timeframe analysis with probability calculations."""
        try:
            # Get standard multi-timeframe results  
            standard_results = await self.analyze_multi_timeframe(symbol)
            
            # Track accuracy for this symbol
            accuracy_data = self.track_prediction_accuracy(symbol, standard_results['overall_signal'], standard_results['overall_confidence'])
            
            # Calculate outcome probabilities
            signal_strength = standard_results['overall_confidence'] if standard_results['overall_signal'] in ['CONFIRMED_BULLISH'] else -standard_results['overall_confidence'] if standard_results['overall_signal'] in ['CONFIRMED_BEARISH'] else 0
            probabilities = self.calculate_signal_probabilities(signal_strength, accuracy_data['accuracy'])
            
            # Enhanced results with probability data
            enhanced_results = standard_results.copy()
            enhanced_results.update({
                'accuracy_tracking': accuracy_data,
                'outcome_probabilities': probabilities,
                'enhanced_analysis': True
            })
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in enhanced multi-timeframe analysis: {e}")
            return await self.analyze_multi_timeframe(symbol)