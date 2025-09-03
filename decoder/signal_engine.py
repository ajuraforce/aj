"""
Automated Trading Signal Engine
Converts patterns into actionable trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sqlite3
import asyncio
import openai
import json
import os
import re
import random
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class TradingSignal:
    symbol: str
    action: SignalAction
    confidence: float
    risk_level: RiskLevel
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    reasoning: Optional[List[str]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.reasoning is None:
            self.reasoning = []

class SignalEngine:
    def __init__(self, config: dict, db_path: str):
        self.config = config
        self.db_path = db_path
        signal_config = config.get('signal_engine', {})
        self.min_confidence = signal_config.get('min_confidence', 0.6)
        self.rules = signal_config.get('rules', {})
        
        # Pattern weights
        self.pattern_weights = {
            'price_spike': 0.25,
            'volume_confirmation': 0.20,
            'sentiment_positive': 0.20,
            'multi_timeframe_confirmation': 0.15,
            'sector_momentum': 0.10,
            'correlation_break': 0.10
        }
        
        # Initialize GPT-5 client for signal validation
        try:
            self.gpt5_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.gpt5_enabled = True
            logger.info("GPT-5 signal validation engine initialized successfully")
        except Exception as e:
            logger.warning(f"GPT-5 signal validation initialization failed: {e}")
            self.gpt5_client = None
            self.gpt5_enabled = False
        
        # Set orchestrator reference for adaptive learning
        self.orchestrator = None
        
    def get_market_data(self, symbol: str, lookback_periods: int = 50) -> pd.DataFrame:
        """Fetch market data for signal analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT timestamp, close, volume, open, high, low
            FROM market_data 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=[symbol, lookback_periods])
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_pattern_data(self, symbol: str, lookback_hours: int = 24) -> pd.DataFrame:
        """Fetch recent pattern data for signal analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT timestamp, type, confidence, signals, asset
            FROM patterns 
            WHERE asset = ? 
            AND timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
            """.format(lookback_hours)
            
            df = pd.read_sql_query(query, conn, params=[symbol])
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching pattern data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_pattern_conditions(self, symbol: str) -> Dict[str, bool]:
        """Calculate whether various pattern conditions are met"""
        try:
            market_data = self.get_market_data(symbol)
            pattern_data = self.get_pattern_data(symbol)
            
            conditions = {
                'price_spike': False,
                'volume_confirmation': False,
                'sentiment_positive': False,
                'multi_timeframe_confirmation': False,
                'sector_momentum': False,
                'correlation_break': False
            }
            
            if market_data.empty:
                return conditions
            
            # Price spike detection
            if len(market_data) >= 2:
                recent_return = (market_data['close'].iloc[-1] - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2]
                if abs(recent_return) > 0.02:  # 2% price movement
                    conditions['price_spike'] = True
            
            # Volume confirmation
            if len(market_data) >= 10:
                avg_volume = market_data['volume'].rolling(10).mean().iloc[-1]
                current_volume = market_data['volume'].iloc[-1]
                if current_volume > avg_volume * 1.5:  # 50% above average volume
                    conditions['volume_confirmation'] = True
            
            # Pattern-based conditions
            for _, pattern in pattern_data.iterrows():
                pattern_type = pattern['type']
                confidence = pattern.get('confidence', 0)
                
                if pattern_type == 'sentiment_signal' and confidence is not None and confidence > 0.6:
                    conditions['sentiment_positive'] = True
                elif pattern_type == 'cross_source_correlation' and confidence is not None and confidence > 0.7:
                    conditions['multi_timeframe_confirmation'] = True
                elif pattern_type == 'correlation_break':
                    conditions['correlation_break'] = True
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error calculating pattern conditions for {symbol}: {e}")
            return {condition: False for condition in self.pattern_weights.keys()}
    
    def evaluate_rule(self, rule: Dict, conditions: Dict[str, bool]) -> Tuple[bool, float]:
        """Evaluate a trading rule against current conditions"""
        try:
            required_conditions = rule.get('conditions', [])
            
            if not required_conditions:
                return False, 0.0
            
            met_conditions = 0
            total_weight = 0
            
            for condition in required_conditions:
                if condition in self.pattern_weights:
                    total_weight += self.pattern_weights[condition]
                    if conditions.get(condition, False):
                        met_conditions += 1
            
            if not required_conditions:
                return False, 0.0
            
            # Calculate signal strength based on met conditions and their weights
            condition_ratio = met_conditions / len(required_conditions)
            weight_ratio = sum(self.pattern_weights[c] for c in required_conditions if conditions.get(c, False)) / total_weight if total_weight > 0 else 0
            
            signal_strength = (condition_ratio + weight_ratio) / 2
            
            # Rule is triggered if more than 60% of conditions are met
            rule_triggered = condition_ratio >= 0.6
            
            return rule_triggered, signal_strength
            
        except Exception as e:
            logger.error(f"Error evaluating rule: {e}")
            return False, 0.0
    
    def calculate_position_sizing(self, symbol: str, signal_strength: float, risk_level: RiskLevel) -> Dict:
        """Calculate position sizing and risk parameters"""
        try:
            market_data = self.get_market_data(symbol, 20)
            
            if market_data.empty:
                return {
                    'position_size': 0.0,
                    'stop_loss': None,
                    'take_profit': None
                }
            
            current_price = market_data['close'].iloc[-1]
            
            # Calculate volatility for risk adjustment
            returns = market_data['close'].pct_change().dropna()
            if len(returns) > 1:
                volatility = returns.std()
            else:
                volatility = 0.02  # Default 2% volatility
            
            # Risk-based position sizing
            trading_config = self.config.get('trading', {})
            base_position_size = trading_config.get('max_position_size_usd', 1000)
            risk_per_trade = trading_config.get('risk_per_trade', 0.02)
            
            # Adjust position size based on signal strength and risk level
            risk_multipliers = {
                RiskLevel.LOW: 0.5,
                RiskLevel.MEDIUM: 1.0,
                RiskLevel.HIGH: 1.5
            }
            
            risk_multiplier = risk_multipliers.get(risk_level, 1.0)
            position_size = base_position_size * signal_strength * risk_multiplier
            
            # Calculate stop loss and take profit
            stop_loss_pct = max(volatility * 2, 0.02)  # At least 2% or 2x volatility
            take_profit_pct = stop_loss_pct * 2  # 2:1 reward to risk ratio
            
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
            
            return {
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': position_size * stop_loss_pct
            }
            
        except Exception as e:
            logger.error(f"Error calculating position sizing for {symbol}: {e}")
            return {
                'position_size': 0.0,
                'stop_loss': None,
                'take_profit': None
            }
    
    async def validate_signal_with_gpt5(self, signal: TradingSignal, conditions: Dict, market_data: pd.DataFrame) -> Dict:
        """Enhanced signal validation using GPT-5 reasoning"""
        if not self.gpt5_enabled or signal is None:
            return {'validation': 'GPT-5 signal validation not available.', 'confidence_adjustment': 0.0}
            
        try:
            # Format market context
            latest_price = market_data['close'].iloc[-1] if not market_data.empty else 0
            price_change_24h = ((market_data['close'].iloc[-1] / market_data['close'].iloc[-2]) - 1) * 100 if len(market_data) > 1 else 0
            volume_ratio = market_data['volume'].iloc[-1] / market_data['volume'].mean() if not market_data.empty else 1
            
            # Format conditions for analysis
            active_conditions = [k for k, v in conditions.items() if v]
            conditions_text = ", ".join(active_conditions) if active_conditions else "No major conditions detected"
            
            prompt = f"""
            As an expert trading analyst, validate this algorithmic trading signal:
            
            Signal Details:
            - Symbol: {signal.symbol}
            - Action: {signal.action.value}
            - Confidence: {signal.confidence:.2%}
            - Risk Level: {signal.risk_level.value}
            - Entry Price: ${latest_price:.2f}
            - Position Size: {signal.position_size or 'Not calculated'}%
            
            Market Context:
            - 24h Price Change: {price_change_24h:+.2f}%
            - Volume Ratio: {volume_ratio:.2f}x average
            - Active Conditions: {conditions_text}
            
            Technical Reasoning from Algorithm:
            {'; '.join(signal.reasoning) if signal.reasoning else 'No specific reasoning provided'}
            
            Provide comprehensive signal validation including:
            1. Signal quality assessment (scale 1-10)
            2. Market timing analysis
            3. Risk-reward evaluation
            4. Alternative perspectives
            5. Confidence adjustment recommendation (-0.3 to +0.3)
            6. Key risks and opportunities
            
            Format response as JSON with keys: signal_quality, timing_analysis, risk_reward, alternative_view, confidence_adjustment, validation_summary
            """
            
            response = self.gpt5_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=800
            )
            
            validation = json.loads(response.choices[0].message.content)
            
            return {
                'gpt5_validation': validation,
                'confidence_adjustment': float(validation.get('confidence_adjustment', 0.0)),
                'validation_summary': validation.get('validation_summary', 'Signal validation completed'),
                'signal_quality': validation.get('signal_quality', 5)
            }
            
        except Exception as e:
            logger.error(f"Error validating signal with GPT-5: {e}")
            return {'validation': 'Signal validation failed.', 'confidence_adjustment': 0.0}
    
    async def generate_signal_reasoning(self, symbol: str, signal_data: Dict) -> str:
        """Generate enhanced reasoning using GPT-5"""
        if not self.gpt5_enabled:
            return "Enhanced reasoning not available - GPT-5 not configured."
            
        try:
            prompt = f"""
            Explain this trading signal in clear, professional language:
            
            Symbol: {symbol}
            Signal Strength: {signal_data.get('confidence', 0):.1%}
            Market Conditions: {signal_data.get('conditions', {})}
            Technical Patterns: {signal_data.get('patterns', [])}
            
            Write a 2-3 sentence explanation suitable for traders explaining:
            1. Why this signal was generated
            2. Key supporting factors
            3. Main risks to consider
            
            Style: Professional, concise, actionable
            """
            
            response = self.gpt5_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating signal reasoning: {e}")
            return "Signal reasoning generation failed."
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal for a symbol"""
        try:
            # Get current conditions
            conditions = self.calculate_pattern_conditions(symbol)
            
            # Evaluate all rules
            best_signal = None
            best_strength = 0
            
            for rule_name, rule in self.rules.items():
                rule_triggered, signal_strength = self.evaluate_rule(rule, conditions)
                
                if rule_triggered and signal_strength >= self.min_confidence:
                    if signal_strength > best_strength:
                        best_strength = signal_strength
                        
                        # Determine action based on rule
                        action_str = rule.get('action', 'HOLD')
                        try:
                            action = SignalAction(action_str)
                        except ValueError:
                            action = SignalAction.HOLD
                        
                        # Determine risk level
                        risk_str = rule.get('risk_level', 'medium')
                        try:
                            risk_level = RiskLevel(risk_str)
                        except ValueError:
                            risk_level = RiskLevel.MEDIUM
                        
                        # Calculate position sizing
                        position_info = self.calculate_position_sizing(symbol, signal_strength, risk_level)
                        
                        # Get current price
                        market_data = self.get_market_data(symbol, 1)
                        entry_price = market_data['close'].iloc[-1] if not market_data.empty else None
                        
                        # Build reasoning
                        reasoning = []
                        for condition, met in conditions.items():
                            if met:
                                reasoning.append(f"{condition.replace('_', ' ').title()}: ✓")
                        
                        best_signal = TradingSignal(
                            symbol=symbol,
                            action=action,
                            confidence=signal_strength,
                            risk_level=risk_level,
                            entry_price=entry_price,
                            stop_loss=position_info.get('stop_loss'),
                            take_profit=position_info.get('take_profit'),
                            position_size=position_info.get('position_size'),
                            reasoning=reasoning
                        )
            
            return best_signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def adaptive_signal_adjustment(self, symbol: str, signal: Dict) -> Dict:
        """
        Adjust confidence dynamically based on historical accuracy for `symbol`.
        """
        try:
            # Fetch recent accuracy from MultiTimeframeAnalyzer if orchestrator is available
            if self.orchestrator and hasattr(self.orchestrator, 'multi_timeframe'):
                accuracy_data = self.orchestrator.multi_timeframe.track_prediction_accuracy(
                    symbol, signal['signal'], signal.get('confidence', 0.0)
                )
                hist_acc = accuracy_data.get('accuracy', 0.5)
            else:
                # Fallback to default historical accuracy
                hist_acc = 0.5
            
            # Combine original confidence with historical accuracy
            adjusted_conf = signal['confidence'] * (0.7 + 0.3 * hist_acc)
            signal['confidence'] = min(1.0, adjusted_conf)
            return signal
        except Exception as e:
            logger.error(f"Error in adaptive signal adjustment for {symbol}: {e}")
            return signal

    async def generate_enhanced_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal with GPT-5 validation and enhancement"""
        try:
            # Generate base signal
            base_signal = self.generate_signal(symbol)
            if not base_signal:
                return None
            
            # Get market data for GPT-5 validation
            market_data = self.get_market_data(symbol, 10)
            conditions = self.calculate_pattern_conditions(symbol)
            
            # Validate signal with GPT-5
            if self.gpt5_enabled:
                validation_result = await self.validate_signal_with_gpt5(base_signal, conditions, market_data)
                
                # Apply confidence adjustment
                confidence_adjustment = validation_result.get('confidence_adjustment', 0.0)
                adjusted_confidence = max(0.0, min(1.0, base_signal.confidence + confidence_adjustment))
                
                # Update signal with GPT-5 insights
                enhanced_signal = TradingSignal(
                    symbol=base_signal.symbol,
                    action=base_signal.action,
                    confidence=adjusted_confidence,
                    risk_level=base_signal.risk_level,
                    entry_price=base_signal.entry_price,
                    stop_loss=base_signal.stop_loss,
                    take_profit=base_signal.take_profit,
                    position_size=base_signal.position_size,
                    reasoning=base_signal.reasoning + [f"GPT-5 Quality: {validation_result.get('signal_quality', 5)}/10"]
                )
                
                # Store validation metadata as dict attribute
                if not hasattr(enhanced_signal, '__dict__'):
                    enhanced_signal.__dict__ = {}
                enhanced_signal.__dict__['gpt5_validation'] = validation_result
                
                # Only return signal if it meets enhanced confidence threshold after GPT-5 validation
                if adjusted_confidence >= self.min_confidence:
                    return enhanced_signal
                else:
                    logger.info(f"Signal for {symbol} filtered out by GPT-5 validation (confidence: {adjusted_confidence:.2%})")
                    return None
            else:
                return base_signal
                
        except Exception as e:
            logger.error(f"Error generating enhanced signal for {symbol}: {e}")
            return None
    
    def generate_signals_for(self, symbol: str) -> List[Dict]:
        """Generate raw signals for adaptive adjustment"""
        try:
            # Get basic signal data
            base_signal = self.generate_signal(symbol)
            if not base_signal:
                return []
            
            # Convert signal to dict format for adaptive adjustment
            signal_dict = {
                'symbol': base_signal.symbol,
                'signal': base_signal.action.value,
                'confidence': base_signal.confidence,
                'risk_level': base_signal.risk_level.value,
                'entry_price': base_signal.entry_price,
                'reasoning': base_signal.reasoning
            }
            
            return [signal_dict]
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []

    async def run_signal_generation(self, symbols: List[str]) -> Dict:
        """Run signal generation for all symbols"""
        try:
            signals = []
            
            for symbol in symbols:
                # Generate raw signals for adaptive adjustment
                raw_signals = self.generate_signals_for(symbol)
                
                # Apply adaptive adjustment if enabled
                if self.orchestrator and hasattr(self.orchestrator, 'adaptive_learning_enabled') and self.orchestrator.adaptive_learning_enabled:
                    for sig in raw_signals:
                        sig = self.adaptive_signal_adjustment(symbol, sig)
                
                # Convert to enhanced signals
                for raw_signal in raw_signals:
                    signal = await self.generate_enhanced_signal(symbol)
                    if signal:
                        signals.append({
                            'symbol': signal.symbol,
                            'action': signal.action.value,
                            'confidence': signal.confidence,
                            'risk_level': signal.risk_level.value,
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'position_size': signal.position_size,
                            'reasoning': signal.reasoning,
                            'timestamp': signal.timestamp.isoformat()
                        })
            
            return {
                'signals': signals,
                'total_signals': len(signals),
                'symbols_analyzed': len(symbols),
                'high_confidence_signals': len([s for s in signals if s['confidence'] > 0.8]),
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in signal generation: {e}")
            return {
                'error': str(e),
                'signals': [],
                'generation_timestamp': datetime.now().isoformat()
            }