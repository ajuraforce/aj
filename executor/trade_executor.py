"""
Trade Executor Module
Implements safe trading execution with risk management and circuit breakers
"""

import os
import ccxt
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import math

logger = logging.getLogger(__name__)

class TradeExecutor:
    """Safe trade execution with comprehensive risk management"""
    
    def __init__(self):
        self.exchange = None
        self.open_trades = []
        self.daily_pnl = 0.0
        self.trade_count_today = 0
        self.last_reset_date = datetime.utcnow().date()
        self.setup_exchange()
    
    def setup_exchange(self):
        """Initialize exchange for trading"""
        try:
            # Use paper trading by default
            sandbox_mode = os.getenv('TRADING_SANDBOX', 'true').lower() == 'true'
            
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', 'default_key'),
                'secret': os.getenv('BINANCE_SECRET', 'default_secret'),
                'sandbox': sandbox_mode,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                    'defaultType': 'spot'  # Use spot trading
                }
            })
            
            if not sandbox_mode:
                logger.warning("LIVE TRADING MODE ENABLED!")
            else:
                logger.info("Paper trading mode enabled")
            
            # Test connection
            self.exchange.load_markets()
            
        except Exception as e:
            logger.error(f"Failed to setup trading exchange: {e}")
            self.exchange = None
    
    async def execute_trade(self, pattern: Dict, score: float):
        """Execute trade based on high-scoring pattern"""
        if not self.exchange:
            logger.warning("Exchange not available for trading")
            return
        
        try:
            # Reset daily limits if new day
            await self.check_daily_reset()
            
            # Pre-trade safety checks
            if not await self.pre_trade_safety_checks(pattern, score):
                return
            
            # Generate trade parameters
            trade_params = await self.generate_trade_parameters(pattern, score)
            if not trade_params:
                return
            
            # Execute the trade
            success = await self.place_trade(trade_params)
            if success:
                await self.record_trade(trade_params, pattern, score)
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def check_daily_reset(self):
        """Reset daily counters if new day"""
        today = datetime.utcnow().date()
        if today > self.last_reset_date:
            logger.info(f"Resetting daily trading limits for {today}")
            self.daily_pnl = 0.0
            self.trade_count_today = 0
            self.last_reset_date = today
    
    async def pre_trade_safety_checks(self, pattern: Dict, score: float) -> bool:
        """Comprehensive pre-trade safety checks"""
        try:
            # Check if live trading is enabled
            live_trading = os.getenv('LIVE_TRADING', 'false').lower() == 'true'
            if not live_trading:
                logger.info("Live trading disabled - trade would be simulated")
                return await self.simulate_trade_check(pattern, score)
            
            # Minimum score threshold
            min_score = float(os.getenv('TRADE_THRESHOLD', '80'))
            if score < min_score:
                logger.info(f"Score {score} below trade threshold {min_score}")
                return False
            
            # Daily trade limit
            max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '5'))
            if self.trade_count_today >= max_daily_trades:
                logger.info(f"Daily trade limit reached: {self.trade_count_today}")
                return False
            
            # Daily PnL limit (stop loss for the day)
            daily_loss_limit = float(os.getenv('DAILY_LOSS_LIMIT', '-500.0'))
            if self.daily_pnl <= daily_loss_limit:
                logger.warning(f"Daily loss limit hit: {self.daily_pnl}")
                return False
            
            # Check account balance
            if not await self.check_sufficient_balance():
                return False
            
            # Market hours check (optional)
            if not await self.check_market_conditions():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in pre-trade safety checks: {e}")
            return False
    
    async def simulate_trade_check(self, pattern: Dict, score: float) -> bool:
        """Simulate trade checks (paper trading)"""
        logger.info(f"[SIMULATION] Would execute trade for {pattern.get('asset', 'UNKNOWN')} with score {score}")
        return True
    
    async def generate_trade_parameters(self, pattern: Dict, score: float) -> Optional[Dict]:
        """Generate trade parameters based on pattern and risk management"""
        try:
            asset = pattern.get('asset', '')
            if not asset:
                return None
            
            # Determine trading symbol
            symbol = f"{asset}/USDT"
            if not self.exchange or not self.exchange.markets or symbol not in self.exchange.markets:
                logger.warning(f"Symbol {symbol} not available")
                return None
            
            # Determine trade direction
            trade_direction = self.determine_trade_direction(pattern)
            if not trade_direction:
                return None
            
            # Calculate position size
            position_size = await self.calculate_position_size(symbol, score)
            if not position_size:
                return None
            
            # Get current market price
            if not self.exchange:
                logger.error("Exchange not available for fetching ticker")
                return None
                
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_ticker, symbol
            )
            current_price = float(ticker['last'])
            
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            stop_loss = self.calculate_stop_loss(entry_price, trade_direction)
            take_profit = self.calculate_take_profit(entry_price, trade_direction, score)
            
            return {
                'symbol': symbol,
                'direction': trade_direction,
                'size': position_size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'pattern_id': pattern.get('id', ''),
                'score': score
            }
            
        except Exception as e:
            logger.error(f"Error generating trade parameters: {e}")
            return None
    
    def determine_trade_direction(self, pattern: Dict) -> Optional[str]:
        """Determine whether to go long or short based on pattern"""
        pattern_type = pattern.get('type', '')
        signals = pattern.get('signals', {})
        
        if pattern_type == 'price_movement':
            change_percent = signals.get('price_change_percent', 0)
            return 'long' if change_percent > 0 else 'short'
        
        elif pattern_type == 'volume_spike':
            # Volume spikes generally suggest continuation
            return 'long'  # Bias toward long on volume
        
        elif pattern_type == 'news_impact':
            impact = signals.get('potential_impact', 'medium')
            keywords = signals.get('keywords', [])
            
            positive_keywords = ['approval', 'adoption', 'partnership', 'upgrade']
            negative_keywords = ['ban', 'regulation', 'hack', 'delay']
            
            positive_count = sum(1 for kw in keywords if kw in positive_keywords)
            negative_count = sum(1 for kw in keywords if kw in negative_keywords)
            
            if positive_count > negative_count:
                return 'long'
            elif negative_count > positive_count:
                return 'short'
        
        elif pattern_type == 'cross_source_correlation':
            # Cross-correlation generally suggests momentum
            return 'long'
        
        return None
    
    async def calculate_position_size(self, symbol: str, score: float) -> Optional[float]:
        """Calculate position size based on risk management"""
        try:
            # Base position size (percentage of available balance)
            base_position_percent = float(os.getenv('BASE_POSITION_PERCENT', '2.0'))  # 2%
            
            # Scale based on confidence (score)
            confidence_multiplier = score / 100.0  # 0.0 - 1.0
            position_percent = base_position_percent * confidence_multiplier
            
            # Get available balance
            if not self.exchange:
                logger.error("Exchange not available for fetching balance")
                return None
                
            balance = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_balance
            )
            
            available_usdt = balance.get('USDT', {}).get('free', 0) or 0
            if float(available_usdt) < 50:  # Minimum $50 balance
                logger.warning(f"Insufficient balance: ${available_usdt}")
                return None
            
            # Calculate position size in USDT
            position_size_usdt = float(available_usdt) * (position_percent / 100.0)
            
            # Apply limits
            max_position = float(os.getenv('MAX_POSITION_SIZE', '200.0'))
            min_position = float(os.getenv('MIN_POSITION_SIZE', '10.0'))
            
            position_size_usdt = max(min_position, min(max_position, position_size_usdt))
            
            return position_size_usdt
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None
    
    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """Calculate stop loss price"""
        stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENT', '2.0'))  # 2%
        
        if direction == 'long':
            return entry_price * (1 - stop_loss_percent / 100.0)
        else:  # short
            return entry_price * (1 + stop_loss_percent / 100.0)
    
    def calculate_take_profit(self, entry_price: float, direction: str, score: float) -> float:
        """Calculate take profit price based on score confidence"""
        # Base take profit
        base_tp_percent = float(os.getenv('BASE_TAKE_PROFIT_PERCENT', '4.0'))  # 4%
        
        # Scale based on confidence
        confidence_multiplier = score / 100.0
        tp_percent = base_tp_percent * (0.5 + confidence_multiplier)  # 50% to 150% of base
        
        if direction == 'long':
            return entry_price * (1 + tp_percent / 100.0)
        else:  # short
            return entry_price * (1 - tp_percent / 100.0)
    
    async def place_trade(self, trade_params: Dict) -> bool:
        """Place the actual trade order"""
        try:
            symbol = trade_params['symbol']
            direction = trade_params['direction']
            size_usdt = trade_params['size']
            entry_price = trade_params['entry_price']
            
            # Calculate quantity in base currency
            quantity = size_usdt / entry_price
            
            # Round to exchange precision
            if not self.exchange or not self.exchange.markets:
                logger.error("Exchange or markets not available")
                return False
                
            market = self.exchange.markets[symbol]
            precision = market['precision']['amount']
            quantity = self.exchange.amount_to_precision(symbol, quantity)
            
            if float(quantity or 0) == 0:
                logger.warning(f"Quantity too small for {symbol}")
                return False
            
            # In simulation mode, just log
            live_trading = os.getenv('LIVE_TRADING', 'false').lower() == 'true'
            if not live_trading:
                logger.info(f"[SIMULATION] {direction.upper()} {quantity} {symbol} at ${entry_price:.4f}")
                logger.info(f"[SIMULATION] Stop Loss: ${trade_params['stop_loss']:.4f}")
                logger.info(f"[SIMULATION] Take Profit: ${trade_params['take_profit']:.4f}")
                return True
            
            # Place market order (implement actual trading logic here)
            logger.warning("LIVE TRADING ORDER WOULD BE PLACED HERE")
            # order = self.exchange.create_market_buy_order(symbol, quantity)
            # ... implement stop loss and take profit orders
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return False
    
    async def record_trade(self, trade_params: Dict, pattern: Dict, score: float):
        """Record trade for tracking and risk management"""
        try:
            trade_record = {
                'id': f"trade_{int(datetime.utcnow().timestamp())}",
                'symbol': trade_params['symbol'],
                'direction': trade_params['direction'],
                'size': trade_params['size'],
                'entry_price': trade_params['entry_price'],
                'stop_loss': trade_params['stop_loss'],
                'take_profit': trade_params['take_profit'],
                'pattern_id': trade_params['pattern_id'],
                'pattern_type': pattern.get('type', ''),
                'score': score,
                'status': 'open',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            self.open_trades.append(trade_record)
            self.trade_count_today += 1
            
            logger.info(f"Recorded trade: {trade_record['id']}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    async def check_sufficient_balance(self) -> bool:
        """Check if there's sufficient balance for trading"""
        try:
            if not self.exchange:
                logger.error("Exchange not available for balance check")
                return False
                
            balance = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_balance
            )
            
            available_usdt = balance.get('USDT', {}).get('free', 0)
            min_balance = float(os.getenv('MIN_TRADING_BALANCE', '100.0'))
            
            return float(available_usdt or 0) >= min_balance
            
        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            return False
    
    async def check_market_conditions(self) -> bool:
        """Check if market conditions are suitable for trading"""
        try:
            # Check volatility, liquidity, etc.
            # For now, always return True
            return True
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            return True
    
    def load_open_trades(self, trades: List[Dict]):
        """Load open trades from state"""
        self.open_trades = trades
    
    def get_open_trades(self) -> List[Dict]:
        """Get open trades for state management"""
        return self.open_trades
    
    async def monitor_trades(self):
        """Monitor open trades and handle exits (run periodically)"""
        try:
            if not self.open_trades:
                return
            
            for trade in self.open_trades[:]:  # Copy list for safe iteration
                if trade['status'] == 'open':
                    await self.check_trade_exit_conditions(trade)
            
        except Exception as e:
            logger.error(f"Error monitoring trades: {e}")
    
    async def check_trade_exit_conditions(self, trade: Dict):
        """Check if a trade should be closed"""
        try:
            if not self.exchange:
                logger.error("Exchange not available for checking trade exit conditions")
                return
                
            symbol = trade['symbol']
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_ticker, symbol
            )
            current_price = float(ticker['last'])
            
            # Check stop loss and take profit
            direction = trade['direction']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            
            should_close = False
            close_reason = ""
            
            if direction == 'long':
                if current_price <= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price >= take_profit:
                    should_close = True
                    close_reason = "take_profit"
            else:  # short
                if current_price >= stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif current_price <= take_profit:
                    should_close = True
                    close_reason = "take_profit"
            
            if should_close:
                await self.close_trade(trade, float(current_price), close_reason)
            
        except Exception as e:
            logger.error(f"Error checking trade exit conditions: {e}")
    
    async def close_trade(self, trade: Dict, exit_price: float, reason: str):
        """Close an open trade"""
        try:
            trade['status'] = 'closed'
            trade['exit_price'] = exit_price
            trade['exit_reason'] = reason
            trade['exit_timestamp'] = datetime.utcnow().isoformat() + 'Z'
            
            # Calculate PnL
            entry_price = trade['entry_price']
            size = trade['size']
            direction = trade['direction']
            
            if direction == 'long':
                pnl = size * (exit_price - entry_price) / entry_price
            else:  # short
                pnl = size * (entry_price - exit_price) / entry_price
            
            trade['pnl'] = pnl
            self.daily_pnl += pnl
            
            # Copy-specific logging
            if trade.get('is_copy', False):
                logger.info(f"Closed copy trade {trade['id']} from leader {trade.get('leader_id')}: {reason}, PnL: ${pnl:.2f}")
            else:
                logger.info(f"Closed trade {trade['id']}: {reason}, PnL: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
    
    async def copy_trade(self, leader_trade: Dict, user_risk_factor: float) -> Dict:
        """
        Execute a copy trade based on leader's trade with user-specific risk adjustment.
        """
        try:
            if not self.exchange:
                logger.error("Exchange not available for copy trading")
                return {'success': False, 'error': 'Exchange unavailable'}
            
            # Adjust size based on user risk
            adjusted_size = leader_trade['size'] * min(1.0, max(0.1, user_risk_factor))
            
            # Generate parameters
            copy_params = {
                'symbol': leader_trade['symbol'],
                'direction': leader_trade['direction'],
                'size': adjusted_size,
                'entry_price': leader_trade['entry_price'],
                'stop_loss': leader_trade['stop_loss'],
                'take_profit': leader_trade['take_profit'],
                'pattern_id': leader_trade.get('pattern_id', ''),
                'score': leader_trade.get('score', 0.0),
                'is_copy': True,
                'leader_id': leader_trade.get('id', '')
            }
            
            # Execute the trade
            success = await self.place_trade(copy_params)
            if success:
                await self.record_trade(copy_params, {}, copy_params['score'])
                logger.info(f"Copy trade executed: {copy_params.get('id', 'unknown')} from leader {copy_params['leader_id']}")
                return {'success': True, 'trade_id': copy_params.get('id'), 'adjusted_size': adjusted_size}
            else:
                return {'success': False, 'error': 'Trade placement failed'}
            
        except Exception as e:
            logger.error(f"Error in copy trading: {e}")
            return {'success': False, 'error': str(e)}
    
    def close_trade(self, trade_id: str) -> bool:
        """Close a trade by ID (synchronous wrapper for API)"""
        try:
            # Find the trade by ID
            trade = None
            for t in self.open_trades:
                if str(t.get('id')) == str(trade_id):
                    trade = t
                    break
            
            if not trade:
                logger.error(f"Trade {trade_id} not found")
                return False
            
            if trade.get('status') != 'open':
                logger.warning(f"Trade {trade_id} is not open")
                return False
            
            # Get current market price for exit
            try:
                if self.exchange:
                    ticker = self.exchange.fetch_ticker(trade['symbol'])
                    current_price = float(ticker['last'])
                else:
                    # Fallback if no exchange connection
                    current_price = trade.get('entry_price', 0)
                    
                # Update trade status
                trade['status'] = 'closed'
                trade['exit_price'] = current_price
                trade['exit_reason'] = 'manual_close'
                trade['exit_timestamp'] = datetime.utcnow().isoformat() + 'Z'
                
                # Calculate PnL
                entry_price = trade['entry_price']
                size = trade.get('size', trade.get('quantity', 0))
                direction = trade.get('direction', 'long')
                
                if direction == 'long':
                    pnl = size * (current_price - entry_price) / entry_price
                else:  # short
                    pnl = size * (entry_price - current_price) / entry_price
                
                trade['pnl'] = pnl
                self.daily_pnl += pnl
                
                logger.info(f"Manually closed trade {trade_id}: PnL: ${pnl:.2f}")
                return True
                
            except Exception as e:
                logger.error(f"Error getting market price for trade {trade_id}: {e}")
                # Still close the trade but with zero PnL
                trade['status'] = 'closed'
                trade['exit_price'] = trade.get('entry_price', 0)
                trade['exit_reason'] = 'manual_close_no_price'
                trade['exit_timestamp'] = datetime.utcnow().isoformat() + 'Z'
                trade['pnl'] = 0
                return True
                
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
            return False
    
    def get_trade_details(self, trade_id: str) -> Optional[Dict]:
        """Get detailed information about a specific trade"""
        try:
            # Search in open trades
            for trade in self.open_trades:
                if str(trade.get('id')) == str(trade_id):
                    return {
                        **trade,
                        'trade_type': 'open',
                        'duration': self._calculate_trade_duration(trade),
                        'current_pnl': self._calculate_current_pnl(trade)
                    }
            
            # Search in completed trades if we have them
            completed_trades = getattr(self, 'completed_trades', [])
            for trade in completed_trades:
                if str(trade.get('id')) == str(trade_id):
                    return {
                        **trade,
                        'trade_type': 'closed',
                        'duration': self._calculate_trade_duration(trade),
                        'final_pnl': trade.get('pnl', 0)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting trade details for {trade_id}: {e}")
            return None
    
    def _calculate_trade_duration(self, trade: Dict) -> str:
        """Calculate how long a trade has been open"""
        try:
            start_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            if trade.get('status') == 'closed' and trade.get('exit_timestamp'):
                end_time = datetime.fromisoformat(trade['exit_timestamp'].replace('Z', '+00:00'))
            else:
                end_time = datetime.utcnow().replace(tzinfo=start_time.tzinfo)
            
            duration = end_time - start_time
            hours = duration.total_seconds() / 3600
            
            if hours < 1:
                return f"{int(duration.total_seconds() / 60)}m"
            elif hours < 24:
                return f"{hours:.1f}h"
            else:
                return f"{hours / 24:.1f}d"
                
        except Exception as e:
            logger.error(f"Error calculating trade duration: {e}")
            return "Unknown"
    
    def _calculate_current_pnl(self, trade: Dict) -> float:
        """Calculate current PnL for an open trade"""
        try:
            if trade.get('status') != 'open' or not self.exchange:
                return trade.get('pnl', 0)
            
            ticker = self.exchange.fetch_ticker(trade['symbol'])
            current_price = float(ticker['last'])
            entry_price = trade['entry_price']
            size = trade.get('size', trade.get('quantity', 0))
            direction = trade.get('direction', 'long')
            
            if direction == 'long':
                return size * (current_price - entry_price) / entry_price
            else:  # short
                return size * (entry_price - current_price) / entry_price
                
        except Exception as e:
            logger.error(f"Error calculating current PnL: {e}")
            return 0.0
