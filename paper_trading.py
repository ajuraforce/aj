# paper_trading.py
import sqlite3
import redis
import json
import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import ccxt
from utils.signal_schema import Signal

logger = logging.getLogger(__name__)

class PaperTradingEngine:
    """Paper trading engine that consumes signals and manages virtual trades"""
    
    def __init__(self, db_path="patterns.db"):
        self.db_path = db_path
        self.running = False
        self.consumer_thread = None
        
        # Redis connection
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            logger.info("Paper trading Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available for paper trading: {e}")
            self.redis_client = None
        
        # Binance client for price data (read-only)
        try:
            self.exchange = ccxt.binance({'sandbox': True})  # Use testnet for safety
            logger.info("Binance connection established for price data")
        except Exception as e:
            logger.warning(f"Binance connection failed: {e}")
            self.exchange = None
        
        # Initialize database
        self.init_paper_trades_db()
        
    def init_paper_trades_db(self):
        """Initialize paper trading database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Paper trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    size REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'OPEN',
                    linked_alert TEXT,
                    entry_time DATETIME NOT NULL,
                    exit_time DATETIME,
                    exit_price REAL,
                    pnl REAL DEFAULT 0.0,
                    confidence REAL,
                    timeframe TEXT,
                    reasons TEXT
                )
            ''')
            
            # Signals table for tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT,
                    created_at DATETIME NOT NULL,
                    expires_in INTEGER DEFAULT 600,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Paper trading database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing paper trading database: {e}")
    
    def start_consumer(self):
        """Start the signal consumer in a background thread"""
        if self.running:
            logger.warning("Paper trading consumer already running")
            return
            
        self.running = True
        self.consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_thread.start()
        logger.info("Paper trading consumer started")
    
    def stop_consumer(self):
        """Stop the signal consumer"""
        self.running = False
        if self.consumer_thread:
            self.consumer_thread.join(timeout=5)
        logger.info("Paper trading consumer stopped")
    
    def _consumer_loop(self):
        """Main consumer loop for processing signals"""
        while self.running:
            try:
                if not self.redis_client:
                    time.sleep(5)
                    continue
                
                # Block for new signals with timeout
                result = self.redis_client.blpop(['signals_queue'], timeout=5)
                if not result:
                    continue
                
                queue_name, signal_json = result
                signal_data = json.loads(signal_json)
                signal = Signal(**signal_data)
                
                # Store signal in database
                self._store_signal(signal)
                
                # Process signal for paper trading
                if self._should_trade_signal(signal):
                    self._create_paper_trade(signal)
                
                # Monitor existing trades
                self._monitor_open_trades()
                
            except Exception as e:
                logger.error(f"Error in paper trading consumer: {e}")
                time.sleep(1)
    
    def _store_signal(self, signal: Signal):
        """Store signal in database for tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO signals 
                (id, symbol, timeframe, signal_type, confidence, reason, created_at, expires_in)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.id, signal.symbol, signal.timeframe, signal.signal_type,
                signal.confidence, json.dumps(signal.reason), signal.created_at,
                signal.expires_in
            ))
            conn.commit()
            conn.close()
            logger.debug(f"Signal {signal.id} stored in database")
            
        except Exception as e:
            logger.error(f"Error storing signal: {e}")
    
    def _should_trade_signal(self, signal: Signal) -> bool:
        """Determine if signal should generate a paper trade"""
        try:
            # Load risk rules from permissions.json
            risk_rules = self._load_risk_rules()
            
            # Check minimum confidence threshold
            min_confidence = risk_rules.get('paper_trading', {}).get('min_confidence', 0.6)
            if signal.confidence < min_confidence:
                logger.debug(f"Signal {signal.id} below confidence threshold: {signal.confidence}")
                return False
            
            # Check maximum exposure per symbol
            if self._check_symbol_exposure(signal.symbol):
                logger.debug(f"Symbol {signal.symbol} has maximum exposure")
                return False
            
            # Check if signal is expired
            if signal.expires_in > 0:
                signal_age = (datetime.utcnow() - signal.created_at).total_seconds()
                if signal_age > signal.expires_in:
                    logger.debug(f"Signal {signal.id} expired")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trade signal: {e}")
            return False
    
    def _load_risk_rules(self) -> Dict:
        """Load risk management rules from permissions.json"""
        try:
            with open('permissions.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load risk rules: {e}")
            return {
                'paper_trading': {
                    'min_confidence': 0.6,
                    'max_trades_per_symbol': 3,
                    'max_position_size': 1000,
                    'stop_loss_percent': 2.0,
                    'take_profit_percent': 2.0
                }
            }
    
    def _check_symbol_exposure(self, symbol: str) -> bool:
        """Check if symbol has reached maximum open trades"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE symbol = ? AND status = 'OPEN'",
                (symbol,)
            )
            open_trades = cursor.fetchone()[0]
            conn.close()
            
            risk_rules = self._load_risk_rules()
            max_trades = risk_rules.get('paper_trading', {}).get('max_trades_per_symbol', 3)
            
            return open_trades >= max_trades
            
        except Exception as e:
            logger.error(f"Error checking symbol exposure: {e}")
            return True  # Conservative: block if error
    
    def _create_paper_trade(self, signal: Signal):
        """Create a paper trade from signal"""
        try:
            # Generate trade ID
            trade_id = f"T-{signal.symbol.replace('/', '')}-{int(time.time())}"
            
            # Get current price
            entry_price = self._fetch_current_price(signal.symbol)
            if not entry_price:
                logger.error(f"Could not fetch price for {signal.symbol}")
                return
            
            # Calculate position size
            risk_rules = self._load_risk_rules()
            max_position = risk_rules.get('paper_trading', {}).get('max_position_size', 1000)
            size = self._calculate_position_size(signal, max_position)
            
            # Calculate stop loss and take profit
            sl_percent = risk_rules.get('paper_trading', {}).get('stop_loss_percent', 2.0) / 100
            tp_percent = risk_rules.get('paper_trading', {}).get('take_profit_percent', 2.0) / 100
            
            if signal.signal_type in ['BUY', 'LONG']:
                stop_loss = entry_price * (1 - sl_percent)
                take_profit = entry_price * (1 + tp_percent)
            else:  # SELL/SHORT
                stop_loss = entry_price * (1 + sl_percent)
                take_profit = entry_price * (1 - tp_percent)
            
            # Store trade in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO paper_trades 
                (trade_id, symbol, side, entry_price, size, stop_loss, take_profit, 
                 linked_alert, entry_time, confidence, timeframe, reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, signal.symbol, signal.signal_type, entry_price, size,
                stop_loss, take_profit, signal.id, datetime.utcnow(),
                signal.confidence, signal.timeframe, json.dumps(signal.reason)
            ))
            conn.commit()
            conn.close()
            
            logger.info(f"Paper trade created: {trade_id} - {signal.signal_type} {size} {signal.symbol} @ {entry_price}")
            
        except Exception as e:
            logger.error(f"Error creating paper trade: {e}")
    
    def _calculate_position_size(self, signal: Signal, max_position: float) -> float:
        """Calculate position size based on confidence and risk"""
        try:
            # Base size proportional to confidence
            base_size = max_position * signal.confidence
            
            # Adjust for signal type (more conservative for shorts)
            if signal.signal_type in ['SHORT', 'SELL']:
                base_size *= 0.7
            
            return round(base_size, 6)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return max_position * 0.1  # Conservative fallback
    
    def _fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for symbol"""
        try:
            if self.exchange:
                # Convert symbol format if needed (BTC/USDT -> BTCUSDT)
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker.get('last', 0)
                return float(price) if price else None
            else:
                # Fallback: mock price based on symbol
                return self._mock_price(symbol)
                
        except Exception as e:
            logger.warning(f"Error fetching price for {symbol}: {e}")
            return self._mock_price(symbol)
    
    def _mock_price(self, symbol: str) -> float:
        """Mock price data for development"""
        mock_prices = {
            'BTC/USDT': 59500.0,
            'ETH/USDT': 3200.0,
            'SOL/USDT': 150.0,
            'ADA/USDT': 0.85,
            'DOT/USDT': 12.5
        }
        return float(mock_prices.get(symbol, 100.0))
    
    def _monitor_open_trades(self):
        """Monitor open trades for stop loss and take profit"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM paper_trades WHERE status = 'OPEN'")
            open_trades = cursor.fetchall()
            conn.close()
            
            for trade_data in open_trades:
                self._check_trade_conditions(trade_data)
                
        except Exception as e:
            logger.error(f"Error monitoring trades: {e}")
    
    def _check_trade_conditions(self, trade_data):
        """Check individual trade for SL/TP conditions"""
        try:
            trade_id, symbol, side, entry_price, size, stop_loss, take_profit = trade_data[:7]
            
            current_price = self._fetch_current_price(symbol)
            if not current_price:
                return
            
            # Calculate current P&L
            pnl = self._calculate_pnl(trade_data, current_price)
            
            # Check stop loss and take profit conditions
            should_close = False
            exit_reason = ""
            
            if side in ['BUY', 'LONG']:
                if current_price <= stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif current_price >= take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
            else:  # SELL/SHORT
                if current_price >= stop_loss:
                    should_close = True
                    exit_reason = "Stop Loss"
                elif current_price <= take_profit:
                    should_close = True
                    exit_reason = "Take Profit"
            
            if should_close:
                self._close_trade(trade_id, current_price, pnl, exit_reason)
                
        except Exception as e:
            logger.error(f"Error checking trade conditions: {e}")
    
    def _calculate_pnl(self, trade_data, current_price: float) -> float:
        """Calculate current P&L for trade"""
        try:
            _, _, side, entry_price, size = trade_data[:5]
            
            if side in ['BUY', 'LONG']:
                return (current_price - entry_price) * size
            else:  # SELL/SHORT
                return (entry_price - current_price) * size
                
        except Exception as e:
            logger.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def _close_trade(self, trade_id: str, exit_price: float, pnl: float, reason: str):
        """Close paper trade with exit conditions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE paper_trades 
                SET status = 'CLOSED', exit_time = ?, exit_price = ?, pnl = ?
                WHERE trade_id = ?
            ''', (datetime.utcnow(), exit_price, pnl, trade_id))
            conn.commit()
            conn.close()
            
            logger.info(f"Closed trade {trade_id} - {reason} - P&L: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
    
    def get_paper_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent paper trades"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM paper_trades 
                ORDER BY entry_time DESC 
                LIMIT ?
            ''', (limit,))
            trades = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching paper trades: {e}")
            return []
    
    def get_analytics(self) -> Dict:
        """Get paper trading analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN status = 'OPEN' THEN 1 END) as open_trades
                FROM paper_trades
            ''')
            
            stats = cursor.fetchone()
            
            # Calculate win rate
            total = stats[0] if stats[0] else 1
            win_rate = (stats[1] / total * 100) if stats[1] else 0
            
            # Recent performance (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            cursor.execute('''
                SELECT SUM(pnl) as weekly_pnl, COUNT(*) as weekly_trades
                FROM paper_trades 
                WHERE entry_time >= ?
            ''', (week_ago,))
            
            weekly_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_trades': stats[0] or 0,
                'winning_trades': stats[1] or 0,
                'losing_trades': stats[2] or 0,
                'win_rate': round(win_rate, 2),
                'avg_pnl': round(stats[3] or 0, 2),
                'total_pnl': round(stats[4] or 0, 2),
                'avg_confidence': round(stats[5] or 0, 3),
                'open_trades': stats[6] or 0,
                'weekly_pnl': round(weekly_stats[0] or 0, 2),
                'weekly_trades': weekly_stats[1] or 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating analytics: {e}")
            return {}
    
    def get_signals(self, limit: int = 20) -> List[Dict]:
        """Get recent signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM signals 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            signals = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            # Parse JSON reason field
            for signal in signals:
                try:
                    signal['reason'] = json.loads(signal['reason'])
                except:
                    signal['reason'] = [signal['reason']] if signal['reason'] else []
            
            return signals
            
        except Exception as e:
            logger.error(f"Error fetching signals: {e}")
            return []

# Global instance for the application
paper_trading_engine = PaperTradingEngine()

def start_paper_trading():
    """Start the paper trading engine"""
    paper_trading_engine.start_consumer()

def stop_paper_trading():
    """Stop the paper trading engine"""
    paper_trading_engine.stop_consumer()

def get_paper_trades_data():
    """Get paper trades for API"""
    return paper_trading_engine.get_paper_trades()

def get_paper_analytics_data():
    """Get analytics for API"""
    return paper_trading_engine.get_analytics()

def get_live_signals_data():
    """Get live signals for API"""
    return paper_trading_engine.get_signals()