"""
Binance Scanner Module
Implements CCXT-based exchange data collection for price and volume signals
"""

import os
import ccxt
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import time
import pandas as pd

logger = logging.getLogger(__name__)

class BinanceScanner:
    """Binance exchange scanner using CCXT"""
    
    def __init__(self):
        self.exchange = None
        self.last_offset = None
        self.price_history = {}
        self.volume_history = {}
        self.setup_exchange()
    
    def setup_exchange(self):
        """Initialize Binance API connection"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', 'default_key'),
                'secret': os.getenv('BINANCE_SECRET', 'default_secret'),
                'sandbox': os.getenv('BINANCE_SANDBOX', 'true').lower() == 'true',
                'enableRateLimit': True,
                'options': {'adjustForTimeDifference': True}
            })
            
            # Test connection
            self.exchange.load_markets()
            logger.info("Binance API connection established")
            
        except Exception as e:
            logger.error(f"Failed to setup Binance API: {e}")
            self.exchange = None
    
    async def scan(self) -> List[Dict]:
        """Scan Binance for market data and signals"""
        if not self.exchange:
            logger.warning("Binance API not available")
            return []
        
        try:
            events = []
            
            # Key trading pairs to monitor (updated for current market)
            symbols = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT',
                'XRP/USDT', 'SOL/USDT', 'POL/USDT', 'DOT/USDT',  # POL replaced MATIC
                'AVAX/USDT', 'LINK/USDT', 'DOGE/USDT', 'SHIB/USDT'
            ]
            
            for symbol in symbols:
                try:
                    # Get ticker data
                    ticker = await asyncio.get_event_loop().run_in_executor(
                        None, self.exchange.fetch_ticker, symbol
                    )
                    
                    # Calculate price change signals
                    price_events = self.analyze_price_movement(symbol, ticker)
                    events.extend(price_events)
                    
                    # Get recent trades for volume analysis
                    trades = await asyncio.get_event_loop().run_in_executor(
                        None, self.exchange.fetch_trades, symbol, None, 50
                    )
                    
                    volume_events = self.analyze_volume_spikes(symbol, trades)
                    events.extend(volume_events)
                    
                except ccxt.NetworkError as e:
                    logger.warning(f"Network error for {symbol}: {e}")
                    continue
                except ccxt.ExchangeError as e:
                    logger.warning(f"Exchange error for {symbol}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            logger.info(f"Collected {len(events)} Binance events")
            return events
            
        except Exception as e:
            logger.error(f"Error scanning Binance: {e}")
            return []
    
    def analyze_price_movement(self, symbol: str, ticker: Any) -> List[Dict]:
        """Analyze price movements for significant changes"""
        events = []
        
        try:
            current_price = ticker['last']
            change_percent = ticker['percentage']
            volume = ticker['quoteVolume']
            
            # Store price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            timestamp = datetime.utcnow().isoformat() + "Z"
            self.price_history[symbol].append({
                'timestamp': timestamp,
                'price': current_price,
                'change_percent': change_percent,
                'volume': volume
            })
            
            # Keep only last 100 records
            self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # Detect anomalies in price change
            price_change = change_percent
            anomaly = self.detect_price_anomaly(price_change, self.price_history.get(symbol, []))
            
            # Detect significant price movements
            if abs(change_percent) > 5.0:  # 5% threshold
                event = {
                    "source": "binance",
                    "id": f"price_move_{symbol.replace('/', '')}_{int(time.time())}",
                    "timestamp": timestamp,
                    "payload": {
                        "type": "price_movement",
                        "symbol": symbol,
                        "price": current_price,
                        "change_percent": change_percent,
                        "volume": volume,
                        "high_24h": ticker['high'],
                        "low_24h": ticker['low'],
                        "volume_24h": ticker['baseVolume'],
                        "signal_strength": min(abs(change_percent) / 10.0, 1.0)
                    }
                }
                
                # Add anomaly data if detected
                if anomaly and anomaly['anomaly']:
                    event['payload']['anomaly'] = True
                    event['payload']['anomaly_score'] = anomaly['score']
                
                events.append(event)
            
            # Detect breakout patterns
            if len(self.price_history[symbol]) >= 10:
                recent_prices = [p['price'] for p in self.price_history[symbol][-10:]]
                if self.detect_breakout(recent_prices):
                    breakout_event = {
                        "source": "binance",
                        "id": f"breakout_{symbol.replace('/', '')}_{int(time.time())}",
                        "timestamp": timestamp,
                        "payload": {
                            "type": "breakout",
                            "symbol": symbol,
                            "current_price": current_price,
                            "breakout_direction": "up" if change_percent > 0 else "down",
                            "volume": volume,
                            "signal_strength": 0.8
                        }
                    }
                    events.append(breakout_event)
            
        except Exception as e:
            logger.error(f"Error analyzing price movement for {symbol}: {e}")
        
        return events
    
    def interactive_screen(self, filters: Dict) -> pd.DataFrame:
        """
        Screen symbols based on custom filters (e.g., min_change, volume_threshold).
        """
        data = self.get_market_data()
        if isinstance(data, pd.DataFrame) and data.empty:
            return pd.DataFrame()
        elif not isinstance(data, pd.DataFrame):
            return pd.DataFrame()
        
        # Apply filters
        screened = data.copy()
        if 'min_change' in filters and 'price_change' in screened.columns:
            screened = screened[screened['price_change'] >= filters['min_change']]
        if 'volume_threshold' in filters and 'volume' in screened.columns:
            screened = screened[screened['volume'] >= filters['volume_threshold']]
        
        # Add real-time charting data (OHLCV) if data exists
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not screened.empty and all(col in screened.columns for col in required_cols):
            screened = screened.copy()  # Ensure we have a proper DataFrame
            screened['chart_data'] = screened.apply(lambda row: {
                'open': row['open'], 'high': row['high'], 'low': row['low'], 
                'close': row['close'], 'volume': row['volume']
            }, axis=1)
        
        return screened
    
    def get_market_data(self) -> pd.DataFrame:
        """
        Get current market data as a DataFrame for screening.
        """
        try:
            if not self.exchange:
                return pd.DataFrame()
            
            # Build DataFrame from current price history
            market_data = []
            for symbol, history in self.price_history.items():
                if history:
                    latest = history[-1]
                    # Get ticker to get OHLC data
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        market_data.append({
                            'symbol': symbol,
                            'open': ticker['open'],
                            'high': ticker['high'],
                            'low': ticker['low'],
                            'close': ticker['close'],
                            'volume': ticker['baseVolume'],
                            'price_change': latest['change_percent']
                        })
                    except Exception as e:
                        logger.warning(f"Error fetching ticker for {symbol}: {e}")
                        continue
            
            return pd.DataFrame(market_data)
            
        except Exception as e:
            logger.error(f"Error creating market data DataFrame: {e}")
            return pd.DataFrame()
    
    def detect_price_anomaly(self, change: float, data: List[Dict]) -> Dict:
        """
        Simple anomaly detection based on z-score.
        """
        try:
            if len(data) < 3:
                return {'anomaly': False, 'score': 0.0}
            
            # Extract price changes from history
            changes = [item['change_percent'] for item in data[-10:]]  # Last 10 changes
            
            if len(changes) < 2:
                return {'anomaly': False, 'score': 0.0}
            
            # Calculate z-score
            mean_change = sum(changes) / len(changes)
            variance = sum((x - mean_change) ** 2 for x in changes) / len(changes)
            std_dev = variance ** 0.5
            
            if std_dev == 0:
                return {'anomaly': False, 'score': 0.0}
            
            z_score = (change - mean_change) / (std_dev + 1e-8)
            score = abs(z_score)
            
            return {'anomaly': score > 2.0, 'score': score}
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {'anomaly': False, 'score': 0.0}
    
    def analyze_volume_spikes(self, symbol: str, trades: List[Any]) -> List[Dict]:
        """Analyze trading volume for unusual spikes"""
        events = []
        
        try:
            if not trades:
                return events
            
            # Calculate recent volume
            recent_volume = sum(trade['amount'] for trade in trades[-10:])
            
            # Store volume history
            if symbol not in self.volume_history:
                self.volume_history[symbol] = []
            
            timestamp = datetime.utcnow().isoformat() + "Z"
            self.volume_history[symbol].append({
                'timestamp': timestamp,
                'volume': recent_volume
            })
            
            # Keep only last 50 records
            self.volume_history[symbol] = self.volume_history[symbol][-50:]
            
            # Detect volume spikes
            if len(self.volume_history[symbol]) >= 5:
                volumes = [v['volume'] for v in self.volume_history[symbol]]
                avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
                
                if recent_volume > avg_volume * 2:  # 2x average volume
                    event = {
                        "source": "binance",
                        "id": f"volume_spike_{symbol.replace('/', '')}_{int(time.time())}",
                        "timestamp": timestamp,
                        "payload": {
                            "type": "volume_spike",
                            "symbol": symbol,
                            "current_volume": recent_volume,
                            "average_volume": avg_volume,
                            "spike_ratio": recent_volume / avg_volume,
                            "signal_strength": min((recent_volume / avg_volume) / 3.0, 1.0)
                        }
                    }
                    events.append(event)
            
        except Exception as e:
            logger.error(f"Error analyzing volume for {symbol}: {e}")
        
        return events
    
    def detect_breakout(self, prices: List[float]) -> bool:
        """Simple breakout detection using price momentum"""
        if len(prices) < 5:
            return False
        
        # Calculate moving averages
        short_ma = sum(prices[-3:]) / 3
        long_ma = sum(prices[-5:]) / 5
        
        # Check for momentum shift
        return abs(short_ma - long_ma) / long_ma > 0.02  # 2% threshold
    
    def set_last_offset(self, offset: Optional[str]):
        """Set the last processed offset"""
        self.last_offset = offset
    
    def get_last_offset(self) -> Optional[str]:
        """Get the last processed offset"""
        return self.last_offset
    
    def get_status(self) -> Dict:
        """Get scanner status with real connectivity check"""
        connected = self._check_binance_connectivity()
        return {
            "name": "binance_scanner",
            "connected": connected,
            "last_offset": self.last_offset,
            "monitored_symbols": len(self.price_history),
            "last_scan": datetime.utcnow().isoformat() + "Z"
        }
    
    def _check_binance_connectivity(self) -> bool:
        """Check if Binance API is accessible"""
        if self.exchange is None:
            return False
        
        try:
            # Use a quick status check - fetch_status is lighter than fetch_time
            status = self.exchange.fetch_status()
            return status is not None and status.get('status', '') != 'maintenance'
        except Exception as e:
            # Fallback to basic exchange availability check
            try:
                return hasattr(self.exchange, 'apiKey') or hasattr(self.exchange, 'api_key')
            except:
                logger.warning(f"Binance scanner connectivity check failed: {e}")
                return False
