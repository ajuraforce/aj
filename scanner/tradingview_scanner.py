"""
TradingView Scanner - Technical Analysis Data Collection
Collects technical analysis signals from TradingView using tradingview-ta library
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
import os
from concurrent.futures import ThreadPoolExecutor

try:
    from tradingview_ta import TA_Handler, Interval, Exchange, get_multiple_analysis
    TRADINGVIEW_AVAILABLE = True
except ImportError:
    TRADINGVIEW_AVAILABLE = False
    logging.warning("TradingView TA library not available. Install with: pip install tradingview-ta")

logger = logging.getLogger(__name__)

class TradingViewScanner:
    """TradingView technical analysis scanner for market signals"""
    
    def __init__(self):
        self.last_offset = None
        self.signal_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Default symbols to monitor
        self.default_symbols = [
            # Crypto pairs
            {"symbol": "BTCUSDT", "screener": "crypto", "exchange": "BINANCE"},
            {"symbol": "ETHUSDT", "screener": "crypto", "exchange": "BINANCE"},
            {"symbol": "ADAUSDT", "screener": "crypto", "exchange": "BINANCE"},
            {"symbol": "SOLUSDT", "screener": "crypto", "exchange": "BINANCE"},
            {"symbol": "DOGEUSDT", "screener": "crypto", "exchange": "BINANCE"},
            
            # Major US stocks
            {"symbol": "AAPL", "screener": "america", "exchange": "NASDAQ"},
            {"symbol": "TSLA", "screener": "america", "exchange": "NASDAQ"},
            {"symbol": "MSFT", "screener": "america", "exchange": "NASDAQ"},
            {"symbol": "GOOGL", "screener": "america", "exchange": "NASDAQ"},
            {"symbol": "NVDA", "screener": "america", "exchange": "NASDAQ"},
            
            # Forex pairs
            {"symbol": "EURUSD", "screener": "forex", "exchange": "FX_IDC"},
            {"symbol": "GBPUSD", "screener": "forex", "exchange": "FX_IDC"},
            {"symbol": "USDJPY", "screener": "forex", "exchange": "FX_IDC"},
        ]
        
        # TradingView configuration
        self.intervals = [
            Interval.INTERVAL_15_MINUTES,
            Interval.INTERVAL_1_HOUR,
            Interval.INTERVAL_4_HOURS,
            Interval.INTERVAL_1_DAY
        ]
        
        # Signal thresholds
        self.strong_buy_threshold = 8  # Number of BUY signals for strong buy
        self.strong_sell_threshold = 8  # Number of SELL signals for strong sell
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        if not TRADINGVIEW_AVAILABLE:
            logger.error("TradingView TA library not available")
    
    async def scan(self) -> List[Dict[str, Any]]:
        """Main scanning method - collects TradingView technical analysis signals"""
        if not TRADINGVIEW_AVAILABLE:
            logger.warning("TradingView TA not available, returning empty results")
            return []
        
        try:
            logger.info("Starting TradingView technical analysis scan...")
            events = []
            
            # Process symbols in batches for better performance
            batch_size = 5
            symbol_batches = [
                self.default_symbols[i:i + batch_size] 
                for i in range(0, len(self.default_symbols), batch_size)
            ]
            
            for batch in symbol_batches:
                batch_events = await self._process_symbol_batch(batch)
                events.extend(batch_events)
                
                # Small delay between batches to avoid rate limiting
                await asyncio.sleep(1)
            
            logger.info(f"TradingView scan completed: {len(events)} events collected")
            return events
            
        except Exception as e:
            logger.error(f"Error in TradingView scan: {e}")
            return []
    
    async def _process_symbol_batch(self, symbols: List[Dict]) -> List[Dict[str, Any]]:
        """Process a batch of symbols for technical analysis"""
        batch_events = []
        
        for symbol_config in symbols:
            try:
                # Get analysis for multiple timeframes
                symbol_events = await self._analyze_symbol(symbol_config)
                batch_events.extend(symbol_events)
                
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol_config['symbol']}: {e}")
                continue
        
        return batch_events
    
    async def _analyze_symbol(self, symbol_config: Dict) -> List[Dict[str, Any]]:
        """Analyze a single symbol across multiple timeframes"""
        events = []
        symbol = symbol_config['symbol']
        screener = symbol_config['screener']
        exchange = symbol_config['exchange']
        
        try:
            # Analyze primary timeframe (1 hour)
            handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_HOUR
            )
            
            # Get analysis in thread executor to avoid blocking
            analysis = await asyncio.get_event_loop().run_in_executor(
                self.executor, handler.get_analysis
            )
            
            # Process the analysis results
            signal_event = self._create_signal_event(symbol_config, analysis, "1H")
            if signal_event:
                events.append(signal_event)
            
            # Also get daily analysis for longer-term perspective
            daily_handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval=Interval.INTERVAL_1_DAY
            )
            
            daily_analysis = await asyncio.get_event_loop().run_in_executor(
                self.executor, daily_handler.get_analysis
            )
            
            daily_event = self._create_signal_event(symbol_config, daily_analysis, "1D")
            if daily_event:
                events.append(daily_event)
            
        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {e}")
        
        return events
    
    def _create_signal_event(self, symbol_config: Dict, analysis, timeframe: str) -> Dict[str, Any]:
        """Create a trading signal event from TradingView analysis"""
        try:
            summary = analysis.summary
            oscillators = analysis.oscillators
            moving_averages = analysis.moving_averages
            indicators = analysis.indicators
            
            # Calculate signal strength
            total_signals = summary['BUY'] + summary['SELL'] + summary['NEUTRAL']
            if total_signals == 0:
                return None
            
            buy_strength = summary['BUY'] / total_signals * 100
            sell_strength = summary['SELL'] / total_signals * 100
            confidence = max(buy_strength, sell_strength)
            
            # Determine overall signal
            recommendation = summary['RECOMMENDATION']
            
            # Get key technical indicators
            rsi = indicators.get('RSI', 50)
            macd = indicators.get('MACD.macd', 0)
            bb_upper = indicators.get('BB.upper', 0)
            bb_lower = indicators.get('BB.lower', 0)
            bb_middle = indicators.get('BB.middle', 0)
            
            # Create signal event
            event = {
                'id': f"tradingview_{symbol_config['symbol']}_{timeframe}_{int(datetime.utcnow().timestamp())}",
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'source': 'tradingview',
                'type': 'technical_analysis',
                'asset': symbol_config['symbol'],
                'screener': symbol_config['screener'],
                'exchange': symbol_config['exchange'],
                'timeframe': timeframe,
                'confidence': confidence / 100,  # Normalize to 0-1
                'signals': {
                    'recommendation': recommendation,
                    'buy_signals': summary['BUY'],
                    'sell_signals': summary['SELL'],
                    'neutral_signals': summary['NEUTRAL'],
                    'buy_strength': buy_strength,
                    'sell_strength': sell_strength,
                    'oscillators': {
                        'recommendation': oscillators['RECOMMENDATION'],
                        'buy': oscillators['BUY'],
                        'sell': oscillators['SELL'],
                        'neutral': oscillators['NEUTRAL']
                    },
                    'moving_averages': {
                        'recommendation': moving_averages['RECOMMENDATION'],
                        'buy': moving_averages['BUY'],
                        'sell': moving_averages['SELL'],
                        'neutral': moving_averages['NEUTRAL']
                    },
                    'indicators': {
                        'rsi': rsi,
                        'macd': macd,
                        'bb_upper': bb_upper,
                        'bb_middle': bb_middle,
                        'bb_lower': bb_lower,
                        'rsi_signal': self._get_rsi_signal(rsi),
                        'macd_signal': 'bullish' if macd > 0 else 'bearish'
                    }
                }
            }
            
            # Add alert conditions
            alerts = []
            if rsi < self.rsi_oversold:
                alerts.append(f"RSI oversold at {rsi:.1f}")
            elif rsi > self.rsi_overbought:
                alerts.append(f"RSI overbought at {rsi:.1f}")
            
            if recommendation in ['STRONG_BUY', 'BUY'] and confidence > 70:
                alerts.append(f"Strong {recommendation.lower()} signal with {confidence:.1f}% confidence")
            elif recommendation in ['STRONG_SELL', 'SELL'] and confidence > 70:
                alerts.append(f"Strong {recommendation.lower()} signal with {confidence:.1f}% confidence")
            
            if alerts:
                event['alerts'] = alerts
                event['priority'] = 'high' if confidence > 80 else 'medium'
            
            return event
            
        except Exception as e:
            logger.error(f"Error creating signal event: {e}")
            return None
    
    def _get_rsi_signal(self, rsi: float) -> str:
        """Get RSI signal interpretation"""
        if rsi < self.rsi_oversold:
            return 'oversold'
        elif rsi > self.rsi_overbought:
            return 'overbought'
        elif rsi < 45:
            return 'bearish'
        elif rsi > 55:
            return 'bullish'
        else:
            return 'neutral'
    
    def get_status(self) -> Dict[str, Any]:
        """Get scanner status"""
        return {
            "name": "TradingView Scanner",
            "enabled": TRADINGVIEW_AVAILABLE,
            "symbols_monitored": len(self.default_symbols),
            "last_scan": self.last_offset,
            "cache_size": len(self.signal_cache)
        }
    
    def set_last_offset(self, offset):
        """Set last offset for state management"""
        self.last_offset = offset
    
    def get_last_offset(self):
        """Get last offset for state management"""
        return self.last_offset or datetime.utcnow().isoformat() + 'Z'
    
    def add_symbol(self, symbol: str, screener: str = "america", exchange: str = "NASDAQ"):
        """Add a new symbol to monitor"""
        symbol_config = {
            "symbol": symbol,
            "screener": screener,
            "exchange": exchange
        }
        
        if symbol_config not in self.default_symbols:
            self.default_symbols.append(symbol_config)
            logger.info(f"Added {symbol} to TradingView monitoring")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from monitoring"""
        self.default_symbols = [
            s for s in self.default_symbols 
            if s['symbol'] != symbol
        ]
        logger.info(f"Removed {symbol} from TradingView monitoring")
    
    async def get_symbol_analysis(self, symbol: str, screener: str = "america", 
                                 exchange: str = "NASDAQ", interval: str = "1h") -> Dict:
        """Get detailed analysis for a specific symbol"""
        try:
            interval_map = {
                "1m": Interval.INTERVAL_1_MINUTE,
                "5m": Interval.INTERVAL_5_MINUTES,
                "15m": Interval.INTERVAL_15_MINUTES,
                "30m": Interval.INTERVAL_30_MINUTES,
                "1h": Interval.INTERVAL_1_HOUR,
                "2h": Interval.INTERVAL_2_HOURS,
                "4h": Interval.INTERVAL_4_HOURS,
                "1d": Interval.INTERVAL_1_DAY,
                "1w": Interval.INTERVAL_1_WEEK,
                "1M": Interval.INTERVAL_1_MONTH
            }
            
            handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval=interval_map.get(interval, Interval.INTERVAL_1_HOUR)
            )
            
            analysis = await asyncio.get_event_loop().run_in_executor(
                self.executor, handler.get_analysis
            )
            
            return {
                "symbol": symbol,
                "screener": screener,
                "exchange": exchange,
                "interval": interval,
                "analysis": {
                    "summary": analysis.summary,
                    "oscillators": analysis.oscillators,
                    "moving_averages": analysis.moving_averages,
                    "indicators": analysis.indicators
                },
                "timestamp": datetime.utcnow().isoformat() + 'Z'
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis for {symbol}: {e}")
            return {"error": str(e)}