"""
India Equity Scanner Module
Collects real-time data from NSE/BSE for NIFTY50, BANKNIFTY, and major Indian stocks
"""

import asyncio
import aiohttp
import logging
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)

class IndiaEquityScanner:
    """Scanner for Indian equity markets - NSE/BSE data"""
    
    def __init__(self):
        self.session = None
        self.last_scan_time = datetime.utcnow()
        
        # Key Indian market symbols for tracking
        self.key_symbols = {
            # Major Indices
            '^NSEI': 'NIFTY50',
            '^NSEBANK': 'BANKNIFTY', 
            '^BSESN': 'SENSEX',
            
            # Large Cap Stocks (Top 10 by market cap)
            'RELIANCE.NS': 'RELIANCE',
            'TCS.NS': 'TCS',
            'HDFCBANK.NS': 'HDFCBANK',
            'ICICIBANK.NS': 'ICICIBANK',
            'BHARTIARTL.NS': 'BHARTIARTL',
            'ITC.NS': 'ITC',
            'SBIN.NS': 'SBIN',
            'LT.NS': 'LT',
            'INFY.NS': 'INFY',
            'HINDUNILVR.NS': 'HINDUNILVR',
            
            # Sectoral Leaders
            'ADANIPORTS.NS': 'ADANIPORTS',
            'ASIANPAINT.NS': 'ASIANPAINT',
            'AXISBANK.NS': 'AXISBANK',
            'BAJAJ-AUTO.NS': 'BAJAJ-AUTO',
            'BAJFINANCE.NS': 'BAJFINANCE',
            'BAJAJFINSV.NS': 'BAJAJFINSV',
            'BPCL.NS': 'BPCL',
            'CIPLA.NS': 'CIPLA',
            'COALINDIA.NS': 'COALINDIA',
            'DIVISLAB.NS': 'DIVISLAB'
        }
        
        # Thresholds for significance
        self.min_price_change = 1.0  # Minimum 1% price change
        self.min_volume_ratio = 1.5  # 50% above average volume
        
    async def scan(self) -> List[Dict]:
        """Scan India equity markets for significant events"""
        try:
            events = []
            current_time = datetime.utcnow()
            
            logger.info("Starting India equity market scan...")
            
            # Get market data for all key symbols
            market_data = await self.get_market_data()
            
            for symbol, name in self.key_symbols.items():
                try:
                    if symbol in market_data:
                        data = market_data[symbol]
                        
                        # Analyze for significant events
                        price_events = await self.analyze_price_movement(symbol, name, data)
                        volume_events = await self.analyze_volume_patterns(symbol, name, data)
                        
                        events.extend(price_events)
                        events.extend(volume_events)
                        
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    continue
            
            # Update scan time
            self.last_scan_time = current_time
            
            logger.info(f"India equity scan completed: {len(events)} events")
            return events
            
        except Exception as e:
            logger.error(f"Error in India equity scan: {e}")
            return []
    
    async def get_market_data(self) -> Dict:
        """Fetch current market data for all tracked symbols"""
        try:
            market_data = {}
            
            # Use yfinance for reliable Indian market data
            symbols_list = list(self.key_symbols.keys())
            
            # Fetch data in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(symbols_list), batch_size):
                batch = symbols_list[i:i + batch_size]
                
                try:
                    # Get current data and 5-day history for trend analysis
                    tickers = yf.Tickers(' '.join(batch))
                    
                    for symbol in batch:
                        try:
                            ticker = tickers.tickers[symbol]
                            hist = ticker.history(period="5d", interval="1d")
                            info = ticker.info
                            
                            if not hist.empty and len(hist) >= 2:
                                current_price = float(hist['Close'].iloc[-1])
                                prev_price = float(hist['Close'].iloc[-2])
                                volume = float(hist['Volume'].iloc[-1])
                                avg_volume = float(hist['Volume'].mean())
                                
                                market_data[symbol] = {
                                    'current_price': current_price,
                                    'previous_price': prev_price,
                                    'volume': volume,
                                    'avg_volume': avg_volume,
                                    'market_cap': info.get('marketCap', 0),
                                    'sector': info.get('sector', 'Unknown'),
                                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                                }
                                
                        except Exception as e:
                            logger.warning(f"Error fetching data for {symbol}: {e}")
                            continue
                    
                    # Rate limiting between batches
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error fetching batch: {e}")
                    continue
            
            logger.info(f"Fetched data for {len(market_data)} India equity symbols")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching India market data: {e}")
            return {}
    
    async def analyze_price_movement(self, symbol: str, name: str, data: Dict) -> List[Dict]:
        """Analyze price movements for significant changes"""
        events = []
        
        try:
            current_price = data['current_price']
            prev_price = data['previous_price']
            
            # Calculate price change
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # Only create events for significant moves
            if abs(price_change) >= self.min_price_change:
                event = {
                    'id': f"india_equity_{symbol}_{int(datetime.utcnow().timestamp())}",
                    'timestamp': data['timestamp'],
                    'source': 'india_equity',
                    'type': 'price_movement',
                    'asset': name,
                    'payload': {
                        'symbol': symbol,
                        'name': name,
                        'current_price': current_price,
                        'previous_price': prev_price,
                        'price_change_percent': price_change,
                        'volume': data['volume'],
                        'market_cap': data['market_cap'],
                        'sector': data['sector'],
                        'type': 'large_cap' if symbol.startswith('^') else 'individual_stock',
                        'market': 'NSE' if symbol.endswith('.NS') else 'BSE',
                        'signal_strength': min(abs(price_change) / 5.0, 1.0)  # Normalize to 0-1
                    }
                }
                events.append(event)
                
        except Exception as e:
            logger.error(f"Error analyzing price movement for {symbol}: {e}")
        
        return events
    
    async def analyze_volume_patterns(self, symbol: str, name: str, data: Dict) -> List[Dict]:
        """Analyze volume patterns for unusual activity"""
        events = []
        
        try:
            volume = data['volume']
            avg_volume = data['avg_volume']
            
            if avg_volume > 0:
                volume_ratio = volume / avg_volume
                
                # Anomaly detection for volume
                anomaly = self.detect_volume_anomaly(volume_ratio, data)
                
                # Create event for unusual volume (50% above average)
                if volume_ratio >= self.min_volume_ratio:
                    event = {
                        'id': f"india_volume_{symbol}_{int(datetime.utcnow().timestamp())}",
                        'timestamp': data['timestamp'],
                        'source': 'india_equity',
                        'type': 'volume_spike',
                        'asset': name,
                        'payload': {
                            'symbol': symbol,
                            'name': name,
                            'volume': volume,
                            'avg_volume': avg_volume,
                            'volume_ratio': volume_ratio,
                            'unusual_activity': volume_ratio > 2.0,
                            'market_cap': data['market_cap'],
                            'sector': data['sector'],
                            'type': 'large_cap' if symbol.startswith('^') else 'individual_stock',
                            'market': 'NSE' if symbol.endswith('.NS') else 'BSE',
                            'signal_strength': min(volume_ratio / 3.0, 1.0)  # Normalize to 0-1
                        }
                    }
                    
                    # Add anomaly details if detected
                    if anomaly['anomaly']:
                        event['payload']['anomaly_detected'] = True
                        event['payload']['anomaly_details'] = anomaly
                    
                    events.append(event)
                    
        except Exception as e:
            logger.error(f"Error analyzing volume patterns for {symbol}: {e}")
        
        return events
    
    async def interactive_screen(self, filters: Dict) -> Dict:
        """
        Asynchronous screening with custom filters.
        """
        market_data = await self.get_market_data()
        screened = {}
        for symbol, data in market_data.items():
            if 'min_change' in filters and abs(data['current_price'] - data['previous_price']) / data['previous_price'] * 100 < filters['min_change']:
                continue
            if 'sector' in filters and data['sector'] != filters['sector']:
                continue
            screened[symbol] = data
        return screened
    
    def detect_volume_anomaly(self, ratio: float, data: Dict) -> Dict:
        """Detect volume anomalies."""
        return {'anomaly': ratio > self.min_volume_ratio * 1.5, 'score': ratio / self.min_volume_ratio}
    
    def get_status(self) -> Dict:
        """Get scanner status information with connectivity check"""
        connected = self._check_yfinance_connectivity()
        return {
            'name': 'India Equity Scanner',
            'active': connected,
            'connected': connected,  # Add for consistency with other scanners
            'last_scan': self.last_scan_time.isoformat() + 'Z',
            'symbols_tracked': len(self.key_symbols),
            'data_source': 'Yahoo Finance (yfinance)',
            'markets': ['NSE', 'BSE'],
            'key_indices': ['NIFTY50', 'BANKNIFTY', 'SENSEX'],
            'coverage': 'Top 25+ Indian stocks + major indices'
        }
    
    def _check_yfinance_connectivity(self) -> bool:
        """Test if Yahoo Finance API is accessible"""
        try:
            import yfinance as yf
            # Quick test with minimal data fetch
            test_ticker = yf.Ticker("^NSEI")  # NIFTY index - always available
            # Just check if we can get current price (fastest call)
            current_price = test_ticker.fast_info.get('lastPrice')
            return current_price is not None and current_price > 0
        except Exception as e:
            # Fallback check using requests to Yahoo Finance
            try:
                import requests
                response = requests.get("https://finance.yahoo.com/quote/%5ENSEI", timeout=2)
                return response.status_code == 200
            except:
                logger.warning(f"India equity scanner connectivity check failed: {e}")
                return False