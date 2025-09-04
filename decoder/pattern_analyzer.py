"""
Pattern Analyzer Module
Implements correlation analysis and pattern detection for decoded signals
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
from collections import defaultdict, deque
import json
import sqlite3
import os
import re
import math
from flask import current_app

logger = logging.getLogger(__name__)

# Import database models - will be set when app context is available
db = None
PatternOutcome = None
AssetMention = None
Correlation = None

def init_db_models():
    """Initialize database models when app context is available"""
    global db, PatternOutcome, AssetMention, Correlation
    try:
        from models import db, PatternOutcome, AssetMention, Correlation
    except ImportError:
        logger.warning("Database models not available - falling back to SQLite")

class PatternAnalyzer:
    """Analyzes patterns and correlations across data sources"""
    
    def __init__(self):
        self.correlations = {}
        self.event_windows = defaultdict(lambda: deque(maxlen=100))
        self.asset_mentions = defaultdict(lambda: deque(maxlen=50))
        self.sentiment_history = defaultdict(lambda: deque(maxlen=30))
        self.pattern_outcomes = {}  # For feedback loop
        self.sector_mentions = defaultdict(lambda: deque(maxlen=30))
        
        # Load configuration files
        self.config = self.load_config()
        self.assets_config = self.load_assets_config()
        self.sectors_config = self.load_sectors_config()
        
        # Legacy asset map for backward compatibility
        self.asset_map = self.build_legacy_asset_map()
        
        # Initialize database models if available
        try:
            init_db_models()
        except:
            logger.info("Using SQLite fallback for pattern storage")
            
        # Load persistent state
        self.load_state()
    
    async def analyze(self, events: List[Dict]) -> List[Dict]:
        """Analyze events for patterns and correlations"""
        try:
            patterns = []
            
            for event in events:
                # Process each event type
                if event['source'] == 'reddit':
                    reddit_patterns = await self.analyze_reddit_event(event)
                    patterns.extend(reddit_patterns)
                elif event['source'] == 'binance':
                    binance_patterns = await self.analyze_binance_event(event)
                    patterns.extend(binance_patterns)
                elif event['source'] == 'india_equity':
                    # India equity events have same structure as Binance events
                    india_patterns = await self.analyze_binance_event(event)
                    patterns.extend(india_patterns)
                elif event['source'] == 'news':
                    news_patterns = await self.analyze_news_event(event)
                    patterns.extend(news_patterns)
            
            # Cross-source correlation analysis
            correlation_patterns = await self.analyze_cross_correlations()
            patterns.extend(correlation_patterns)
            
            logger.info(f"Generated {len(patterns)} patterns from {len(events)} events")
            
            # **CRITICAL FIX: Save patterns to database**
            if patterns:
                self.save_patterns_to_db(patterns)
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return []
    
    async def analyze_reddit_event(self, event: Dict) -> List[Dict]:
        """Analyze Reddit events for social patterns"""
        patterns = []
        
        try:
            payload = event['payload']
            timestamp = event['timestamp']
            
            # Extract mentioned assets/symbols
            mentioned_assets = self.extract_assets_from_text(
                payload.get('title', '') + ' ' + payload.get('body', '') + ' ' + payload.get('selftext', '')
            )
            
            for asset in mentioned_assets:
                # Track mention frequency
                self.asset_mentions[asset].append({
                    'timestamp': timestamp,
                    'score': payload.get('score', 0),
                    'source': 'reddit',
                    'subreddit': payload.get('subreddit', ''),
                    'type': payload.get('type', 'submission')
                })
                
                # Detect mention velocity spikes using adaptive thresholds
                if self.detect_mention_spike(asset):
                        pattern = {
                            'id': f'reddit_mention_spike_{asset}_{timestamp}',
                            'timestamp': timestamp,
                            'type': 'mention_spike',
                            'asset': asset,
                            'source': 'reddit',
                            'signals': {
                                'mention_count': len([m for m in self.asset_mentions[asset] if self.is_recent(m['timestamp'], hours=1)]),
                                'avg_score': np.mean([m['score'] for m in self.asset_mentions[asset]]),
                                'velocity': len([m for m in self.asset_mentions[asset] if self.is_recent(m['timestamp'], hours=1)]) / 1.0,
                                'cross_subreddit': len(set(m['subreddit'] for m in self.asset_mentions[asset])),
                                'sentiment_shift': self.calculate_sentiment_shift(asset, 'reddit')
                            }
                        }
                        patterns.append(pattern)
            
            # Sentiment analysis for high-engagement posts
            if payload.get('score', 0) > 100 or payload.get('num_comments', 0) > 50:
                sentiment = await self.analyze_sentiment(
                    payload.get('title', '') + ' ' + payload.get('selftext', '')
                )
                
                if abs(sentiment) > 0.6:  # Strong sentiment
                    pattern = {
                        'id': f'reddit_sentiment_{event["id"]}',
                        'timestamp': timestamp,
                        'type': 'sentiment_signal',
                        'source': 'reddit',
                        'signals': {
                            'sentiment_score': sentiment,
                            'engagement_score': payload.get('score', 0),
                            'discussion_level': payload.get('num_comments', 0),
                            'subreddit': payload.get('subreddit', ''),
                            'mentioned_assets': mentioned_assets
                        }
                    }
                    patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Error analyzing Reddit event: {e}")
        
        return patterns
    
    async def analyze_binance_event(self, event: Dict) -> List[Dict]:
        """Analyze Binance events for market patterns"""
        patterns = []
        
        try:
            payload = event['payload']
            timestamp = event['timestamp']
            symbol = payload.get('symbol', '')
            
            # Store price/volume data for correlation analysis
            self.event_windows['binance'].append({
                'timestamp': timestamp,
                'symbol': symbol,
                'data': payload
            })
            
            # Generate pattern based on event type
            if payload.get('type') == 'price_movement':
                pattern = {
                    'id': f'price_pattern_{symbol.replace("/", "")}_{timestamp}',
                    'timestamp': timestamp,
                    'type': 'price_movement',
                    'asset': symbol.split('/')[0],
                    'source': 'binance',
                    'signals': {
                        'price_change_percent': payload.get('change_percent', 0),
                        'volume': payload.get('volume', 0),
                        'signal_strength': payload.get('signal_strength', 0),
                        'breakout_signal': abs(payload.get('change_percent', 0)) > 10,
                        'volume_confirmation': payload.get('volume', 0) > 1000000
                    }
                }
                patterns.append(pattern)
            
            elif payload.get('type') == 'volume_spike':
                pattern = {
                    'id': f'volume_pattern_{symbol.replace("/", "")}_{timestamp}',
                    'timestamp': timestamp,
                    'type': 'volume_spike',
                    'asset': symbol.split('/')[0],
                    'source': 'binance',
                    'signals': {
                        'volume_ratio': payload.get('spike_ratio', 1),
                        'signal_strength': payload.get('signal_strength', 0),
                        'current_volume': payload.get('current_volume', 0),
                        'unusual_activity': payload.get('spike_ratio', 1) > 3
                    }
                }
                patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Error analyzing Binance event: {e}")
        
        return patterns
    
    async def analyze_news_event(self, event: Dict) -> List[Dict]:
        """Analyze news events for fundamental patterns"""
        patterns = []
        
        try:
            payload = event['payload']
            timestamp = event['timestamp']
            
            # Extract mentioned assets
            mentioned_assets = self.extract_assets_from_text(
                payload.get('title', '') + ' ' + payload.get('summary', '')
            )
            
            # High-relevance news pattern
            if payload.get('relevance_score', 0) > 0.7:
                pattern = {
                    'id': f'news_pattern_{event["id"]}',
                    'timestamp': timestamp,
                    'type': 'news_impact',
                    'source': 'news',
                    'signals': {
                        'relevance_score': payload.get('relevance_score', 0),
                        'mentioned_assets': mentioned_assets,
                        'keywords': payload.get('keywords', []),
                        'news_source': payload.get('source', ''),
                        'potential_impact': self.assess_news_impact(payload),
                        'urgency': self.calculate_news_urgency(payload)
                    }
                }
                patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error analyzing news event: {e}")
        
        return patterns
    
    async def analyze_cross_correlations(self) -> List[Dict]:
        """Analyze correlations across different data sources"""
        patterns = []
        
        try:
            # Look for assets mentioned across multiple sources recently
            recent_assets = defaultdict(list)
            
            # Collect recent mentions from all sources
            cutoff_time = datetime.utcnow() - timedelta(hours=2)
            
            for asset, mentions in self.asset_mentions.items():
                recent_mentions = [m for m in mentions 
                                 if datetime.fromisoformat(m['timestamp'].replace('Z', '')) > cutoff_time]
                if recent_mentions:
                    recent_assets[asset] = recent_mentions
            
            # Find assets with cross-source signals
            for asset, mentions in recent_assets.items():
                if len(mentions) >= 2:
                    sources = set(m['source'] for m in mentions)
                    if len(sources) > 1:  # Multi-source confirmation
                        
                        # Calculate cross-source correlation strength
                        correlation_strength = self.calculate_cross_source_correlation(asset, mentions)
                        
                        if correlation_strength > 0.6:
                            pattern = {
                                'id': f'cross_correlation_{asset}_{int(datetime.utcnow().timestamp())}',
                                'timestamp': datetime.utcnow().isoformat() + 'Z',
                                'type': 'cross_source_correlation',
                                'asset': asset,
                                'source': 'multi_source',
                                'signals': {
                                    'correlation_strength': correlation_strength,
                                    'sources': list(sources),
                                    'mention_count': len(mentions),
                                    'time_window_hours': 2,
                                    'confirmation_level': 'high' if correlation_strength > 0.8 else 'medium'
                                }
                            }
                            patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Error analyzing cross-correlations: {e}")
        
        return patterns
    
    def extract_assets_from_text(self, text: str) -> List[str]:
        """Enhanced asset extraction using comprehensive asset mappings"""
        matched_assets = set()
        text_upper = text.upper()
        
        # Check all asset categories
        for category, assets in self.assets_config.items():
            if isinstance(assets, dict):
                for asset_symbol, config in assets.items():
                    if isinstance(config, dict):
                        # Check aliases
                        aliases = config.get('aliases', [])
                        if any(alias.upper() in text_upper for alias in aliases):
                            matched_assets.add(asset_symbol)
                        
                        # Check regex patterns if available
                        patterns = config.get('patterns', [])
                        for pattern in patterns:
                            try:
                                if re.search(pattern, text_upper, re.IGNORECASE):
                                    matched_assets.add(asset_symbol)
                            except re.error:
                                continue
        
        # Check synonyms
        synonyms = self.assets_config.get('synonyms', {})
        for synonym, canonical in synonyms.items():
            if synonym.upper() in text_upper:
                matched_assets.add(canonical)
        
        return list(matched_assets)
    
    def get_asset_sector(self, asset: str) -> Optional[str]:
        """Get the sector for a given asset"""
        for sector, config in self.sectors_config.get('sectors', {}).items():
            if asset in config.get('assets', []):
                return sector
        return None
    
    def detect_sector_spike(self, sector: str) -> bool:
        """Detect if a sector is experiencing coordinated movement"""
        try:
            sector_config = self.sectors_config.get('sectors', {}).get(sector, {})
            min_assets = self.config.get('pattern_analysis', {}).get('sector_spike_min_assets', 3)
            correlation_threshold = sector_config.get('correlation_threshold', 0.7)
            
            assets_in_sector = sector_config.get('assets', [])
            spiking_assets = []
            
            for asset in assets_in_sector:
                if self.detect_mention_spike(asset):
                    spiking_assets.append(asset)
            
            return len(spiking_assets) >= min_assets
        except Exception as e:
            logger.error(f"Error detecting sector spike: {e}")
            return False
    
    def detect_mention_spike(self, asset: str) -> bool:
        """Detect mention spikes using adaptive Z-score thresholds with time decay"""
        try:
            # Get weighted counts for last 24 hours
            weighted_counts = []
            for hour in range(1, 25):
                hour_mentions = [m for m in self.asset_mentions[asset] 
                               if self.is_recent(m['timestamp'], hours=hour)]
                # Apply time decay weighting
                weighted_count = sum(
                    self.calculate_time_weight(m['timestamp']) 
                    for m in hour_mentions
                )
                weighted_counts.append(weighted_count)
            
            if len(weighted_counts) < 5: 
                return False
                
            current = weighted_counts[0]
            baseline = weighted_counts[1:]
            
            if len(baseline) == 0 or np.std(baseline) == 0:
                return False
                
            mean, std = np.mean(baseline), np.std(baseline)
            z_score_threshold = self.config.get('pattern_analysis', {}).get('mention_spike_zscore_threshold', 2.0)
            
            return bool(current > mean + z_score_threshold * std)
        except Exception as e:
            logger.error(f"Error in mention spike detection: {e}")
            return False
    
    async def analyze_sentiment(self, text: str) -> float:
        """Enhanced sentiment analysis using keyword-based approach"""
        # Use only keyword-based sentiment to avoid AI analysis patterns
        return self.keyword_sentiment(text)
    
    def keyword_sentiment(self, text: str) -> float:
        """Keyword-based sentiment analysis fallback"""
        positive_words = [
            'bullish', 'moon', 'pump', 'rise', 'surge', 'rally', 'break', 'breakthrough',
            'adoption', 'institutional', 'growth', 'gains', 'profit', 'up', 'high'
        ]
        
        negative_words = [
            'bearish', 'dump', 'crash', 'fall', 'decline', 'drop', 'loss', 'bear',
            'sell', 'down', 'low', 'panic', 'fear', 'regulation', 'ban'
        ]
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_score - negative_score) / max(total_words / 10, 1)
        return max(-1.0, min(1.0, sentiment))
    
    def calculate_sentiment_shift(self, asset: str, source: str) -> float:
        """Calculate sentiment shift for an asset"""
        try:
            if asset not in self.sentiment_history:
                return 0.0
            
            recent_sentiments = list(self.sentiment_history[asset])
            if len(recent_sentiments) < 3:
                return 0.0
            
            recent_avg = np.mean(recent_sentiments[-3:])
            older_avg = np.mean(recent_sentiments[:-3]) if len(recent_sentiments) > 3 else 0
            
            return float(recent_avg - older_avg)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment shift: {e}")
            return 0.0
    
    def assess_news_impact(self, payload: Dict) -> str:
        """Assess potential market impact of news"""
        keywords = payload.get('keywords', [])
        relevance = payload.get('relevance_score', 0)
        
        high_impact_keywords = ['regulation', 'sec', 'etf', 'institutional', 'ban', 'approval']
        medium_impact_keywords = ['adoption', 'partnership', 'launch', 'upgrade']
        
        has_high_impact = any(keyword in keywords for keyword in high_impact_keywords)
        has_medium_impact = any(keyword in keywords for keyword in medium_impact_keywords)
        
        if has_high_impact and relevance > 0.8:
            return 'high'
        elif has_medium_impact and relevance > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def calculate_news_urgency(self, payload: Dict) -> float:
        """Calculate urgency score for news"""
        relevance = payload.get('relevance_score', 0)
        recency_hours = 1  # Assume news is recent from scanner
        
        urgency = relevance * (24 / max(recency_hours, 1))  # Higher urgency for recent + relevant
        return min(urgency, 1.0)
    
    def calculate_cross_source_correlation(self, asset: str, mentions: List[Dict]) -> float:
        """Calculate correlation strength across sources"""
        try:
            sources = set(m['source'] for m in mentions)
            time_clustering = self.calculate_time_clustering([m['timestamp'] for m in mentions])
            
            # Base correlation on source diversity and time clustering
            source_diversity = len(sources) / 3.0  # Max 3 sources
            correlation = (source_diversity + time_clustering) / 2.0
            
            return min(correlation, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating cross-source correlation: {e}")
            return 0.0
    
    def calculate_time_clustering(self, timestamps: List[str]) -> float:
        """Calculate how clustered in time the mentions are"""
        try:
            times = [datetime.fromisoformat(ts.replace('Z', '')) for ts in timestamps]
            times.sort()
            
            if len(times) < 2:
                return 0.0
            
            # Calculate time spread
            time_span = (times[-1] - times[0]).total_seconds() / 3600  # hours
            
            # More clustered = higher score
            clustering = 1.0 / max(time_span, 0.1)
            return min(clustering, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating time clustering: {e}")
            return 0.0
    
    def is_recent(self, timestamp: str, hours: int = 1) -> bool:
        """Check if timestamp is within recent hours"""
        try:
            ts = datetime.fromisoformat(timestamp.replace('Z', ''))
            return datetime.utcnow() - ts < timedelta(hours=hours)
        except:
            return False
    
    def load_correlations(self, correlations: Dict):
        """Load correlation data from state"""
        self.correlations = correlations
    
    def get_correlations(self) -> Dict:
        """Get current correlation data for state saving"""
        return self.correlations
    
    def record_outcome(self, pattern_id: str, actual_outcome: Dict):
        """Record pattern outcome for later calibration"""
        # Save outcome for later calibration
        self.pattern_outcomes[pattern_id] = actual_outcome
        # Also save to persistent storage
        self.save_state()
    
    def get_pattern_confidence(self, pattern_id: str) -> float:
        """Get pattern confidence based on historical accuracy"""
        # Weight by historical accuracy
        outcomes = self.pattern_outcomes.get(pattern_id, [])
        if not outcomes: 
            return 0.5
        if isinstance(outcomes, dict):
            outcomes = [outcomes]
        return float(np.mean([o.get("hit", 0.5) for o in outcomes]))  # 0.0–1.0
    
    def save_state(self):
        """Save correlations, mentions, outcomes to PostgreSQL or SQLite fallback"""
        try:
            # Try PostgreSQL first if models are available
            if db is not None and PatternOutcome is not None:
                self._save_state_postgresql()
            else:
                self._save_state_sqlite()
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            # Fallback to SQLite if PostgreSQL fails
            try:
                self._save_state_sqlite()
            except Exception as e2:
                logger.error(f"Error saving state to SQLite fallback: {e2}")
    
    def _save_state_postgresql(self):
        """Save to PostgreSQL using SQLAlchemy"""
        with current_app.app_context():
            # Save pattern outcomes
            for pattern_id, outcome in self.pattern_outcomes.items():
                pattern_outcome = PatternOutcome.query.filter_by(pattern_id=pattern_id).first()
                if pattern_outcome:
                    pattern_outcome.outcome = outcome
                    pattern_outcome.timestamp = datetime.utcnow()
                else:
                    pattern_outcome = PatternOutcome(
                        pattern_id=pattern_id,
                        outcome=outcome
                    )
                    db.session.add(pattern_outcome)
            
            # Save asset mentions (recent ones only)
            AssetMention.query.delete()
            for asset, mentions in self.asset_mentions.items():
                asset_mention = AssetMention(
                    asset=asset,
                    mentions=list(mentions)
                )
                db.session.add(asset_mention)
            
            # Save correlations
            for key, value in self.correlations.items():
                correlation = Correlation.query.filter_by(key=key).first()
                if correlation:
                    correlation.value = value
                    correlation.timestamp = datetime.utcnow()
                else:
                    correlation = Correlation(
                        key=key,
                        value=value
                    )
                    db.session.add(correlation)
            
            db.session.commit()
    
    def _save_state_sqlite(self):
        """Fallback to SQLite"""
        conn = sqlite3.connect("patterns.db")
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_outcomes (
                pattern_id TEXT PRIMARY KEY,
                outcome TEXT,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_mentions (
                asset TEXT,
                mentions TEXT,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS correlations (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp TEXT
            )
        ''')
        
        # Save pattern outcomes
        for pattern_id, outcome in self.pattern_outcomes.items():
            cursor.execute(
                "INSERT OR REPLACE INTO pattern_outcomes VALUES (?, ?, ?)",
                (pattern_id, json.dumps(outcome), datetime.utcnow().isoformat())
            )
        
        # Save asset mentions (recent ones only)
        cursor.execute("DELETE FROM asset_mentions")
        for asset, mentions in self.asset_mentions.items():
            mentions_data = json.dumps(list(mentions))
            cursor.execute(
                "INSERT INTO asset_mentions VALUES (?, ?, ?)",
                (asset, mentions_data, datetime.utcnow().isoformat())
            )
        
        # Save correlations
        for key, value in self.correlations.items():
            cursor.execute(
                "INSERT OR REPLACE INTO correlations VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.utcnow().isoformat())
            )
        
        conn.commit()
        conn.close()
    
    def save_patterns_to_db(self, patterns):
        """Save detected patterns to the main patterns table"""
        try:
            conn = sqlite3.connect("patterns.db")
            cursor = conn.cursor()
            
            # Ensure patterns table exists (should already be created by orchestrator)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    asset TEXT NOT NULL,
                    type TEXT,
                    confidence REAL,
                    signals TEXT,
                    price REAL,
                    volume REAL,
                    pattern_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Save each pattern
            saved_count = 0
            for pattern in patterns:
                try:
                    timestamp = pattern.get('timestamp', datetime.utcnow().timestamp())
                    asset = pattern.get('asset', 'UNKNOWN')
                    pattern_type = pattern.get('type', 'GENERAL')
                    confidence = pattern.get('confidence', 0.5)
                    signals = json.dumps(pattern.get('signals', {}))
                    price = pattern.get('price', 0.0)
                    volume = pattern.get('volume', 0.0)
                    pattern_data = json.dumps(pattern)
                    
                    cursor.execute('''
                        INSERT INTO patterns 
                        (timestamp, asset, type, confidence, signals, price, volume, pattern_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (timestamp, asset, pattern_type, confidence, signals, price, volume, pattern_data))
                    
                    saved_count += 1
                    
                except Exception as e:
                    logger.error(f"Error saving individual pattern: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            if saved_count > 0:
                logger.info(f"Saved {saved_count} patterns to database")
                
        except Exception as e:
            logger.error(f"Error saving patterns to database: {e}")
    
    def load_state(self):
        """Load everything from PostgreSQL or SQLite fallback"""
        try:
            # Try PostgreSQL first if models are available
            if db is not None and PatternOutcome is not None:
                self._load_state_postgresql()
            else:
                self._load_state_sqlite()
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            # Fallback to SQLite if PostgreSQL fails
            try:
                self._load_state_sqlite()
            except Exception as e2:
                logger.error(f"Error loading state from SQLite fallback: {e2}")
    
    def _load_state_postgresql(self):
        """Load from PostgreSQL using SQLAlchemy"""
        try:
            with current_app.app_context():
                # Load pattern outcomes
                pattern_outcomes = PatternOutcome.query.all()
                for outcome in pattern_outcomes:
                    self.pattern_outcomes[outcome.pattern_id] = outcome.outcome
                
                # Load asset mentions
                asset_mentions = AssetMention.query.all()
                for mention in asset_mentions:
                    self.asset_mentions[mention.asset] = deque(mention.mentions, maxlen=50)
                
                # Load correlations
                correlations = Correlation.query.all()
                for corr in correlations:
                    self.correlations[corr.key] = corr.value
                
                logger.info("State loaded successfully from PostgreSQL database")
        except Exception as e:
            logger.warning(f"Error loading from PostgreSQL: {e}, falling back to SQLite")
            self._load_state_sqlite()
    
    def _load_state_sqlite(self):
        """Load from SQLite fallback"""
        if not os.path.exists("patterns.db"):
            return
            
        conn = sqlite3.connect("patterns.db")
        cursor = conn.cursor()
        
        # Load pattern outcomes
        try:
            cursor.execute("SELECT pattern_id, outcome FROM pattern_outcomes")
            for pattern_id, outcome_str in cursor.fetchall():
                self.pattern_outcomes[pattern_id] = json.loads(outcome_str)
        except:
            pass
        
        # Load asset mentions
        try:
            cursor.execute("SELECT asset, mentions FROM asset_mentions")
            for asset, mentions_str in cursor.fetchall():
                mentions_list = json.loads(mentions_str)
                self.asset_mentions[asset] = deque(mentions_list, maxlen=50)
        except:
            pass
        
        # Load correlations
        try:
            cursor.execute("SELECT key, value FROM correlations")
            for key, value_str in cursor.fetchall():
                self.correlations[key] = json.loads(value_str)
        except:
            pass
        
        conn.close()
        logger.info("State loaded successfully from SQLite database")
    
    def load_config(self) -> Dict:
        """Load main configuration file"""
        try:
            with open("config.json") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")
            return {"pattern_analysis": {"time_decay_half_life_hours": 6}}
    
    def load_assets_config(self) -> Dict:
        """Load enhanced assets configuration"""
        try:
            with open("assets-config.json") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load assets-config.json: {e}")
            return {}
    
    def load_sectors_config(self) -> Dict:
        """Load sectors configuration"""
        try:
            with open("sectors-config.json") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load sectors-config.json: {e}")
            return {}
    
    def build_legacy_asset_map(self) -> Dict:
        """Build legacy asset map for backward compatibility"""
        legacy_map = {}
        for category, assets in self.assets_config.items():
            if isinstance(assets, dict):
                for asset, config in assets.items():
                    if isinstance(config, dict) and 'aliases' in config:
                        legacy_map[asset] = config['aliases']
        return legacy_map
    
    def calculate_time_weight(self, timestamp: str, half_life_hours: Optional[float] = None) -> float:
        """Calculate time-decay weight for events"""
        try:
            if half_life_hours is None:
                half_life_hours = self.config.get('pattern_analysis', {}).get('time_decay_half_life_hours', 6)
            
            half_life_hours = float(half_life_hours or 6)
            event_time = datetime.fromisoformat(timestamp.replace('Z', ''))
            age_hours = (datetime.utcnow() - event_time).total_seconds() / 3600
            
            return math.exp(-age_hours * math.log(2) / half_life_hours)
        except Exception as e:
            logger.warning(f"Error calculating time weight: {e}")
            return 1.0
