"""
Event Ontology & Normalizer
Converts raw events from all sources into standardized JSON schema
"""

import json
import sqlite3
import os
import psycopg2
from datetime import datetime
from typing import Dict, List, Optional
import re
import logging

logger = logging.getLogger(__name__)

class EventNormalizer:
    def __init__(self, db_path='patterns.db'):
        # Use PostgreSQL if available, fallback to SQLite
        self.database_url = os.getenv('DATABASE_URL')
        if self.database_url:
            self.conn = psycopg2.connect(self.database_url)
        else:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_tables()
        self.event_types = {
            'POLICY_SIGNAL': ['policy', 'rbi', 'fed', 'rate', 'monetary', 'fiscal'],
            'LIQUIDITY_SHIFT': ['liquidity', 'funding', 'repo', 'treasury', 'm2', 'money supply'],
            'FLOW_SPIKE': ['whale', 'block trade', 'institutional', 'volume spike', 'options flow'],
            'MICROSTRUCTURE': ['bid ask', 'spread', 'volatility', 'level break', 'support', 'resistance'],
            'SENTIMENT_SWING': ['sentiment', 'fear', 'greed', 'bullish', 'bearish', 'optimistic'],
            'EARNINGS_SURPRISE': ['earnings', 'revenue', 'profit', 'guidance', 'results'],
            'TECHNICAL_BREAK': ['breakout', 'breakdown', 'trend', 'pattern', 'moving average']
        }

    def init_tables(self):
        """Initialize database tables for events and entity linking"""
        if self.database_url:
            # PostgreSQL
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events_normalized (
                    id TEXT PRIMARY KEY,
                    ts TIMESTAMP NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT,
                    entities JSONB,
                    assets JSONB,
                    features JSONB,
                    region TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_assets (
                    event_id TEXT,
                    asset_symbol TEXT,
                    relevance_score REAL,
                    FOREIGN KEY(event_id) REFERENCES events_normalized(id)
                )
            ''')
            self.conn.commit()
        else:
            # SQLite
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS events_normalized (
                    id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT,
                    entities TEXT,
                    assets TEXT,
                    features TEXT,
                    region TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS event_assets (
                    event_id TEXT,
                    asset_symbol TEXT,
                    relevance_score REAL,
                    FOREIGN KEY(event_id) REFERENCES events_normalized(id)
                )
            ''')
            self.conn.commit()

    def normalize(self, raw_event: Dict) -> Dict:
        """Convert raw event into ontology-compliant format"""
        try:
            # Defensive check: ensure raw_event is a dictionary
            if not isinstance(raw_event, dict):
                logger.error(f"Expected dict for raw_event but got {type(raw_event)}: {raw_event}")
                return {}
            
            # Generate unique ID
            event_id = raw_event.get('id', f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(raw_event)) % 1000}")
            
            # Extract and normalize fields
            event = {
                "id": event_id,
                "ts": self._normalize_timestamp(raw_event.get('timestamp')),
                "type": self._classify_event_type(raw_event),
                "source": raw_event.get('source', 'unknown'),
                "entities": self._extract_entities(raw_event),
                "assets": self._extract_assets(raw_event),
                "features": self._extract_features(raw_event),
                "region": raw_event.get('region', 'GLOBAL'),
                "confidence": float(raw_event.get('confidence', 0.5))
            }
            
            # Store in database
            self._store_event(event)
            logger.info(f"Normalized event: {event_id} [{event['type']}]")
            return event
            
        except Exception as e:
            logger.error(f"Error normalizing event: {e}")
            return {}

    def _normalize_timestamp(self, timestamp) -> str:
        """Ensure timestamp is in ISO format"""
        if not timestamp:
            return datetime.now().isoformat()
        if isinstance(timestamp, str):
            return timestamp
        return timestamp.isoformat()

    def _classify_event_type(self, event: Dict) -> str:
        """Classify event type based on content"""
        text = (event.get('title', '') + ' ' + event.get('summary', '') + ' ' + 
                ' '.join(event.get('keywords', []))).lower()
        
        for event_type, keywords in self.event_types.items():
            if any(keyword in text for keyword in keywords):
                return event_type
        
        return 'GENERAL_EVENT'

    def _extract_entities(self, event: Dict) -> List[str]:
        """Extract entities (RBI, companies, etc.)"""
        entities = event.get('entities', [])
        
        # Add pattern-based extraction
        text = event.get('title', '') + ' ' + event.get('summary', '')
        
        # Common Indian financial entities
        entity_patterns = {
            'RBI': r'\b(rbi|reserve bank|central bank)\b',
            'SEBI': r'\b(sebi|securities board)\b',
            'NSE': r'\b(nse|national stock)\b',
            'BSE': r'\b(bse|bombay stock)\b'
        }
        
        for entity, pattern in entity_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                entities.append(entity)
        
        return list(set(entities))

    def _extract_assets(self, event: Dict) -> List[str]:
        """Extract relevant asset symbols"""
        assets = event.get('assets', [])
        
        # Load from existing assets config
        try:
            with open('assets-config.json', 'r') as f:
                asset_config = json.load(f)
            
            text = event.get('title', '') + ' ' + event.get('summary', '')
            
            # Check for mentions of tracked assets
            for category in asset_config.values():
                for symbol, details in category.items():
                    aliases = details.get('aliases', [symbol])
                    patterns = details.get('patterns', [f"\\b{symbol}\\b"])
                    
                    for pattern in patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            assets.append(symbol)
                            
        except FileNotFoundError:
            logger.warning("assets-config.json not found")
            
        return list(set(assets))

    def _extract_features(self, event: Dict) -> Dict:
        """Extract quantitative features from event"""
        features = event.get('features', {})
        
        text = event.get('title', '') + ' ' + event.get('summary', '')
        
        # Sentiment polarity
        positive_words = ['bullish', 'growth', 'positive', 'rally', 'surge', 'optimistic']
        negative_words = ['bearish', 'decline', 'negative', 'fall', 'crash', 'pessimistic']
        
        pos_score = sum(1 for word in positive_words if word in text.lower())
        neg_score = sum(1 for word in negative_words if word in text.lower())
        
        if pos_score + neg_score > 0:
            features['sentiment_polarity'] = (pos_score - neg_score) / (pos_score + neg_score)
        
        # Extract numbers (could be prices, percentages, etc.)
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            features['numbers_mentioned'] = [float(n) for n in numbers[:5]]  # Limit to 5
            
        # Urgency indicators
        urgent_words = ['urgent', 'breaking', 'alert', 'immediate', 'emergency']
        features['urgency_score'] = sum(1 for word in urgent_words if word in text.lower()) / len(urgent_words)
        
        return features

    def _store_event(self, event: Dict):
        """Store event in database with asset linking"""
        try:
            if self.database_url:
                # PostgreSQL
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO events_normalized (id, ts, type, source, entities, assets, features, region, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                    ts = EXCLUDED.ts,
                    type = EXCLUDED.type,
                    confidence = EXCLUDED.confidence
                ''', (
                    event['id'], event['ts'], event['type'], event['source'],
                    json.dumps(event['entities']), json.dumps(event['assets']), 
                    json.dumps(event['features']), event['region'], event['confidence']
                ))
                
                # Store asset relationships
                for asset in event['assets']:
                    cursor.execute('''
                        INSERT INTO event_assets (event_id, asset_symbol, relevance_score)
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING
                    ''', (event['id'], asset, 0.8))
            else:
                # SQLite
                self.conn.execute('''
                    INSERT OR REPLACE INTO events_normalized VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event['id'], event['ts'], event['type'], event['source'],
                    json.dumps(event['entities']), json.dumps(event['assets']), 
                    json.dumps(event['features']), event['region'], event['confidence']
                ))
                
                # Store asset relationships
                for asset in event['assets']:
                    self.conn.execute('''
                        INSERT OR REPLACE INTO event_assets VALUES (?, ?, ?)
                    ''', (event['id'], asset, 0.8))
                    
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing event: {e}")

    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Dict]:
        """Retrieve events by type"""
        try:
            if self.database_url:
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT * FROM events_normalized WHERE type = %s ORDER BY ts DESC LIMIT %s
                ''', (event_type, limit))
            else:
                cursor = self.conn.execute('''
                    SELECT * FROM events_normalized WHERE type = ? ORDER BY ts DESC LIMIT ?
                ''', (event_type, limit))
            
            return [dict(zip([col[0] for col in cursor.description], row)) 
                    for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting events by type: {e}")
            return []

    def get_events_for_asset(self, asset_symbol: str, limit: int = 50) -> List[Dict]:
        """Get events affecting specific asset"""
        try:
            if self.database_url:
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT e.* FROM events_normalized e
                    JOIN event_assets ea ON e.id = ea.event_id
                    WHERE ea.asset_symbol = %s
                    ORDER BY e.ts DESC LIMIT %s
                ''', (asset_symbol, limit))
            else:
                cursor = self.conn.execute('''
                    SELECT e.* FROM events_normalized e
                    JOIN event_assets ea ON e.id = ea.event_id
                    WHERE ea.asset_symbol = ?
                    ORDER BY e.ts DESC LIMIT ?
                ''', (asset_symbol, limit))
            
            return [dict(zip([col[0] for col in cursor.description], row)) 
                    for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting events for asset: {e}")
            return []