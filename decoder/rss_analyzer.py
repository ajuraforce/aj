"""
RSS Analyzer Module
Enhanced RSS feed analysis with AI-powered sentiment, asset detection, and relevance scoring
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
import asyncio

logger = logging.getLogger(__name__)

class RSSAnalyzer:
    """Enhanced RSS analyzer with AI-powered analysis"""
    
    def __init__(self, db_path='patterns.db'):
        self.db_path = db_path
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        
        # Initialize OpenAI client if available
        if self.openai_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.openai_key)
                logger.info("OpenAI client initialized for RSS analysis")
            except ImportError:
                logger.warning("OpenAI library not available")
        else:
            logger.warning("OpenAI API key not configured")
        
        # Initialize database
        self.init_db()
    
    def init_db(self):
        """Initialize RSS articles database table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create RSS articles table with enhanced fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rss_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                link TEXT,
                description TEXT,
                pub_date TEXT,
                source TEXT,
                category TEXT,
                fetched_at TEXT,
                sentiment TEXT,
                confidence REAL,
                mentioned_assets TEXT,
                relevance_score REAL,
                key_themes TEXT,
                analyzed_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("RSS articles database table initialized")
    
    def analyze_article(self, article_id: int, title: str, description: str) -> Optional[Dict]:
        """Analyze a single article for sentiment, assets, and relevance"""
        try:
            if not self.client:
                # Fallback analysis without OpenAI
                return self.fallback_analysis(title, description)
            
            prompt = f"""Analyze this news article for trading insights:
Title: {title}
Description: {description}

Provide JSON with:
- sentiment: positive/negative/neutral
- confidence: 0-1
- mentioned_assets: list of stocks/crypto (e.g., BTC, RELIANCE.NS, ETH, NIFTY)
- relevance_score: 0-1 for trading impact
- key_themes: list of themes (e.g., oil_crisis, tech_disruption, crypto_regulation)

Only include assets that are explicitly mentioned or clearly relevant."""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                
                # Validate and clean the response
                analysis = self.validate_analysis(analysis)
                
                # Store analysis in database
                self.store_analysis(article_id, analysis)
                
                logger.debug(f"Analyzed article {article_id} with sentiment: {analysis.get('sentiment')}")
                return analysis
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response: {e}")
                return self.fallback_analysis(title, description)
                
        except Exception as e:
            logger.error(f"Analysis error for article {article_id}: {e}")
            return self.fallback_analysis(title, description)
    
    def fallback_analysis(self, title: str, description: str) -> Dict:
        """Fallback analysis when OpenAI is not available"""
        text = (title + ' ' + description).lower()
        
        # Simple sentiment analysis
        positive_words = ['surge', 'rally', 'bullish', 'pump', 'gain', 'rise', 'breakthrough', 'adoption', 'approval']
        negative_words = ['crash', 'dump', 'bearish', 'fall', 'decline', 'drop', 'ban', 'regulation', 'concern']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = 'positive'
        elif negative_count > positive_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Simple asset detection
        mentioned_assets = []
        crypto_assets = ['btc', 'bitcoin', 'eth', 'ethereum', 'crypto', 'blockchain']
        indian_assets = ['nifty', 'sensex', 'reliance', 'tcs', 'hdfc', 'icici']
        
        for asset in crypto_assets:
            if asset in text:
                if asset in ['btc', 'bitcoin']:
                    mentioned_assets.append('BTC')
                elif asset in ['eth', 'ethereum']:
                    mentioned_assets.append('ETH')
                elif asset in ['crypto', 'blockchain']:
                    mentioned_assets.append('CRYPTO')
        
        for asset in indian_assets:
            if asset in text:
                if asset == 'nifty':
                    mentioned_assets.append('NIFTY50')
                elif asset == 'sensex':
                    mentioned_assets.append('SENSEX')
                elif asset == 'reliance':
                    mentioned_assets.append('RELIANCE.NS')
        
        # Simple relevance scoring
        relevance_score = min(0.3 + (positive_count + negative_count) * 0.1 + len(mentioned_assets) * 0.1, 1.0)
        
        return {
            'sentiment': sentiment,
            'confidence': 0.6,  # Lower confidence for fallback
            'mentioned_assets': list(set(mentioned_assets)),
            'relevance_score': relevance_score,
            'key_themes': ['market_movement'] if mentioned_assets else ['general_news']
        }
    
    def validate_analysis(self, analysis: Dict) -> Dict:
        """Validate and clean analysis results"""
        # Ensure required fields exist
        analysis['sentiment'] = analysis.get('sentiment', 'neutral')
        analysis['confidence'] = max(0, min(1, analysis.get('confidence', 0.5)))
        analysis['mentioned_assets'] = analysis.get('mentioned_assets', [])
        analysis['relevance_score'] = max(0, min(1, analysis.get('relevance_score', 0.0)))
        analysis['key_themes'] = analysis.get('key_themes', [])
        
        # Validate sentiment
        if analysis['sentiment'] not in ['positive', 'negative', 'neutral']:
            analysis['sentiment'] = 'neutral'
        
        # Ensure mentioned_assets is a list
        if not isinstance(analysis['mentioned_assets'], list):
            analysis['mentioned_assets'] = []
        
        # Ensure key_themes is a list
        if not isinstance(analysis['key_themes'], list):
            analysis['key_themes'] = []
        
        return analysis
    
    def store_analysis(self, article_id: int, analysis: Dict):
        """Store analysis results in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE rss_articles
                SET sentiment = ?, confidence = ?, mentioned_assets = ?, 
                    relevance_score = ?, key_themes = ?, analyzed_at = ?
                WHERE id = ?
            ''', (
                analysis['sentiment'],
                analysis['confidence'],
                json.dumps(analysis['mentioned_assets']),
                analysis['relevance_score'],
                json.dumps(analysis['key_themes']),
                datetime.now().isoformat(),
                article_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing analysis for article {article_id}: {e}")
    
    def process_batch(self, batch_size: int = 50) -> int:
        """Process a batch of unanalyzed articles"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get unanalyzed articles
            cursor.execute('''
                SELECT id, title, description 
                FROM rss_articles 
                WHERE sentiment IS NULL 
                LIMIT ?
            ''', (batch_size,))
            
            articles = cursor.fetchall()
            conn.close()
            
            analyzed_count = 0
            for article_id, title, description in articles:
                if self.analyze_article(article_id, title or '', description or ''):
                    analyzed_count += 1
            
            logger.info(f"Analyzed {analyzed_count} articles in batch")
            return analyzed_count
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return 0
    
    def get_analyzed_articles(self, category: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get analyzed articles from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if category:
                cursor.execute('''
                    SELECT * FROM rss_articles 
                    WHERE category = ? AND sentiment IS NOT NULL
                    ORDER BY pub_date DESC 
                    LIMIT ?
                ''', (category, limit))
            else:
                cursor.execute('''
                    SELECT * FROM rss_articles 
                    WHERE sentiment IS NOT NULL
                    ORDER BY pub_date DESC 
                    LIMIT ?
                ''', (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            articles = []
            
            for row in cursor.fetchall():
                article = dict(zip(columns, row))
                
                # Parse JSON fields
                if article['mentioned_assets']:
                    try:
                        article['mentioned_assets'] = json.loads(article['mentioned_assets'])
                    except json.JSONDecodeError:
                        article['mentioned_assets'] = []
                
                if article['key_themes']:
                    try:
                        article['key_themes'] = json.loads(article['key_themes'])
                    except json.JSONDecodeError:
                        article['key_themes'] = []
                
                articles.append(article)
            
            conn.close()
            return articles
            
        except Exception as e:
            logger.error(f"Error getting analyzed articles: {e}")
            return []
    
    def get_status(self) -> Dict:
        """Get analyzer status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total articles
            cursor.execute('SELECT COUNT(*) FROM rss_articles')
            total_articles = cursor.fetchone()[0]
            
            # Analyzed articles
            cursor.execute('SELECT COUNT(*) FROM rss_articles WHERE sentiment IS NOT NULL')
            analyzed_articles = cursor.fetchone()[0]
            
            # Pending analysis
            cursor.execute('SELECT COUNT(*) FROM rss_articles WHERE sentiment IS NULL')
            pending_articles = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_articles': total_articles,
                'analyzed_articles': analyzed_articles,
                'pending_articles': pending_articles,
                'openai_available': self.client is not None,
                'last_batch_processed': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                'total_articles': 0,
                'analyzed_articles': 0,
                'pending_articles': 0,
                'openai_available': False,
                'error': str(e)
            }