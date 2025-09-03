"""
News Scanner Module
Implements RSS and news feed monitoring for market-moving events
Enhanced with story clustering and narrative intelligence
"""

import os
import feedparser
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import aiohttp
import json
import time
import re
import hashlib

logger = logging.getLogger(__name__)

class NewsScanner:
    """News scanner for RSS feeds and financial news"""
    
    def __init__(self):
        self.last_offset = None
        self.processed_articles = set()
        self.session = None
        self.recent_events = []  # Store recent events for API access
        
        # Enhanced deduplication tracking
        self.seen_urls = set()
        self.seen_titles = set()
        self.article_summaries = {}  # Store AI-generated summaries
        self.hourly_counts = {}  # Track articles per hour for trends
        
        # Initialize OpenAI client for AI summaries and rationale
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.ai_client = None
        if self.openai_key:
            try:
                import openai
                self.ai_client = openai.OpenAI(api_key=self.openai_key)
                logger.info("OpenAI client initialized for news intelligence")
            except ImportError:
                logger.warning("OpenAI library not available for news intelligence")
        
        # Initialize story clustering engine
        try:
            from decoder.story_cluster_engine import StoryClusterEngine
            self.story_engine = StoryClusterEngine()
            logger.info("Story clustering engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize story clustering engine: {e}")
            self.story_engine = None
        
        self.news_sources = {
            # Global Finance & Markets
            "global_finance": [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
                'https://www.investing.com/rss/news.rss',
                # Google News for global markets
                'https://news.google.com/rss/search?q=global+stock+market',
                'https://news.google.com/rss/search?q=NYSE+NASDAQ+trading',
                'https://news.google.com/rss/search?q=US+Federal+Reserve'
            ],
            
            # India-Specific Markets
            "india_markets": [
                'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
                'https://www.moneycontrol.com/rss/latestnews.xml',
                'https://www.livemint.com/rss/markets',
                'https://www.moneycontrol.com/rss/MCtopnews.xml',
                # Google News for India markets
                'https://news.google.com/rss/search?q=India+stock+market',
                'https://news.google.com/rss/search?q=NIFTY50+BSE+NSE',
                'https://news.google.com/rss/search?q=Indian+economy+RBI'
            ],
            
            # Crypto & Blockchain
            "crypto": [
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://cointelegraph.com/rss',
                'https://decrypt.co/feed',
                'https://cryptoslate.com/feed/',
                'https://cryptopotato.com/feed/',
                'https://beincrypto.com/feed/',
                'https://cryptonews.com/news/feed/',
                'https://cryptobriefing.com/feed/',
                'https://u.today/rss',
                # Additional crypto feeds
                'https://www.newsbtc.com/feed/'
            ],
            
            # Geopolitics & Macro
            "geopolitics": [
                'https://www.aljazeera.com/xml/rss/all.xml',
                'http://feeds.bbci.co.uk/news/world/rss.xml',
                'https://www.thehindu.com/news/international/feeder/default.rss',
                'https://indianexpress.com/section/world/feed/',
                'https://www.theguardian.com/world/rss',
                'https://rss.dw.com/rdf/rss-en-world',
                # Google News for geopolitics
                'https://news.google.com/rss/search?q=geopolitics+trade+war',
                'https://news.google.com/rss/search?q=central+bank+policy'
            ],
            
            # Technology
            "technology": [
                'http://feeds.feedburner.com/TechCrunch/',
                'https://www.wired.com/feed/rss',
                'https://www.theverge.com/rss/index.xml'
            ],
            
            # Sentiment & Social Indicators
            "sentiment": [
                'https://news.google.com/rss/search?q=stock+market+india',
                'https://news.google.com/rss/search?q=crypto+bitcoin+sentiment',
                'https://news.google.com/rss/search?q=market+volatility+fear',
                'https://www.reddit.com/r/IndianStockMarket/.rss',
                'https://www.reddit.com/r/CryptoCurrency/.rss'
            ]
        }
    
    async def scan(self) -> List[Dict]:
        """Scan news sources for market-relevant content"""
        try:
            events = []
            stored_articles = []
            
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30)
                )
            
            # Scan all news sources by category
            tasks = []
            for category, urls in self.news_sources.items():
                for url in urls:
                    tasks.append(self.scan_rss_feed(url, category))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"News source error: {result}")
                elif result and isinstance(result, list):
                    events.extend(result)
            
            # Filter out already processed articles
            new_events = [e for e in events if e['id'] not in self.processed_articles]
            
            # Update processed set
            for event in new_events:
                self.processed_articles.add(event['id'])
            
            # Limit processed articles memory
            if len(self.processed_articles) > 1000:
                self.processed_articles = set(list(self.processed_articles)[-500:])
            
            # Store recent events for API access (keep last 50)
            self.recent_events.extend(new_events)
            if len(self.recent_events) > 50:
                self.recent_events = self.recent_events[-50:]
            
            # Store articles in database and create narratives if story engine is available
            if self.story_engine and new_events:
                await self.process_articles_for_narratives(new_events)
            
            logger.info(f"Collected {len(new_events)} new news events")
            return new_events
            
        except Exception as e:
            logger.error(f"Error scanning news: {e}")
            return []
    
    async def scan_rss_feed(self, feed_url: str, category: str = "general") -> List[Dict]:
        """Scan a single RSS feed"""
        try:
            if not self.session:
                return []
            
            async with self.session.get(feed_url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {feed_url}: {response.status}")
                    return []
                
                content = await response.text()
                feed = feedparser.parse(content)
                
                events = []
                for entry in feed.entries[:10]:  # Limit to 10 most recent
                    try:
                        # Parse publication date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_time = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_time = datetime(*entry.updated_parsed[:6])
                        else:
                            pub_time = datetime.utcnow()
                        
                        # Only process recent articles (last 24 hours)
                        if datetime.utcnow() - pub_time > timedelta(days=1):
                            continue
                        
                        # Extract and analyze content
                        title = entry.get('title', '')
                        summary = entry.get('summary', entry.get('description', ''))
                        
                        # Check for duplicates using enhanced deduplication
                        article_url = entry.get('link', '')
                        normalized_title = self.normalize_whitespace(title)
                        
                        if self.is_duplicate(article_url, normalized_title):
                            continue
                        
                        # Calculate relevance score
                        relevance_score = self.calculate_relevance(title, summary)
                        
                        if relevance_score > 0.3:  # Only high-relevance articles
                            # Mark as seen to prevent future duplicates
                            self.seen_urls.add(article_url)
                            self.seen_titles.add(normalized_title.lower())
                            
                            # Generate AI summary and rationale if available
                            enhanced_summary = await self.generate_enhanced_summary(title, summary)
                            rationale = await self.generate_rationale(title, enhanced_summary)
                            
                            # Track hourly counts for trend analysis
                            self.update_hourly_counts(pub_time)
                            
                            article_id = f"news_{abs(hash(entry.get('id', entry.get('link', ''))))}"
                            event = {
                                "source": "news",
                                "id": article_id,
                                "timestamp": pub_time.isoformat() + "Z",
                                "payload": {
                                    "type": "news_article",
                                    "title": title,
                                    "summary": summary[:500],
                                    "url": entry.get('link', ''),
                                    "source": feed.feed.get('title', feed_url),
                                    "author": entry.get('author', ''),
                                    "category": category,
                                    "relevance_score": relevance_score,
                                    "keywords": self.extract_keywords(title + ' ' + summary),
                                    "published": pub_time
                                }
                            }
                            
                            # Anomaly detection for high-impact news
                            if relevance_score > 0.8:  # High impact threshold
                                event['payload']['anomaly'] = True
                                event['payload']['impact_score'] = relevance_score
                                event['payload']['urgency'] = 'high'
                            else:
                                event['payload']['urgency'] = 'medium' if relevance_score > 0.6 else 'low'
                            
                            events.append(event)
                    
                    except Exception as e:
                        logger.warning(f"Error processing news entry: {e}")
                        continue
                
                return events
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {feed_url}")
            return []
        except Exception as e:
            logger.error(f"Error scanning RSS feed {feed_url}: {e}")
            return []
    
    def interactive_screen(self, filters: Dict) -> List[Dict]:
        """
        Screen news events based on filters (e.g., keywords, relevance_score).
        """
        events = self.recent_events  # Use stored recent events
        screened = []
        for event in events:
            if 'min_relevance' in filters and event['payload'].get('relevance_score', 0) < filters['min_relevance']:
                continue
            if 'keywords' in filters and not any(kw in event['payload'].get('keywords', []) for kw in filters['keywords']):
                continue
            screened.append(event)
        return screened
    
    def calculate_relevance(self, title: str, summary: str) -> float:
        """Calculate relevance score for news content"""
        try:
            text = (title + ' ' + summary).lower()
            score = 0.0
            
            # Cryptocurrency keywords (high weight)
            crypto_keywords = [
                'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
                'blockchain', 'defi', 'binance', 'coinbase', 'trading', 'altcoin'
            ]
            for keyword in crypto_keywords:
                if keyword in text:
                    score += 0.3
            
            # Market movement keywords (medium weight)
            movement_keywords = [
                'surge', 'pump', 'dump', 'rally', 'crash', 'breakout',
                'bullish', 'bearish', 'volatility', 'volume'
            ]
            for keyword in movement_keywords:
                if keyword in text:
                    score += 0.2
            
            # Regulatory/institutional keywords (high weight)
            institutional_keywords = [
                'regulation', 'sec', 'etf', 'institutional', 'adoption',
                'ban', 'approval', 'government', 'central bank'
            ]
            for keyword in institutional_keywords:
                if keyword in text:
                    score += 0.4
            
            # Technical keywords (low weight)
            technical_keywords = [
                'support', 'resistance', 'technical analysis', 'chart',
                'fibonacci', 'moving average', 'rsi', 'macd'
            ]
            for keyword in technical_keywords:
                if keyword in text:
                    score += 0.1
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract key financial and crypto terms from text"""
        try:
            text = text.lower()
            
            # Define important keyword patterns
            patterns = [
                r'\b(bitcoin|btc)\b', r'\b(ethereum|eth)\b', r'\b(binance|bnb)\b',
                r'\b(cardano|ada)\b', r'\b(solana|sol)\b', r'\b(polkadot|dot)\b',
                r'\b(chainlink|link)\b', r'\b(polygon|matic)\b', r'\b(avalanche|avax)\b',
                r'\b(ripple|xrp)\b', r'\b(dogecoin|doge)\b', r'\b(shiba|shib)\b',
                r'\b(defi|dao|nft)\b', r'\b(staking|yield|farming)\b',
                r'\b(regulation|sec|etf)\b', r'\b(institutional|adoption)\b'
            ]
            
            keywords = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                keywords.extend(matches)
            
            return list(set(keywords))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def process_articles_for_narratives(self, events: List[Dict]):
        """Process news events for database storage and narrative creation"""
        try:
            if not self.story_engine:
                return
            
            # Convert events to article format for database storage
            for event in events:
                payload = event['payload']
                
                # Extract asset mentions from keywords
                mentioned_assets = []
                keywords = payload.get('keywords', [])
                for keyword in keywords:
                    # Map keywords to assets
                    if 'btc' in keyword.lower() or 'bitcoin' in keyword.lower():
                        mentioned_assets.append('BTC')
                    elif 'eth' in keyword.lower() or 'ethereum' in keyword.lower():
                        mentioned_assets.append('ETH')
                    # Add more asset mappings as needed
                
                # Calculate sentiment score (simple version)
                sentiment_score = self.calculate_sentiment(payload['title'], payload['summary'])
                
                # Prepare article data for database storage
                article_data = {
                    'id': event['id'],
                    'title': payload['title'],
                    'summary': payload['summary'],
                    'link': payload['url'],
                    'source': payload['source'],
                    'published': payload['published'],
                    'content': payload['summary'],  # Using summary as content for now
                    'sentiment_score': sentiment_score,
                    'relevance_score': payload['relevance_score'],
                    'mentioned_assets': mentioned_assets,
                    'sector': payload.get('category', ''),
                    'urgency': payload.get('urgency', 'low')
                }
                
                # Store article in database
                self.story_engine.store_news_article(article_data)
            
            # Create narratives from recent articles (async operation)
            narratives_created = await self.story_engine.create_narratives_from_recent_articles(hours_back=6)
            if narratives_created > 0:
                logger.info(f"Created {narratives_created} new narratives from recent articles")
            
        except Exception as e:
            logger.error(f"Error processing articles for narratives: {e}")
    
    def calculate_sentiment(self, title: str, summary: str) -> float:
        """Calculate basic sentiment score for news content"""
        try:
            text = (title + ' ' + summary).lower()
            
            # Positive sentiment keywords
            positive_keywords = [
                'surge', 'rally', 'bullish', 'pump', 'gain', 'rise', 'increase',
                'breakthrough', 'adoption', 'approval', 'success', 'growth', 'up'
            ]
            
            # Negative sentiment keywords  
            negative_keywords = [
                'crash', 'dump', 'bearish', 'fall', 'decline', 'drop', 'ban',
                'regulation', 'concern', 'risk', 'volatile', 'down', 'loss'
            ]
            
            positive_score = sum(1 for keyword in positive_keywords if keyword in text)
            negative_score = sum(1 for keyword in negative_keywords if keyword in text)
            
            # Normalize between -1 and 1
            if positive_score + negative_score == 0:
                return 0.0
            
            sentiment = (positive_score - negative_score) / (positive_score + negative_score)
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return 0.0
    
    def set_last_offset(self, offset: Optional[str]):
        """Set the last processed offset"""
        self.last_offset = offset
    
    def get_last_offset(self) -> Optional[str]:
        """Get the last processed offset"""
        return self.last_offset
    
    def get_status(self) -> Dict:
        """Get scanner status with real connectivity check"""
        connected = self._check_connectivity()
        return {
            "name": "news_scanner",
            "connected": connected,
            "last_offset": self.last_offset,
            "sources_count": len(self.news_sources),
            "processed_articles": len(self.processed_articles),
            "last_scan": datetime.utcnow().isoformat() + "Z"
        }
    
    def _check_connectivity(self) -> bool:
        """Check if news sources are accessible"""
        try:
            # Use aiohttp session if available for faster checks
            if hasattr(self, 'session') and self.session:
                return True  # Session exists and was created successfully
            else:
                # Fallback: quick connectivity test
                import requests
                test_response = requests.get("https://httpbin.org/status/200", timeout=2)
                return test_response.status_code == 200
        except Exception as e:
            logger.warning(f"News scanner connectivity check failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text for deduplication"""
        return ' '.join(text.split())
    
    def is_duplicate(self, url: str, title: str) -> bool:
        """Check if article is duplicate based on URL or title"""
        if url in self.seen_urls:
            return True
        if title.lower() in self.seen_titles:
            return True
        return False
    
    async def generate_enhanced_summary(self, title: str, summary: str) -> str:
        """Generate AI-enhanced summary if available"""
        if not self.ai_client:
            return summary[:100]  # Fallback to truncated summary
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ai_client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=f"Summarize this news in 1-2 sentences for market analysis: {title}. {summary}",
                    max_tokens=50,
                    temperature=0.3
                )
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logger.warning(f"AI summary generation failed: {e}")
            return summary[:100]
    
    async def generate_rationale(self, title: str, summary: str) -> str:
        """Generate 'Why It Matters' rationale using AI"""
        if not self.ai_client:
            return "Market relevance based on content analysis."
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ai_client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=f"Why is this news important for markets? {title}. {summary}",
                    max_tokens=30,
                    temperature=0.3
                )
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logger.warning(f"AI rationale generation failed: {e}")
            return "Market relevance based on content analysis."
    
    def update_hourly_counts(self, pub_time: datetime):
        """Update hourly article counts for trend analysis"""
        hour_key = pub_time.strftime('%Y-%m-%d %H:00')
        self.hourly_counts[hour_key] = self.hourly_counts.get(hour_key, 0) + 1
        
        # Keep only last 48 hours of data
        cutoff = datetime.utcnow() - timedelta(hours=48)
        cutoff_key = cutoff.strftime('%Y-%m-%d %H:00')
        self.hourly_counts = {k: v for k, v in self.hourly_counts.items() if k >= cutoff_key}
    
    def get_trend_data(self) -> Dict:
        """Get hourly trend data for the last 24 hours"""
        now = datetime.utcnow()
        labels = []
        data = []
        
        for i in range(24):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime('%Y-%m-%d %H:00')
            label = hour.strftime('%H:00')
            count = self.hourly_counts.get(hour_key, 0)
            
            labels.insert(0, label)
            data.insert(0, count)
        
        return {'labels': labels, 'data': data}