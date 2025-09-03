"""
Reddit Scanner Module
Implements PRAW-based Reddit data collection with rate limiting and safety controls
"""

import os
import praw
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import json

logger = logging.getLogger(__name__)

class RedditScanner:
    """Reddit scanner using PRAW with safety controls"""
    
    def __init__(self):
        self.reddit = None
        self.last_offset = None
        self.rate_limiter = {}
        self.recent_events = []  # Store recent events for screening
        self.mention_history = {}  # Track mention counts for anomaly detection
        self.setup_reddit()
    
    def setup_reddit(self):
        """Initialize Reddit API connection"""
        try:
            # Try read-only mode first (no username/password needed)
            self.reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID', 'default_client_id'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'default_secret'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'SocialIntelligenceBot/1.0')
            )
            
            # Test connection with a simple read operation instead of user.me()
            test_subreddit = self.reddit.subreddit('test')
            test_subreddit.display_name  # Simple read test
            logger.info("Reddit API connection established (read-only mode)")
            
        except Exception as e:
            logger.error(f"Failed to setup Reddit API: {e}")
            # Try with username/password as fallback
            try:
                username = os.getenv('REDDIT_USERNAME')
                password = os.getenv('REDDIT_PASSWORD')
                if username and password and username != 'default_user':
                    self.reddit = praw.Reddit(
                        client_id=os.getenv('REDDIT_CLIENT_ID', 'default_client_id'),
                        client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'default_secret'),
                        user_agent=os.getenv('REDDIT_USER_AGENT', 'SocialIntelligenceBot/1.0'),
                        username=username,
                        password=password
                    )
                    self.reddit.user.me()
                    logger.info("Reddit API connection established (authenticated mode)")
                else:
                    self.reddit = None
                    logger.warning("Reddit API connection failed - no valid credentials")
            except Exception as fallback_e:
                logger.error(f"Reddit fallback authentication also failed: {fallback_e}")
                self.reddit = None
    
    async def scan(self) -> List[Dict]:
        """Scan Reddit for relevant posts and comments"""
        if not self.reddit:
            # Reduced logging frequency for cleaner output
            return []
        
        try:
            events = []
            
            # Target subreddits for crypto/trading content
            subreddits = [
                'CryptoCurrency', 'Bitcoin', 'ethereum', 'CryptoMarkets',
                'altcoin', 'CryptoMoonShots', 'binance', 'trading',
                'SecurityAnalysis', 'investing', 'StockMarket'
            ]
            
            for subreddit_name in subreddits:
                if await self.check_rate_limit(subreddit_name):
                    sub_events = await self.scan_subreddit(subreddit_name)
                    events.extend(sub_events)
            
            # Store recent events for API access (keep last 50)
            self.recent_events.extend(events)
            if len(self.recent_events) > 50:
                self.recent_events = self.recent_events[-50:]
            
            logger.info(f"Collected {len(events)} Reddit events")
            return events
            
        except Exception as e:
            logger.error(f"Error scanning Reddit: {e}")
            return []
    
    async def check_rate_limit(self, subreddit: str) -> bool:
        """Check if we can safely scan a subreddit"""
        now = time.time()
        
        if subreddit not in self.rate_limiter:
            self.rate_limiter[subreddit] = now
            return True
        
        # Allow scanning every 2 minutes per subreddit
        if now - self.rate_limiter[subreddit] > 120:
            self.rate_limiter[subreddit] = now
            return True
        
        return False
    
    async def scan_subreddit(self, subreddit_name: str) -> List[Dict]:
        """Scan a specific subreddit for new content"""
        try:
            if not self.reddit:
                logger.warning("Reddit API not available for subreddit scanning")
                return []
            
            subreddit = self.reddit.subreddit(subreddit_name)
            events = []
            
            # Scan hot posts
            for submission in subreddit.hot(limit=10):
                event = {
                    "source": "reddit",
                    "id": f"post_{submission.id}",
                    "timestamp": datetime.fromtimestamp(submission.created_utc).isoformat() + "Z",
                    "payload": {
                        "type": "submission",
                        "subreddit": subreddit_name,
                        "title": submission.title,
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "url": submission.url,
                        "selftext": submission.selftext[:500] if submission.selftext else "",
                        "author": str(submission.author) if submission.author else "deleted",
                        "upvote_ratio": submission.upvote_ratio,
                        "gilded": submission.gilded,
                        "permalink": submission.permalink,
                        "sentiment_score": self.calculate_sentiment_score(submission.title, submission.selftext),
                        "mention_count": submission.num_comments
                    }
                }
                
                # Check for mention spike anomaly
                spike_anomaly = self.detect_mention_anomaly(submission.num_comments, subreddit_name)
                if spike_anomaly:
                    event['payload']['anomaly_detected'] = True
                
                events.append(event)
            
            # Scan recent comments from hot posts
            for submission in subreddit.hot(limit=5):
                try:
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments[:5]:  # Top 5 comments
                        if comment.score > 5:  # Only high-scoring comments
                            comment_event = {
                                "source": "reddit",
                                "id": f"comment_{comment.id}",
                                "timestamp": datetime.fromtimestamp(comment.created_utc).isoformat() + "Z",
                                "payload": {
                                    "type": "comment",
                                    "subreddit": subreddit_name,
                                    "submission_title": submission.title,
                                    "body": comment.body[:300] if comment.body else "",
                                    "score": comment.score,
                                    "author": str(comment.author) if comment.author else "deleted",
                                    "gilded": comment.gilded,
                                    "permalink": comment.permalink
                                }
                            }
                            events.append(comment_event)
                except Exception as e:
                    logger.warning(f"Error processing comments for submission {submission.id}: {e}")
                    continue
            
            logger.debug(f"Scanned {subreddit_name}: {len(events)} events")
            return events
            
        except Exception as e:
            # Only log critical errors, skip 401 auth errors to reduce noise
            if "401" not in str(e):
                logger.error(f"Error scanning subreddit {subreddit_name}: {e}")
            return []
    
    def interactive_screen(self, filters: Dict) -> List[Dict]:
        """
        Screen Reddit events by sentiment score or mentions.
        """
        events = self.recent_events  # Use stored recent events
        screened = []
        for event in events:
            if 'min_sentiment' in filters and event['payload'].get('sentiment_score', 0) < filters['min_sentiment']:
                continue
            if 'min_mentions' in filters and event['payload'].get('mention_count', 0) < filters['min_mentions']:
                continue
            screened.append(event)
        return screened
    
    def calculate_sentiment_score(self, title: str, text: str) -> float:
        """
        Simple sentiment calculation based on positive/negative keywords.
        """
        try:
            content = (title + " " + (text or "")).lower()
            
            positive_words = ['bullish', 'moon', 'pump', 'buy', 'hodl', 'diamond', 'gains', 'profit', 'up', 'rise']
            negative_words = ['bearish', 'dump', 'sell', 'crash', 'loss', 'down', 'fall', 'fear', 'panic', 'rekt']
            
            positive_count = sum(1 for word in positive_words if word in content)
            negative_count = sum(1 for word in negative_words if word in content)
            
            total_words = len(content.split())
            if total_words == 0:
                return 0.5  # Neutral
            
            # Calculate sentiment score (0-1 scale)
            sentiment = (positive_count - negative_count) / max(total_words, 1)
            return max(0, min(1, sentiment + 0.5))  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return 0.5  # Neutral default
    
    def detect_mention_anomaly(self, current: int, subreddit: str) -> bool:
        """
        Detect mention spikes based on historical averages.
        """
        try:
            if subreddit not in self.mention_history:
                self.mention_history[subreddit] = []
            
            # Add current count
            self.mention_history[subreddit].append(current)
            
            # Keep only last 20 entries
            if len(self.mention_history[subreddit]) > 20:
                self.mention_history[subreddit] = self.mention_history[subreddit][-20:]
            
            # Calculate average (need at least 3 data points)
            if len(self.mention_history[subreddit]) < 3:
                return False
            
            recent_history = self.mention_history[subreddit][:-1]  # Exclude current
            avg = sum(recent_history) / len(recent_history)
            
            # Spike if current is 3x average
            return current > avg * 3.0
            
        except Exception as e:
            logger.error(f"Error in mention anomaly detection: {e}")
            return False
    
    def set_last_offset(self, offset: Optional[str]):
        """Set the last processed offset"""
        self.last_offset = offset
    
    def get_last_offset(self) -> Optional[str]:
        """Get the last processed offset"""
        return self.last_offset
    
    def get_status(self) -> Dict:
        """Get scanner status with real connectivity check"""
        connected = self._check_reddit_connectivity()
        return {
            "name": "reddit_scanner",
            "connected": connected,
            "last_offset": self.last_offset,
            "rate_limited_subreddits": len(self.rate_limiter),
            "last_scan": datetime.utcnow().isoformat() + "Z"
        }
    
    def _check_reddit_connectivity(self) -> bool:
        """Check if Reddit API is accessible"""
        if self.reddit is None:
            return False
        
        try:
            # Try a lightweight API call to verify connectivity
            test_subreddit = self.reddit.subreddit('test')
            # Just check if we can access the display name (minimal API call)
            _ = test_subreddit.display_name
            return True
        except Exception as e:
            logger.warning(f"Reddit scanner connectivity check failed: {e}")
            return False
