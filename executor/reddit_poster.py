"""
Reddit Poster Module
Implements safe Reddit posting with rate limiting and ban detection
"""

import os
import praw
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import random
import json

logger = logging.getLogger(__name__)

class RedditPoster:
    """Safe Reddit posting with circuit breakers and rate limiting"""
    
    def __init__(self):
        self.reddit = None
        self.recent_posts = []
        self.post_cooldowns = {}
        self.subreddit_limits = {}
        self.banned_subreddits = set()
        self.personas = self.load_personas()
        self.setup_reddit()
    
    def setup_reddit(self):
        """Initialize Reddit API for posting"""
        try:
            # For posting, we need authenticated access
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
                
                # Test connection
                self.reddit.user.me()
                logger.info("Reddit posting API connection established (authenticated mode)")
            else:
                # Try read-only mode for testing
                self.reddit = praw.Reddit(
                    client_id=os.getenv('REDDIT_CLIENT_ID', 'default_client_id'),
                    client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'default_secret'),
                    user_agent=os.getenv('REDDIT_USER_AGENT', 'SocialIntelligenceBot/1.0')
                )
                test_subreddit = self.reddit.subreddit('test')
                test_subreddit.display_name  # Simple read test
                logger.info("Reddit posting API connection established (read-only mode - posting disabled)")
            
        except Exception as e:
            logger.error(f"Failed to setup Reddit posting API: {e}")
            self.reddit = None
    
    def load_personas(self) -> List[Dict]:
        """Load posting personas for variety"""
        return [
            {
                'name': 'analytical',
                'style': 'technical',
                'templates': [
                    "Interesting {} movement. Volume suggests {}. Thoughts?",
                    "Technical analysis shows {} forming a pattern. Worth watching.",
                    "{} breaking key levels. Risk/reward looking favorable."
                ]
            },
            {
                'name': 'news_focused',
                'style': 'informative',
                'templates': [
                    "News: {}. This could impact {} significantly.",
                    "Development in {}: {}. Market implications?",
                    "Update on {}: {}. Community thoughts?"
                ]
            },
            {
                'name': 'community',
                'style': 'engaging',
                'templates': [
                    "Anyone else watching {}? Seeing some interesting signals.",
                    "{} community - what's your take on recent developments?",
                    "Thoughts on {} recent performance? Data suggests {}."
                ]
            },
            {
                'name': 'educational',
                'style': 'helpful',
                'templates': [
                    "For those following {}: key levels to watch are {}.",
                    "Quick {} analysis: {}. Educational purposes only.",
                    "{} update: {}. Always DYOR!"
                ]
            },
            {
                'name': 'collaborative',
                'style': 'community-driven',
                'templates': [
                    "Sharing {} strategy - let's improve it together!",
                    "Community input: {} pattern analysis. Thoughts?",
                    "Collaborative trading: {} signals discussion."
                ]
            }
        ]
    
    async def safe_post(self, pattern: Dict, score: float):
        """Safely post to Reddit with all safety checks"""
        if not self.reddit:
            logger.warning("Reddit API not available for posting")
            return
        
        try:
            # Safety checks
            if not await self.check_posting_safety(pattern, score):
                return
            
            # Generate post content
            post_data = await self.generate_post_content(pattern, score)
            if not post_data:
                return
            
            # Find appropriate subreddit
            target_subreddit = await self.select_target_subreddit(pattern)
            if not target_subreddit:
                return
            
            # Rate limit check
            if not await self.check_rate_limits(target_subreddit):
                logger.info(f"Rate limited for {target_subreddit}, skipping post")
                return
            
            # Simulate post (replace with actual posting in production)
            success = await self.execute_post(target_subreddit, post_data)
            
            if success:
                await self.record_post(target_subreddit, post_data, pattern, score)
                logger.info(f"Posted to r/{target_subreddit}: {post_data['title'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error in safe posting: {e}")
    
    async def check_posting_safety(self, pattern: Dict, score: float) -> bool:
        """Check if posting is safe and appropriate"""
        try:
            # Minimum score threshold
            min_score = float(os.getenv('REDDIT_POST_THRESHOLD', '70'))
            if score < min_score:
                return False
            
            # Check daily post limit
            today = datetime.utcnow().date()
            today_posts = [p for p in self.recent_posts 
                          if datetime.fromisoformat(p['timestamp']).date() == today]
            
            max_daily_posts = int(os.getenv('MAX_DAILY_POSTS', '5'))
            if len(today_posts) >= max_daily_posts:
                logger.info(f"Daily post limit reached: {len(today_posts)}")
                return False
            
            # Check for duplicate patterns
            recent_assets = [p['asset'] for p in self.recent_posts[-10:]]
            if pattern.get('asset') in recent_assets[-3:]:  # Don't repeat same asset
                return False
            
            # Check account age and karma (placeholder - implement based on needs)
            account_safe = await self.check_account_safety()
            if not account_safe:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking posting safety: {e}")
            return False
    
    async def generate_post_content(self, pattern: Dict, score: float) -> Optional[Dict]:
        """Generate post content based on pattern and persona"""
        try:
            # Select persona based on pattern type
            persona = self.select_persona(pattern)
            
            # Generate title and content
            asset = pattern.get('asset', 'CRYPTO')
            pattern_type = pattern.get('type', '')
            
            if pattern_type == 'mention_spike':
                title, content = self.generate_mention_spike_post(pattern, persona, asset)
            elif pattern_type == 'price_movement':
                title, content = self.generate_price_movement_post(pattern, persona, asset)
            elif pattern_type == 'news_impact':
                title, content = self.generate_news_impact_post(pattern, persona, asset)
            elif pattern_type == 'cross_source_correlation':
                title, content = self.generate_correlation_post(pattern, persona, asset)
            else:
                return None
            
            if not title or not content:
                return None
            
            return {
                'title': title,
                'content': content,
                'persona': persona['name'],
                'flair': self.select_post_flair(pattern_type)
            }
            
        except Exception as e:
            logger.error(f"Error generating post content: {e}")
            return None
    
    def select_persona(self, pattern: Dict) -> Dict:
        """Select appropriate persona for the pattern"""
        pattern_type = pattern.get('type', '')
        
        if pattern_type == 'news_impact':
            return next(p for p in self.personas if p['name'] == 'news_focused')
        elif pattern_type == 'price_movement':
            return next(p for p in self.personas if p['name'] == 'analytical')
        else:
            return random.choice(self.personas)
    
    def generate_mention_spike_post(self, pattern: Dict, persona: Dict, asset: str) -> tuple:
        """Generate post for mention spike patterns"""
        signals = pattern.get('signals', {})
        mention_count = signals.get('mention_count', 0)
        velocity = signals.get('velocity', 0)
        
        templates = persona['templates']
        template = random.choice(templates)
        
        title = f"{asset} seeing increased discussion - {mention_count} mentions in last hour"
        content = f"Noticed {asset} getting {mention_count} mentions across multiple subreddits in the past hour (velocity: {velocity:.1f}/hr). Anyone else seeing this trend? What's driving the discussion?\n\nAlways do your own research - this is just observational data."
        
        return title, content
    
    def generate_price_movement_post(self, pattern: Dict, persona: Dict, asset: str) -> tuple:
        """Generate post for price movement patterns"""
        signals = pattern.get('signals', {})
        change_percent = signals.get('price_change_percent', 0)
        volume_confirmation = signals.get('volume_confirmation', False)
        
        direction = "up" if change_percent > 0 else "down"
        volume_text = "with strong volume" if volume_confirmation else "on moderate volume"
        
        title = f"{asset} moving {direction} {abs(change_percent):.1f}% {volume_text}"
        content = f"Technical observation: {asset} showing {abs(change_percent):.1f}% movement {direction} {volume_text}. Key levels to watch for continuation or reversal.\n\nNot financial advice - just sharing technical observations."
        
        return title, content
    
    def generate_news_impact_post(self, pattern: Dict, persona: Dict, asset: str) -> tuple:
        """Generate post for news impact patterns"""
        signals = pattern.get('signals', {})
        keywords = signals.get('keywords', [])
        impact = signals.get('potential_impact', 'medium')
        
        keyword_text = ", ".join(keywords[:3]) if keywords else "development"
        
        title = f"{asset} news update - {impact} impact potential"
        content = f"Recent {asset} news involving {keyword_text}. Potential {impact} impact on price action. Community thoughts on implications?\n\nAs always, DYOR and consider multiple sources."
        
        return title, content
    
    def generate_correlation_post(self, pattern: Dict, persona: Dict, asset: str) -> tuple:
        """Generate post for cross-source correlation patterns"""
        signals = pattern.get('signals', {})
        sources = signals.get('sources', [])
        correlation_strength = signals.get('correlation_strength', 0)
        
        source_text = " + ".join(sources)
        
        title = f"{asset} multi-source signals - {source_text} correlation"
        content = f"Interesting correlation spotted: {asset} showing signals across {source_text} simultaneously. Correlation strength: {correlation_strength:.2f}. Worth monitoring.\n\nData-driven observation - not trading advice."
        
        return title, content
    
    async def select_target_subreddit(self, pattern: Dict) -> Optional[str]:
        """Select appropriate subreddit for posting"""
        try:
            asset = pattern.get('asset', '').upper()
            pattern_type = pattern.get('type', '')
            
            # Asset-specific subreddits
            asset_subs = {
                'BTC': ['Bitcoin', 'CryptoCurrency'],
                'ETH': ['ethereum', 'CryptoCurrency'],
                'ADA': ['cardano', 'CryptoCurrency'],
                'SOL': ['solana', 'CryptoCurrency']
            }
            
            # General subreddits by pattern type
            pattern_subs = {
                'price_movement': ['CryptoMarkets', 'CryptoCurrency', 'trading'],
                'news_impact': ['CryptoCurrency', 'CryptoNews'],
                'mention_spike': ['CryptoCurrency', 'CryptoMoonShots'],
                'cross_source_correlation': ['CryptoCurrency', 'CryptoMarkets']
            }
            
            # Get possible subreddits
            possible_subs = []
            if asset in asset_subs:
                possible_subs.extend(asset_subs[asset])
            if pattern_type in pattern_subs:
                possible_subs.extend(pattern_subs[pattern_type])
            
            # Filter out banned subreddits
            available_subs = [sub for sub in possible_subs if sub not in self.banned_subreddits]
            
            if not available_subs:
                return None
            
            # Select based on current limits
            for sub in available_subs:
                if await self.check_subreddit_limits(sub):
                    return sub
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting target subreddit: {e}")
            return None
    
    async def check_rate_limits(self, subreddit: str) -> bool:
        """Check if we can post to this subreddit (rate limiting)"""
        now = time.time()
        
        # Global cooldown between any posts
        if hasattr(self, 'last_post_time'):
            if now - self.last_post_time < 1800:  # 30 minutes between posts
                return False
        
        # Subreddit-specific cooldown
        if subreddit in self.post_cooldowns:
            last_post = self.post_cooldowns[subreddit]
            if now - last_post < 3600:  # 1 hour per subreddit
                return False
        
        return True
    
    async def check_subreddit_limits(self, subreddit: str) -> bool:
        """Check subreddit-specific posting limits"""
        # Check daily limit per subreddit
        today = datetime.utcnow().date()
        today_posts = [p for p in self.recent_posts 
                      if p['subreddit'] == subreddit and 
                      datetime.fromisoformat(p['timestamp']).date() == today]
        
        max_per_subreddit = int(os.getenv('MAX_POSTS_PER_SUBREDDIT', '1'))
        return len(today_posts) < max_per_subreddit
    
    def select_post_flair(self, pattern_type: str) -> Optional[str]:
        """Select appropriate post flair"""
        flair_map = {
            'price_movement': 'Technical Analysis',
            'news_impact': 'News',
            'mention_spike': 'Discussion',
            'cross_source_correlation': 'Analysis'
        }
        return flair_map.get(pattern_type)
    
    async def execute_post(self, subreddit: str, post_data: Dict) -> bool:
        """Execute the actual Reddit post (simulation mode)"""
        try:
            # In simulation mode, just log the post
            logger.info(f"[SIMULATION] Would post to r/{subreddit}:")
            logger.info(f"Title: {post_data['title']}")
            logger.info(f"Content: {post_data['content'][:100]}...")
            
            # In production, uncomment this:
            # subreddit_obj = self.reddit.subreddit(subreddit)
            # submission = subreddit_obj.submit(
            #     title=post_data['title'],
            #     selftext=post_data['content'],
            #     flair_text=post_data.get('flair')
            # )
            
            # Simulate random delay
            await asyncio.sleep(random.uniform(1, 3))
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing post: {e}")
            return False
    
    async def record_post(self, subreddit: str, post_data: Dict, pattern: Dict, score: float):
        """Record successful post for tracking"""
        try:
            post_record = {
                'id': f"post_{int(time.time())}",
                'subreddit': subreddit,
                'title': post_data['title'],
                'persona': post_data['persona'],
                'asset': pattern.get('asset', ''),
                'pattern_type': pattern.get('type', ''),
                'score': score,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            self.recent_posts.append(post_record)
            
            # Update cooldowns
            self.last_post_time = time.time()
            self.post_cooldowns[subreddit] = time.time()
            
            # Limit recent posts memory
            self.recent_posts = self.recent_posts[-50:]
            
        except Exception as e:
            logger.error(f"Error recording post: {e}")
    
    async def check_account_safety(self) -> bool:
        """Check if account is safe to post (placeholder)"""
        # Implement account age, karma, and ban checks
        # For now, always return True
        return True
    
    def get_recent_posts(self) -> List[Dict]:
        """Get recent posts for state management"""
        return self.recent_posts
    
    def detect_shadowban(self) -> bool:
        """Detect if account is shadowbanned (placeholder)"""
        # Implement shadowban detection logic
        return False
    
    async def share_strategy(self, strategy: Dict, subreddit: str) -> bool:
        """
        Share trading strategies with community for collaboration.
        """
        try:
            # Generate strategy post content
            title = f"Community Strategy Share: {strategy.get('name', 'New Strategy')}"
            content = f"""
Shared Strategy: {strategy.get('name', '')}

Description: {strategy.get('description', '')}

Key Parameters:
- Entry Rules: {strategy.get('entry_rules', '')}
- Exit Rules: {strategy.get('exit_rules', '')}
- Risk Level: {strategy.get('risk_level', 'medium')}

Performance: {strategy.get('backtest_performance', 'N/A')}

What do you think? Improvements? Results from testing?

#CommunityStrategy #TradingIdeas
"""
            
            post_data = {
                'title': title,
                'content': content,
                'flair': 'Strategy Discussion'
            }
            
            # Post to subreddit
            success = await self.execute_post(subreddit, post_data)
            if success:
                logger.info(f"Strategy shared to r/{subreddit}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error sharing strategy: {e}")
            return False
