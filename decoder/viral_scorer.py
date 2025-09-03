"""
Viral Scorer Module
Implements viral score computation for pattern prioritization
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import deque
import math

logger = logging.getLogger(__name__)

class ViralScorer:
    """Computes viral scores for patterns to prioritize actions"""
    
    def __init__(self):
        self.recent_alerts = deque(maxlen=100)
        self.score_history = {}
        self.asset_momentum = {}
    
    async def compute_score(self, pattern: Dict) -> float:
        """Compute viral score for a pattern"""
        try:
            base_score = 0.0
            
            # Get pattern type and signals
            pattern_type = pattern.get('type', '')
            signals = pattern.get('signals', {})
            source = pattern.get('source', '')
            
            # Type-specific scoring
            if pattern_type == 'mention_spike':
                base_score = await self.score_mention_spike(pattern, signals)
            elif pattern_type == 'price_movement':
                base_score = await self.score_price_movement(pattern, signals)
            elif pattern_type == 'volume_spike':
                base_score = await self.score_volume_spike(pattern, signals)
            elif pattern_type == 'news_impact':
                base_score = await self.score_news_impact(pattern, signals)
            elif pattern_type == 'cross_source_correlation':
                base_score = await self.score_cross_correlation(pattern, signals)
            elif pattern_type == 'sentiment_signal':
                base_score = await self.score_sentiment(pattern, signals)
            
            # Apply multipliers
            multiplied_score = await self.apply_multipliers(pattern, base_score)
            
            # Final viral score (0-100 scale)
            viral_score = min(multiplied_score * 100, 100.0)
            
            # Store in history
            self.store_score(pattern, viral_score)
            
            logger.debug(f"Computed viral score {viral_score:.2f} for {pattern_type}")
            return viral_score
            
        except Exception as e:
            logger.error(f"Error computing viral score: {e}")
            return 0.0
    
    async def score_mention_spike(self, pattern: Dict, signals: Dict) -> float:
        """Score Reddit mention spikes"""
        try:
            # Base score from mention velocity
            mention_count = signals.get('mention_count', 0)
            velocity = signals.get('velocity', 0)
            cross_subreddit = signals.get('cross_subreddit', 1)
            avg_score = signals.get('avg_score', 0)
            
            # Velocity component (0.0 - 0.5)
            velocity_norm = min(velocity / 10.0, 0.5)  # Normalize to max 0.5
            
            # Cross-platform reach (0.0 - 0.3)
            reach_norm = min(cross_subreddit / 5.0, 0.3)  # Multiple subreddits
            
            # Engagement quality (0.0 - 0.2)
            engagement_norm = min(avg_score / 1000.0, 0.2)  # High-scoring posts
            
            score = velocity_norm + reach_norm + engagement_norm
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring mention spike: {e}")
            return 0.0
    
    async def score_price_movement(self, pattern: Dict, signals: Dict) -> float:
        """Score price movement patterns"""
        try:
            change_percent = abs(signals.get('price_change_percent', 0))
            volume = signals.get('volume', 0)
            breakout_signal = signals.get('breakout_signal', False)
            volume_confirmation = signals.get('volume_confirmation', False)
            
            # Price change component (0.0 - 0.6)
            price_norm = min(change_percent / 20.0, 0.6)  # 20% = max score
            
            # Volume confirmation (0.0 - 0.2)
            volume_component = 0.2 if volume_confirmation else 0.1
            
            # Breakout bonus (0.0 - 0.2)
            breakout_component = 0.2 if breakout_signal else 0.0
            
            score = price_norm + volume_component + breakout_component
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring price movement: {e}")
            return 0.0
    
    async def score_volume_spike(self, pattern: Dict, signals: Dict) -> float:
        """Score volume spike patterns"""
        try:
            volume_ratio = signals.get('volume_ratio', 1)
            unusual_activity = signals.get('unusual_activity', False)
            
            # Volume ratio component (0.0 - 0.7)
            ratio_norm = min(math.log(volume_ratio) / 5.0, 0.7)  # Log scale
            
            # Unusual activity bonus (0.0 - 0.3)
            activity_bonus = 0.3 if unusual_activity else 0.1
            
            score = ratio_norm + activity_bonus
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring volume spike: {e}")
            return 0.0
    
    async def score_news_impact(self, pattern: Dict, signals: Dict) -> float:
        """Score news impact patterns"""
        try:
            relevance_score = signals.get('relevance_score', 0)
            potential_impact = signals.get('potential_impact', 'low')
            urgency = signals.get('urgency', 0)
            mentioned_assets = signals.get('mentioned_assets', [])
            
            # Relevance component (0.0 - 0.4)
            relevance_component = relevance_score * 0.4
            
            # Impact level component (0.0 - 0.4)
            impact_scores = {'low': 0.1, 'medium': 0.25, 'high': 0.4}
            impact_component = impact_scores.get(potential_impact, 0.1)
            
            # Urgency component (0.0 - 0.2)
            urgency_component = urgency * 0.2
            
            score = relevance_component + impact_component + urgency_component
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring news impact: {e}")
            return 0.0
    
    async def score_cross_correlation(self, pattern: Dict, signals: Dict) -> float:
        """Score cross-source correlation patterns"""
        try:
            correlation_strength = signals.get('correlation_strength', 0)
            sources = signals.get('sources', [])
            mention_count = signals.get('mention_count', 0)
            confirmation_level = signals.get('confirmation_level', 'low')
            
            # Correlation strength (0.0 - 0.5)
            correlation_component = correlation_strength * 0.5
            
            # Source diversity (0.0 - 0.3)
            diversity_component = min(len(sources) / 3.0, 1.0) * 0.3
            
            # Confirmation level (0.0 - 0.2)
            confirmation_scores = {'low': 0.05, 'medium': 0.125, 'high': 0.2}
            confirmation_component = confirmation_scores.get(confirmation_level, 0.05)
            
            score = correlation_component + diversity_component + confirmation_component
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring cross correlation: {e}")
            return 0.0
    
    async def score_sentiment(self, pattern: Dict, signals: Dict) -> float:
        """Score sentiment signals"""
        try:
            sentiment_score = abs(signals.get('sentiment_score', 0))
            engagement_score = signals.get('engagement_score', 0)
            discussion_level = signals.get('discussion_level', 0)
            
            # Sentiment strength (0.0 - 0.4)
            sentiment_component = sentiment_score * 0.4
            
            # Engagement component (0.0 - 0.3)
            engagement_norm = min(engagement_score / 500.0, 0.3)
            
            # Discussion level (0.0 - 0.3)
            discussion_norm = min(discussion_level / 100.0, 0.3)
            
            score = sentiment_component + engagement_norm + discussion_norm
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring sentiment: {e}")
            return 0.0
    
    async def apply_multipliers(self, pattern: Dict, base_score: float) -> float:
        """Apply multipliers based on asset momentum and timing"""
        try:
            multiplied_score = base_score
            asset = pattern.get('asset', '')
            
            # Asset momentum multiplier
            if asset:
                momentum = await self.get_asset_momentum(asset)
                momentum_multiplier = 1.0 + (momentum * 0.3)  # Up to 30% bonus
                multiplied_score *= momentum_multiplier
            
            # Time-of-day multiplier (higher during market hours)
            time_multiplier = self.get_time_multiplier()
            multiplied_score *= time_multiplier
            
            # Recency decay (newer patterns get higher scores)
            recency_multiplier = self.get_recency_multiplier(pattern.get('timestamp', ''))
            multiplied_score *= recency_multiplier
            
            return multiplied_score
            
        except Exception as e:
            logger.error(f"Error applying multipliers: {e}")
            return base_score
    
    async def get_asset_momentum(self, asset: str) -> float:
        """Calculate asset momentum from recent patterns"""
        try:
            if asset not in self.asset_momentum:
                return 0.0
            
            recent_scores = self.asset_momentum[asset]
            if not recent_scores:
                return 0.0
            
            # Calculate momentum as trend in recent scores
            if len(recent_scores) < 3:
                return 0.0
            
            recent_list = list(recent_scores)
            recent_avg = np.mean(recent_list[-3:])
            older_avg = np.mean(recent_list[:-3]) if len(recent_scores) > 3 else 0
            
            momentum = (recent_avg - older_avg) / max(older_avg, 1.0)
            return max(-0.5, min(0.5, momentum))  # Cap between -0.5 and 0.5
            
        except Exception as e:
            logger.error(f"Error calculating asset momentum: {e}")
            return 0.0
    
    def get_time_multiplier(self) -> float:
        """Get time-based multiplier (higher during active trading hours)"""
        try:
            current_hour = datetime.utcnow().hour
            
            # Peak trading hours (UTC): 13-21 (covers US + Europe)
            if 13 <= current_hour <= 21:
                return 1.2
            # Moderate hours
            elif 9 <= current_hour <= 13 or 21 <= current_hour <= 23:
                return 1.0
            # Off-peak hours
            else:
                return 0.8
                
        except Exception as e:
            logger.error(f"Error calculating time multiplier: {e}")
            return 1.0
    
    def get_recency_multiplier(self, timestamp: str) -> float:
        """Get recency-based multiplier"""
        try:
            if not timestamp:
                return 1.0
            
            event_time = datetime.fromisoformat(timestamp.replace('Z', ''))
            age_hours = (datetime.utcnow() - event_time).total_seconds() / 3600
            
            # Decay function: newer = higher multiplier
            if age_hours < 0.5:  # Very recent
                return 1.3
            elif age_hours < 2:   # Recent
                return 1.1
            elif age_hours < 6:   # Moderately recent
                return 1.0
            else:                 # Old
                return 0.8
                
        except Exception as e:
            logger.error(f"Error calculating recency multiplier: {e}")
            return 1.0
    
    def store_score(self, pattern: Dict, score: float):
        """Store pattern score for tracking and momentum calculation"""
        try:
            asset = pattern.get('asset', '')
            timestamp = pattern.get('timestamp', '')
            
            # Store in recent alerts if high scoring
            if score > 50:  # Threshold for significant alerts
                alert = {
                    'id': pattern.get('id', ''),
                    'score': score,
                    'asset': asset,
                    'type': pattern.get('type', ''),
                    'timestamp': timestamp
                }
                self.recent_alerts.append(alert)
            
            # Update asset momentum tracking
            if asset:
                if asset not in self.asset_momentum:
                    self.asset_momentum[asset] = deque(maxlen=20)
                self.asset_momentum[asset].append(score / 100.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Error storing score: {e}")
    
    def get_recent_alerts(self) -> List[Dict]:
        """Get recent high-scoring alerts"""
        return list(self.recent_alerts)
    
    def get_asset_scores(self, asset: str) -> List[float]:
        """Get recent scores for an asset"""
        return list(self.asset_momentum.get(asset, []))
