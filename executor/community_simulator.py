"""
Enhanced Community Simulator
Extends reddit_poster.py with advanced simulation and real engagement hooks
"""

import asyncio
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import time

logger = logging.getLogger(__name__)

class CommunitySimulator:
    """Enhanced community simulation with real engagement potential"""
    
    def __init__(self, reddit_poster=None):
        self.posts = []
        self.engagement_metrics = {}
        self.reddit_poster = reddit_poster
        self.user_personas = self.generate_user_personas()
        self.engagement_patterns = self.load_engagement_patterns()
        
    def generate_user_personas(self) -> List[Dict]:
        """Generate diverse user personas for realistic simulation"""
        return [
            {
                'type': 'technical_analyst',
                'engagement_style': 'analytical',
                'response_probability': 0.7,
                'typical_responses': [
                    'Looking at the technicals, this makes sense',
                    'RSI is showing {sentiment} divergence here',
                    'Support/resistance levels align with this analysis',
                    'Volume profile suggests {direction} movement'
                ]
            },
            {
                'type': 'news_trader',
                'engagement_style': 'information_focused',
                'response_probability': 0.6,
                'typical_responses': [
                    'This correlates with recent news about {topic}',
                    'Fundamental analysis supports this view',
                    'Market reaction to {event} seems delayed',
                    'Institutional activity suggests {sentiment}'
                ]
            },
            {
                'type': 'hodler',
                'engagement_style': 'long_term',
                'response_probability': 0.4,
                'typical_responses': [
                    'Short-term noise, long-term this looks bullish',
                    'DCA strategy remains unchanged',
                    'These dips are buying opportunities',
                    'Zoom out - the trend is still up'
                ]
            },
            {
                'type': 'day_trader',
                'engagement_style': 'action_oriented',
                'response_probability': 0.8,
                'typical_responses': [
                    'Already positioned for this move',
                    'Stop loss at {level}, target {target}',
                    'Quick scalp opportunity here',
                    'Risk/reward ratio looks good'
                ]
            },
            {
                'type': 'community_builder',
                'engagement_style': 'collaborative',
                'response_probability': 0.9,
                'typical_responses': [
                    'Great analysis! What do others think?',
                    'Adding this to our community discussion',
                    'Would love to hear more perspectives',
                    'This deserves wider attention'
                ]
            }
        ]
    
    def load_engagement_patterns(self) -> Dict:
        """Load realistic engagement patterns"""
        return {
            'peak_hours': [9, 10, 11, 14, 15, 16, 20, 21],  # UTC hours with high engagement
            'response_delays': {
                'immediate': 0.2,  # 20% respond within minutes
                'quick': 0.4,      # 40% respond within hour
                'delayed': 0.3,    # 30% respond within day
                'late': 0.1        # 10% respond later
            },
            'viral_thresholds': {
                'comment_velocity': 5,  # comments per hour
                'upvote_ratio': 0.8,
                'cross_platform_mentions': 3
            }
        }
    
    async def simulate_post(self, content: str, subreddit: str = 'general', metadata: Dict = None) -> Dict:
        """
        Simulate posting with enhanced community responses and metrics
        """
        try:
            post_id = f"post_{int(time.time())}_{random.randint(1000, 9999)}"
            current_hour = datetime.utcnow().hour
            
            # Calculate engagement multiplier based on timing
            engagement_multiplier = 1.5 if current_hour in self.engagement_patterns['peak_hours'] else 1.0
            
            post_data = {
                'id': post_id,
                'content': content,
                'subreddit': subreddit,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': metadata or {},
                'replies': [],
                'metrics': {
                    'views': 0,
                    'upvotes': 0,
                    'downvotes': 0,
                    'comments': 0,
                    'shares': 0
                },
                'engagement_score': 0.0,
                'viral_indicators': {
                    'comment_velocity': 0,
                    'cross_platform_buzz': False,
                    'influencer_engagement': False
                }
            }
            
            self.posts.append(post_data)
            
            # Simulate network delay
            await asyncio.sleep(random.uniform(1, 3))
            
            # Generate initial engagement
            await self.generate_community_responses(post_id, engagement_multiplier)
            
            # Calculate engagement metrics
            await self.calculate_engagement_metrics(post_id)
            
            logger.info(f"Simulated post {post_id} in r/{subreddit} with {len(post_data['replies'])} responses")
            
            return {
                'success': True,
                'post_id': post_id,
                'engagement_preview': {
                    'responses': len(post_data['replies']),
                    'estimated_reach': post_data['metrics']['views'],
                    'engagement_score': post_data['engagement_score']
                }
            }
            
        except Exception as e:
            logger.error(f"Error simulating post: {e}")
            return {'success': False, 'error': str(e)}
    
    async def generate_community_responses(self, post_id: str, engagement_multiplier: float):
        """Generate realistic community responses"""
        try:
            post = next((p for p in self.posts if p['id'] == post_id), None)
            if not post:
                return
            
            # Determine number of responses based on content and timing
            base_responses = random.randint(3, 15)
            num_responses = int(base_responses * engagement_multiplier)
            
            for i in range(num_responses):
                # Select persona
                persona = random.choice(self.user_personas)
                
                # Check if this persona will engage
                if random.random() > persona['response_probability']:
                    continue
                
                # Generate response based on persona
                response_template = random.choice(persona['typical_responses'])
                
                # Fill in template variables based on post content
                response = self.customize_response(response_template, post)
                
                # Simulate response timing
                delay_type = random.choices(
                    list(self.engagement_patterns['response_delays'].keys()),
                    weights=list(self.engagement_patterns['response_delays'].values())
                )[0]
                
                response_data = {
                    'id': f"reply_{post_id}_{i}",
                    'user': f"{persona['type']}_{random.randint(1, 100)}",
                    'text': response,
                    'persona': persona['type'],
                    'timestamp': (datetime.utcnow() + timedelta(minutes=self.get_delay_minutes(delay_type))).isoformat(),
                    'upvotes': random.randint(0, 20),
                    'sentiment': self.analyze_response_sentiment(response)
                }
                
                post['replies'].append(response_data)
            
            # Simulate threading (replies to replies)
            if len(post['replies']) > 3:
                await self.generate_threaded_responses(post_id)
                
        except Exception as e:
            logger.error(f"Error generating community responses: {e}")
    
    def customize_response(self, template: str, post: Dict) -> str:
        """Customize response template based on post content"""
        try:
            content = post['content'].lower()
            
            # Extract sentiment
            if any(word in content for word in ['bullish', 'up', 'rise', 'pump', 'moon']):
                sentiment = 'bullish'
                direction = 'upward'
            elif any(word in content for word in ['bearish', 'down', 'fall', 'dump', 'crash']):
                sentiment = 'bearish'
                direction = 'downward'
            else:
                sentiment = 'neutral'
                direction = 'sideways'
            
            # Extract topic/asset
            topic = 'the market'
            for word in content.split():
                if word.upper() in ['BTC', 'ETH', 'ADA', 'SOL', 'BITCOIN', 'ETHEREUM']:
                    topic = word.upper()
                    break
            
            # Replace template variables
            response = template.replace('{sentiment}', sentiment)
            response = response.replace('{direction}', direction)
            response = response.replace('{topic}', topic)
            response = response.replace('{event}', 'recent developments')
            response = response.replace('{level}', f'{random.randint(45000, 55000)}')
            response = response.replace('{target}', f'{random.randint(55000, 65000)}')
            
            return response
            
        except Exception as e:
            logger.warning(f"Error customizing response: {e}")
            return template
    
    def get_delay_minutes(self, delay_type: str) -> int:
        """Get realistic response delay in minutes"""
        delays = {
            'immediate': random.randint(1, 5),
            'quick': random.randint(5, 60),
            'delayed': random.randint(60, 1440),  # up to 24 hours
            'late': random.randint(1440, 10080)   # up to 7 days
        }
        return delays.get(delay_type, 30)
    
    def analyze_response_sentiment(self, response: str) -> str:
        """Analyze sentiment of generated response"""
        positive_words = ['good', 'great', 'bullish', 'up', 'buy', 'opportunity']
        negative_words = ['bad', 'bearish', 'down', 'sell', 'risk', 'dump']
        
        positive_count = sum(1 for word in positive_words if word in response.lower())
        negative_count = sum(1 for word in negative_words if word in response.lower())
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    async def generate_threaded_responses(self, post_id: str):
        """Generate threaded discussion (replies to replies)"""
        try:
            post = next((p for p in self.posts if p['id'] == post_id), None)
            if not post or len(post['replies']) < 3:
                return
            
            # Generate 1-3 threaded responses
            num_threads = random.randint(1, 3)
            
            for _ in range(num_threads):
                # Pick a parent comment
                parent_reply = random.choice(post['replies'])
                
                # Generate response to the parent
                persona = random.choice(self.user_personas)
                thread_templates = [
                    "Interesting point about {topic}",
                    "I disagree with {sentiment} - here's why...",
                    "Can you elaborate on {detail}?",
                    "This reminds me of {comparison}"
                ]
                
                template = random.choice(thread_templates)
                threaded_response = self.customize_response(template, post)
                
                thread_data = {
                    'id': f"thread_{post_id}_{int(time.time())}",
                    'parent_id': parent_reply['id'],
                    'user': f"{persona['type']}_{random.randint(1, 100)}",
                    'text': threaded_response,
                    'timestamp': (datetime.utcnow() + timedelta(minutes=random.randint(10, 120))).isoformat(),
                    'upvotes': random.randint(0, 10),
                    'is_thread': True
                }
                
                post['replies'].append(thread_data)
                
        except Exception as e:
            logger.error(f"Error generating threaded responses: {e}")
    
    async def calculate_engagement_metrics(self, post_id: str):
        """Calculate comprehensive engagement metrics"""
        try:
            post = next((p for p in self.posts if p['id'] == post_id), None)
            if not post:
                return
            
            # Basic metrics
            num_replies = len(post['replies'])
            total_upvotes = sum(reply.get('upvotes', 0) for reply in post['replies'])
            
            # Advanced metrics
            unique_users = len(set(reply['user'] for reply in post['replies']))
            avg_response_length = np.mean([len(reply['text']) for reply in post['replies']]) if post['replies'] else 0
            
            # Engagement score calculation
            engagement_score = (
                num_replies * 0.3 +
                total_upvotes * 0.2 +
                unique_users * 0.3 +
                (avg_response_length / 100) * 0.2
            )
            
            # Viral indicators
            comment_velocity = num_replies / max(1, (datetime.utcnow() - datetime.fromisoformat(post['timestamp'])).total_seconds() / 3600)
            
            post['metrics'].update({
                'views': random.randint(num_replies * 10, num_replies * 50),
                'upvotes': random.randint(max(1, num_replies - 2), num_replies * 3),
                'downvotes': random.randint(0, max(1, num_replies // 3)),
                'comments': num_replies,
                'shares': random.randint(0, max(1, num_replies // 5))
            })
            
            post['engagement_score'] = engagement_score
            post['viral_indicators'].update({
                'comment_velocity': comment_velocity,
                'cross_platform_buzz': comment_velocity > self.engagement_patterns['viral_thresholds']['comment_velocity'],
                'influencer_engagement': any(persona in reply.get('persona', '') for reply in post['replies'] for persona in ['technical_analyst', 'community_builder'])
            })
            
        except Exception as e:
            logger.error(f"Error calculating engagement metrics: {e}")
    
    async def track_engagement(self, post_id: str) -> Dict:
        """Track engagement metrics for a specific post"""
        try:
            post = next((p for p in self.posts if p['id'] == post_id), None)
            if not post:
                return {'error': 'Post not found'}
            
            return {
                'post_id': post_id,
                'metrics': post['metrics'],
                'engagement_score': post['engagement_score'],
                'viral_indicators': post['viral_indicators'],
                'community_sentiment': self.analyze_community_sentiment(post),
                'response_quality': self.assess_response_quality(post),
                'recommendations': self.generate_engagement_recommendations(post)
            }
            
        except Exception as e:
            logger.error(f"Error tracking engagement for {post_id}: {e}")
            return {'error': str(e)}
    
    def analyze_community_sentiment(self, post: Dict) -> Dict:
        """Analyze overall sentiment of community responses"""
        try:
            if not post['replies']:
                return {'overall': 'neutral', 'distribution': {}}
            
            sentiments = [reply.get('sentiment', 'neutral') for reply in post['replies']]
            sentiment_counts = {
                'positive': sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')
            }
            
            total = len(sentiments)
            sentiment_percentages = {k: v/total for k, v in sentiment_counts.items()}
            
            # Determine overall sentiment
            if sentiment_percentages['positive'] > 0.6:
                overall = 'very_positive'
            elif sentiment_percentages['positive'] > 0.4:
                overall = 'positive'
            elif sentiment_percentages['negative'] > 0.6:
                overall = 'very_negative'
            elif sentiment_percentages['negative'] > 0.4:
                overall = 'negative'
            else:
                overall = 'neutral'
            
            return {
                'overall': overall,
                'distribution': sentiment_percentages,
                'confidence': max(sentiment_percentages.values())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing community sentiment: {e}")
            return {'overall': 'neutral', 'distribution': {}}
    
    def assess_response_quality(self, post: Dict) -> Dict:
        """Assess quality of community responses"""
        try:
            if not post['replies']:
                return {'average_quality': 0, 'quality_distribution': {}}
            
            quality_scores = []
            for reply in post['replies']:
                # Simple quality assessment based on length, upvotes, and persona
                length_score = min(1.0, len(reply['text']) / 200)  # Normalize to max 1.0
                upvote_score = min(1.0, reply.get('upvotes', 0) / 20)  # Normalize to max 1.0
                
                # Persona quality weights
                persona_weights = {
                    'technical_analyst': 0.9,
                    'community_builder': 0.8,
                    'news_trader': 0.7,
                    'day_trader': 0.6,
                    'hodler': 0.5
                }
                persona_score = persona_weights.get(reply.get('persona', ''), 0.5)
                
                overall_quality = (length_score + upvote_score + persona_score) / 3
                quality_scores.append(overall_quality)
            
            avg_quality = np.mean(quality_scores)
            
            # Categorize quality levels
            high_quality = sum(1 for q in quality_scores if q > 0.7)
            medium_quality = sum(1 for q in quality_scores if 0.4 <= q <= 0.7)
            low_quality = sum(1 for q in quality_scores if q < 0.4)
            
            total = len(quality_scores)
            quality_distribution = {
                'high': high_quality / total,
                'medium': medium_quality / total,
                'low': low_quality / total
            }
            
            return {
                'average_quality': avg_quality,
                'quality_distribution': quality_distribution,
                'total_responses': total
            }
            
        except Exception as e:
            logger.error(f"Error assessing response quality: {e}")
            return {'average_quality': 0, 'quality_distribution': {}}
    
    def generate_engagement_recommendations(self, post: Dict) -> List[str]:
        """Generate recommendations to improve engagement"""
        recommendations = []
        
        try:
            metrics = post['metrics']
            engagement_score = post['engagement_score']
            
            if engagement_score < 5:
                recommendations.append("Consider posting during peak hours (9-11 UTC, 14-16 UTC, 20-21 UTC)")
                recommendations.append("Add more engaging questions to encourage responses")
            
            if metrics['comments'] < 5:
                recommendations.append("Include controversial or thought-provoking statements")
                recommendations.append("Ask specific questions that require expertise to answer")
            
            if metrics['upvotes'] / max(1, metrics['comments']) < 2:
                recommendations.append("Focus on providing more value in initial post")
                recommendations.append("Include data or charts to support arguments")
            
            viral_indicators = post['viral_indicators']
            if not viral_indicators['cross_platform_buzz']:
                recommendations.append("Consider cross-posting to related communities")
                recommendations.append("Share insights on Twitter/Telegram for broader reach")
            
            if not viral_indicators['influencer_engagement']:
                recommendations.append("Tag relevant community leaders or analysts")
                recommendations.append("Provide unique insights that influencers would want to share")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    async def simulate_viral_potential(self, content: str, target_subreddits: List[str]) -> Dict:
        """Simulate viral potential across multiple subreddits"""
        try:
            viral_analysis = {
                'content': content,
                'subreddits': {},
                'overall_viral_score': 0,
                'cross_platform_potential': False,
                'estimated_reach': 0
            }
            
            total_engagement = 0
            
            for subreddit in target_subreddits:
                # Simulate posting to each subreddit
                result = await self.simulate_post(content, subreddit)
                
                if result['success']:
                    post_id = result['post_id']
                    engagement = await self.track_engagement(post_id)
                    
                    viral_analysis['subreddits'][subreddit] = {
                        'engagement_score': engagement.get('engagement_score', 0),
                        'viral_indicators': engagement.get('viral_indicators', {}),
                        'estimated_views': engagement.get('metrics', {}).get('views', 0)
                    }
                    
                    total_engagement += engagement.get('engagement_score', 0)
            
            # Calculate overall viral potential
            avg_engagement = total_engagement / len(target_subreddits) if target_subreddits else 0
            viral_analysis['overall_viral_score'] = avg_engagement
            viral_analysis['cross_platform_potential'] = avg_engagement > 15
            viral_analysis['estimated_reach'] = sum(
                data.get('estimated_views', 0) 
                for data in viral_analysis['subreddits'].values()
            )
            
            return viral_analysis
            
        except Exception as e:
            logger.error(f"Error simulating viral potential: {e}")
            return {'error': str(e)}
    
    def get_community_insights(self) -> Dict:
        """Get insights about community engagement patterns"""
        try:
            if not self.posts:
                return {'error': 'No posts available for analysis'}
            
            # Aggregate metrics
            total_posts = len(self.posts)
            total_replies = sum(len(post['replies']) for post in self.posts)
            avg_engagement = np.mean([post['engagement_score'] for post in self.posts])
            
            # Time-based analysis
            post_times = [datetime.fromisoformat(post['timestamp']).hour for post in self.posts]
            peak_hours = [hour for hour, count in Counter(post_times).most_common(3)]
            
            # Best performing content analysis
            top_posts = sorted(self.posts, key=lambda x: x['engagement_score'], reverse=True)[:3]
            
            return {
                'overview': {
                    'total_posts': total_posts,
                    'total_community_responses': total_replies,
                    'average_engagement_score': avg_engagement
                },
                'timing_insights': {
                    'peak_engagement_hours': peak_hours,
                    'optimal_posting_recommendation': 'Peak hours show highest engagement'
                },
                'top_performing_posts': [
                    {
                        'content_preview': post['content'][:100] + '...',
                        'engagement_score': post['engagement_score'],
                        'responses': len(post['replies'])
                    }
                    for post in top_posts
                ],
                'engagement_trends': {
                    'average_responses_per_post': total_replies / total_posts if total_posts > 0 else 0,
                    'viral_post_percentage': len([p for p in self.posts if p['viral_indicators']['cross_platform_buzz']]) / total_posts * 100 if total_posts > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting community insights: {e}")
            return {'error': str(e)}

# Helper imports
try:
    import numpy as np
    from collections import Counter
except ImportError:
    # Fallback implementations
    class np:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
    
    def Counter(data):
        counts = {}
        for item in data:
            counts[item] = counts.get(item, 0) + 1
        return counts