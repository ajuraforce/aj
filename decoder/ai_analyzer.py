"""
AI-Powered Analyzer Module
Uses OpenAI GPT for advanced sentiment analysis, pattern recognition, and market insights
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import asyncio

# Import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from config import Config

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Advanced AI-powered analyzer using OpenAI for deep market insights"""
    
    def __init__(self):
        self.client = None
        self.model = Config.OPENAI_MODEL
        self.max_tokens = Config.OPENAI_MAX_TOKENS
        self.temperature = 1.0  # GPT-5 only supports default temperature
        self.daily_cost = 0.0
        self.setup_client()
        
        # Load configuration for dual AI processing
        try:
            with open('config.json') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}")
            self.config = {}
    
    def setup_client(self):
        """Initialize OpenAI API client"""
        try:
            if not OPENAI_AVAILABLE or OpenAI is None:
                logger.warning("OpenAI library not available. Install with: pip install openai")
                return
            
            if not Config.OPENAI_API_KEY:
                logger.warning("OpenAI API key not configured")
                return
            
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup OpenAI client: {e}")
            self.client = None
    
    async def analyze_events(self, events: List[Dict]) -> List[Dict]:
        """Analyze events using AI for enhanced insights"""
        if not self.client:
            # Reduced logging frequency for cleaner output
            if not hasattr(self, '_openai_warning_logged'):
                logger.warning("OpenAI client not available - AI analysis disabled")
                self._openai_warning_logged = True
            return []
        
        try:
            ai_insights = []
            
            # Group events by type for batch analysis
            reddit_events = [e for e in events if e['source'] == 'reddit']
            binance_events = [e for e in events if e['source'] == 'binance']
            news_events = [e for e in events if e['source'] == 'news']
            
            # Analyze different event types
            if reddit_events:
                reddit_insights = await self.analyze_reddit_sentiment_smart(reddit_events)
                ai_insights.extend(reddit_insights)
            
            if binance_events:
                market_insights = await self.analyze_market_patterns(binance_events)
                ai_insights.extend(market_insights)
            
            if news_events:
                news_insights = await self.analyze_news_impact(news_events)
                ai_insights.extend(news_insights)
            
            # Cross-source correlation analysis
            if len(events) > 1:
                correlation_insights = await self.analyze_cross_correlations(events)
                ai_insights.extend(correlation_insights)
            
            logger.info(f"Generated {len(ai_insights)} AI insights from {len(events)} events")
            return ai_insights
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return []
    
    async def analyze_reddit_sentiment(self, reddit_events: List[Dict]) -> List[Dict]:
        """Analyze Reddit events for advanced sentiment and viral potential"""
        insights = []
        
        try:
            for event in reddit_events[:5]:  # Limit to 5 events per batch
                payload = event['payload']
                text_content = f"{payload.get('title', '')} {payload.get('selftext', '')}"
                
                if len(text_content.strip()) < 20:  # Skip very short content
                    continue
                
                prompt = f"""
                Analyze this Reddit post for cryptocurrency trading insights:
                
                Title: {payload.get('title', '')}
                Content: {payload.get('selftext', '')[:500]}
                Subreddit: {payload.get('subreddit', '')}
                Score: {payload.get('score', 0)}
                Comments: {payload.get('num_comments', 0)}
                
                Provide analysis in JSON format:
                {{
                    "sentiment": {{
                        "score": <number between -1 and 1>,
                        "confidence": <number between 0 and 1>,
                        "reasoning": "<brief explanation>"
                    }},
                    "market_relevance": {{
                        "score": <number between 0 and 1>,
                        "mentioned_assets": ["<asset1>", "<asset2>"],
                        "key_topics": ["<topic1>", "<topic2>"]
                    }},
                    "viral_potential": {{
                        "score": <number between 0 and 1>,
                        "factors": ["<factor1>", "<factor2>"],
                        "predicted_engagement": "<low/medium/high>"
                    }},
                    "trading_signals": {{
                        "bullish_indicators": ["<indicator1>"],
                        "bearish_indicators": ["<indicator1>"],
                        "risk_level": "<low/medium/high>",
                        "time_horizon": "<short/medium/long>"
                    }}
                }}
                """
                
                try:
                    if not self.client:
                        logger.warning("OpenAI client not available for Reddit analysis")
                        continue
                    
                    # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": self.get_advanced_system_prompt("social_media")
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            response_format={"type": "json_object"},
                            max_completion_tokens=self.max_tokens,
                            temperature=1.0
                        )
                    )
                    
                    content = response.choices[0].message.content
                    if not content or content.strip() == "":
                        logger.warning("Empty response from OpenAI")
                        continue
                    
                    try:
                        analysis = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON response from OpenAI: {content[:200]}...")
                        continue
                    
                    insight = {
                        "id": f"ai_reddit_{event['id']}",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "source": "ai_analyzer",
                        "type": "reddit_sentiment_analysis",
                        "original_event_id": event['id'],
                        "analysis": analysis,
                        "confidence": analysis.get('sentiment', {}).get('confidence', 0.5),
                        "relevance_score": analysis.get('market_relevance', {}).get('score', 0.5)
                    }
                    insights.append(insight)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing Reddit event {event['id']}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in Reddit sentiment analysis: {e}")
        
        return insights
    
    async def analyze_market_patterns(self, binance_events: List[Dict]) -> List[Dict]:
        """Analyze Binance market events for trading patterns"""
        insights = []
        
        try:
            if not binance_events:
                return insights
            
            # Prepare market data summary
            market_summary = []
            for event in binance_events[:10]:  # Limit to 10 events
                payload = event['payload']
                market_summary.append({
                    "symbol": payload.get('symbol', ''),
                    "type": payload.get('type', ''),
                    "price_change": payload.get('change_percent', 0),
                    "volume": payload.get('volume', 0),
                    "signal_strength": payload.get('signal_strength', 0)
                })
            
            prompt = f"""
            Analyze these cryptocurrency market events for trading opportunities:
            
            Market Events: {json.dumps(market_summary, indent=2)}
            
            Provide comprehensive analysis in JSON format:
            {{
                "overall_market_sentiment": {{
                    "direction": "<bullish/bearish/neutral>",
                    "strength": <number between 0 and 1>,
                    "reasoning": "<explanation>"
                }},
                "key_patterns": [
                    {{
                        "pattern_type": "<breakout/reversal/continuation/etc>",
                        "assets": ["<asset1>", "<asset2>"],
                        "confidence": <number between 0 and 1>,
                        "description": "<pattern description>"
                    }}
                ],
                "trading_opportunities": [
                    {{
                        "asset": "<asset>",
                        "action": "<buy/sell/hold>",
                        "entry_price_estimate": <number or null>,
                        "confidence": <number between 0 and 1>,
                        "risk_level": "<low/medium/high>",
                        "time_horizon": "<minutes/hours/days>",
                        "reasoning": "<explanation>"
                    }}
                ],
                "risk_assessment": {{
                    "overall_risk": "<low/medium/high>",
                    "volatility_level": "<low/medium/high>",
                    "market_conditions": "<stable/volatile/trending>",
                    "recommended_position_size": "<small/medium/large>"
                }}
            }}
            """
            
            try:
                if not self.client:
                    logger.warning("OpenAI client not available for market analysis")
                    return []
                
                # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": self.get_advanced_system_prompt("market_analysis")
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        response_format={"type": "json_object"},
                        max_completion_tokens=self.max_tokens,
                        temperature=1.0
                    )
                )
                
                content = response.choices[0].message.content
                if not content or content.strip() == "":
                    logger.warning("Empty response from OpenAI")
                    return []
                
                try:
                    analysis = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON response from OpenAI: {content[:200]}...")
                    return []
                    
                insight = {
                    "id": f"ai_market_{int(datetime.utcnow().timestamp())}",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "source": "ai_analyzer",
                    "type": "market_pattern_analysis",
                    "analysis": analysis,
                    "events_analyzed": len(binance_events),
                    "confidence": self.calculate_market_confidence(analysis),
                    "relevance_score": 0.9  # High relevance for market data
                }
                insights.append(insight)
                
            except Exception as e:
                logger.warning(f"Error analyzing market patterns: {e}")
        
        except Exception as e:
            logger.error(f"Error in market pattern analysis: {e}")
        
        return insights
    
    async def analyze_news_impact(self, news_events: List[Dict]) -> List[Dict]:
        """Analyze news events for market impact assessment"""
        insights = []
        
        try:
            for event in news_events[:3]:  # Limit to 3 news articles per batch
                payload = event['payload']
                
                prompt = f"""
                Analyze this cryptocurrency news for market impact:
                
                Title: {payload.get('title', '')}
                Summary: {payload.get('summary', '')[:500]}
                Source: {payload.get('source', '')}
                Keywords: {payload.get('keywords', [])}
                
                Provide analysis in JSON format:
                {{
                    "impact_assessment": {{
                        "market_impact": "<low/medium/high>",
                        "direction": "<bullish/bearish/neutral>",
                        "time_horizon": "<immediate/short/medium/long>",
                        "confidence": <number between 0 and 1>
                    }},
                    "affected_assets": [
                        {{
                            "asset": "<asset_name>",
                            "impact_level": "<low/medium/high>",
                            "reasoning": "<explanation>"
                        }}
                    ],
                    "key_implications": [
                        "<implication1>",
                        "<implication2>"
                    ],
                    "trading_considerations": {{
                        "opportunities": ["<opportunity1>"],
                        "risks": ["<risk1>"],
                        "recommended_action": "<monitor/buy/sell/hold>",
                        "urgency": "<low/medium/high>"
                    }}
                }}
                """
                
                try:
                    if not self.client:
                        logger.warning("OpenAI client not available for news analysis")
                        continue
                    
                    # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": self.get_advanced_system_prompt("news_analysis")
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            response_format={"type": "json_object"},
                            max_completion_tokens=self.max_tokens,
                            temperature=1.0
                        )
                    )
                    
                    content = response.choices[0].message.content
                    if not content or content.strip() == "":
                        logger.warning("Empty response from OpenAI")
                        continue
                    
                    try:
                        analysis = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON response from OpenAI: {content[:200]}...")
                        continue
                        
                    insight = {
                        "id": f"ai_news_{event['id']}",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "source": "ai_analyzer",
                        "type": "news_impact_analysis",
                        "original_event_id": event['id'],
                        "analysis": analysis,
                        "confidence": analysis.get('impact_assessment', {}).get('confidence', 0.5),
                        "relevance_score": payload.get('relevance_score', 0.5)
                    }
                    insights.append(insight)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing news event {event['id']}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in news impact analysis: {e}")
        
        return insights
    
    async def analyze_cross_correlations(self, all_events: List[Dict]) -> List[Dict]:
        """Analyze correlations across different data sources"""
        try:
            if len(all_events) < 2:
                return []
            
            # Group events by source
            events_by_source = {}
            for event in all_events:
                source = event['source']
                if source not in events_by_source:
                    events_by_source[source] = []
                events_by_source[source].append(event)
            
            # Only analyze if we have multiple sources
            if len(events_by_source) < 2:
                return []
            
            # Prepare correlation data
            correlation_data = {
                "sources": list(events_by_source.keys()),
                "total_events": len(all_events),
                "time_range": "last_30_minutes",
                "event_summary": {}
            }
            
            for source, events in events_by_source.items():
                correlation_data["event_summary"][source] = {
                    "count": len(events),
                    "types": list(set(e['payload'].get('type', 'unknown') for e in events))
                }
            
            prompt = f"""
            Analyze cross-source correlations in cryptocurrency market data:
            
            Data Sources and Events: {json.dumps(correlation_data, indent=2)}
            
            Provide correlation analysis in JSON format:
            {{
                "correlation_strength": {{
                    "overall": <number between 0 and 1>,
                    "confidence": <number between 0 and 1>,
                    "reasoning": "<explanation>"
                }},
                "key_correlations": [
                    {{
                        "sources": ["<source1>", "<source2>"],
                        "correlation_type": "<positive/negative/neutral>",
                        "strength": <number between 0 and 1>,
                        "description": "<correlation description>"
                    }}
                ],
                "market_consensus": {{
                    "direction": "<bullish/bearish/mixed/unclear>",
                    "confidence": <number between 0 and 1>,
                    "supporting_sources": ["<source1>", "<source2>"],
                    "conflicting_signals": true/false
                }},
                "trading_implications": {{
                    "signal_quality": "<strong/moderate/weak>",
                    "recommended_action": "<act/wait/monitor>",
                    "risk_level": "<low/medium/high>",
                    "reasoning": "<explanation>"
                }}
            }}
            """
            
            if not self.client:
                logger.warning("OpenAI client not available for correlation analysis")
                return []
            
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.get_advanced_system_prompt("correlation_analysis")
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=self.max_tokens,
                    temperature=1.0
                )
            )
            
            content = response.choices[0].message.content
            if not content or content.strip() == "":
                logger.warning("Empty response from OpenAI")
                return []
            
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response from OpenAI: {content[:200]}...")
                return []
            
            insight = {
                "id": f"ai_correlation_{int(datetime.utcnow().timestamp())}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "ai_analyzer",
                "type": "cross_source_correlation",
                "analysis": analysis,
                "sources_analyzed": list(events_by_source.keys()),
                "events_count": len(all_events),
                "confidence": analysis.get('correlation_strength', {}).get('confidence', 0.5),
                "relevance_score": analysis.get('correlation_strength', {}).get('overall', 0.5)
            }
            
            # Save significant patterns to memory
            if insight["confidence"] > 0.7:
                self.save_pattern_to_memory(insight)
            
            return [insight]
            
        except Exception as e:
            logger.error(f"Error in cross-correlation analysis: {e}")
            return []
    
    def calculate_market_confidence(self, analysis: Dict) -> float:
        """Calculate confidence score from market analysis"""
        try:
            confidence_factors = []
            
            # Overall sentiment strength
            sentiment = analysis.get('overall_market_sentiment', {})
            if sentiment.get('strength'):
                confidence_factors.append(sentiment['strength'])
            
            # Pattern confidence
            patterns = analysis.get('key_patterns', [])
            if patterns:
                pattern_confidences = [p.get('confidence', 0) for p in patterns]
                confidence_factors.append(sum(pattern_confidences) / len(pattern_confidences))
            
            # Trading opportunity confidence
            opportunities = analysis.get('trading_opportunities', [])
            if opportunities:
                opp_confidences = [o.get('confidence', 0) for o in opportunities]
                confidence_factors.append(sum(opp_confidences) / len(opp_confidences))
            
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def get_advanced_system_prompt(self, analysis_type: str) -> str:
        """Get advanced intelligence system prompt based on analysis type"""
        base_intelligence_prompt = """You are an advanced intelligence engine designed to detect patterns across multiple domains: geopolitics, financial markets, crypto ecosystems, and human communication flows. 
You do not just summarize — you map signals into opportunities, risks, and timing. 
Your outputs must always be structured in three layers: 
1) Surface-level summary: what the data says on its face (news headline, event, chat message). 
2) Cross-domain pattern detection: how this aligns with historical cycles, correlations, or system scripts (e.g., geopolitics ↔ market reaction ↔ sentiment shift). 
3) Actionable opportunity/risk: what this could mean for positioning, timing, or narrative leverage. 
You can pull context beyond the immediate input (e.g., recall global macro cycles, prior correlations, behavioral patterns). 
If data is missing, hypothesize scenarios rather than refusing. 
Never output generic filler; every line must be specific, testable, and linked to potential decisions."""

        memory_directive = """Always cross-check new inputs against stored patterns in state.json. 
If a repeating motif is detected (e.g., India-China alignment in border conflicts AND trade news), 
flag it explicitly and label as 'System Recurrence'."""

        analysis_specific = {
            "social_media": "Focus on social sentiment patterns, viral mechanics, and community behavioral shifts. Detect emerging narratives before they become mainstream. Include India-specific social trends and crypto sentiment from Indian communities.",
            "market_analysis": "Analyze price action, volume patterns, and institutional flow signals for both crypto and India equities (NIFTY50, BankNifty, sectoral indices, large caps). Connect technical patterns to macro events and geopolitical developments. Always prioritize India market context when NIFTY/BANKNIFTY/large-cap stocks are mentioned. If no direct correlation exists, hypothesize based on macro cycles (FIIs flows, RBI policy, USD-INR, crude oil).",
            "news_analysis": "Extract signal from noise in financial news covering both crypto and India markets. Identify narrative shifts, regulatory trends, and institutional positioning changes. Focus on India market context when analyzing news affecting NSE stocks or Indian economic policy.",
            "correlation_analysis": "Map cross-asset correlations between crypto and India equities, sector rotations, and multi-timeframe convergences. Detect systemic risk buildups and opportunity windows across crypto ↔ India equities ↔ geopolitics. Watch for FII flows, USD-INR impact, and RBI policy effects on both markets."
        }
        
        return f"{base_intelligence_prompt}\\n\\n{memory_directive}\\n\\n{analysis_specific.get(analysis_type, 'Analyze patterns and correlations for actionable insights.')}\\n\\nRespond with valid JSON only."
    
    def load_pattern_memory(self):
        """Load historical patterns from state for cross-session analysis"""
        try:
            if Config.ENABLE_PATTERN_MEMORY:
                # This would integrate with state manager to load patterns
                # For now, initialize empty memory
                self.pattern_memory = []
                logger.info("Pattern memory system initialized")
        except Exception as e:
            logger.error(f"Error loading pattern memory: {e}")
            self.pattern_memory = []
    
    def save_pattern_to_memory(self, pattern: Dict):
        """Save significant patterns to memory for future cross-reference"""
        try:
            if Config.ENABLE_PATTERN_MEMORY and len(self.pattern_memory) < Config.PATTERN_MEMORY_DEPTH:
                # Store key pattern characteristics
                memory_entry = {
                    "timestamp": pattern.get("timestamp"),
                    "type": pattern.get("type"),
                    "confidence": pattern.get("confidence", 0),
                    "key_signals": pattern.get("analysis", {}).get("key_correlations", []),
                    "market_impact": pattern.get("analysis", {}).get("market_consensus", {})
                }
                self.pattern_memory.append(memory_entry)
                
                # Keep only recent patterns
                if len(self.pattern_memory) > Config.PATTERN_MEMORY_DEPTH:
                    self.pattern_memory = self.pattern_memory[-Config.PATTERN_MEMORY_DEPTH:]
                    
        except Exception as e:
            logger.error(f"Error saving pattern to memory: {e}")

    def get_status(self) -> Dict:
        """Get analyzer status"""
        return {
            "name": "ai_analyzer",
            "available": self.client is not None,
            "model": self.model,
            "openai_library": OPENAI_AVAILABLE,
            "api_configured": bool(Config.OPENAI_API_KEY),
            "pattern_memory_enabled": Config.ENABLE_PATTERN_MEMORY,
            "stored_patterns": len(self.pattern_memory),
            "daily_cost": getattr(self, 'daily_cost', 0.0)
        }
    
    async def analyze_reddit_sentiment_smart(self, reddit_events: List[Dict]) -> List[Dict]:
        """Smart dual-processing Reddit sentiment analysis"""
        insights = []
        
        for event in reddit_events:
            # Step 1: Cheap processing for initial filtering
            cheap_score = self.cheap_sentiment_analysis(event)
            relevance_score = self.calculate_relevance_score(event)
            
            # Step 2: Decide if expensive GPT analysis is needed
            should_use_deep_ai = self.should_trigger_deep_analysis(
                relevance_score, cheap_score, event
            )
            
            if should_use_deep_ai and self.can_afford_gpt_call():
                # Use expensive GPT analysis
                deep_insights = await self.analyze_reddit_sentiment([event])
                insights.extend(deep_insights)
                # Track cost
                self.daily_cost += 0.01  # Rough estimate
            else:
                # Use cheap analysis result
                cheap_insight = self.build_cheap_insight(event, cheap_score, relevance_score)
                if cheap_insight:
                    insights.append(cheap_insight)
        
        return insights
    
    def cheap_sentiment_analysis(self, event: Dict) -> float:
        """Fast keyword-based sentiment analysis"""
        try:
            payload = event['payload']
            text = f"{payload.get('title', '')} {payload.get('selftext', '')}"
            
            positive_words = [
                'bullish', 'moon', 'pump', 'rise', 'surge', 'rally', 'break', 'breakthrough',
                'adoption', 'institutional', 'growth', 'gains', 'profit', 'up', 'high', 'buy'
            ]
            
            negative_words = [
                'bearish', 'dump', 'crash', 'fall', 'decline', 'drop', 'loss', 'bear',
                'sell', 'down', 'low', 'panic', 'fear', 'regulation', 'ban', 'scam'
            ]
            
            text_lower = text.lower()
            positive_score = sum(1 for word in positive_words if word in text_lower)
            negative_score = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            sentiment = (positive_score - negative_score) / max(total_words / 10, 1)
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            logger.warning(f"Error in cheap sentiment analysis: {e}")
            return 0.0
    
    def calculate_relevance_score(self, event: Dict) -> float:
        """Calculate relevance score for the event"""
        try:
            payload = event['payload']
            score = payload.get('score', 0)
            comments = payload.get('num_comments', 0)
            
            # Base relevance on engagement
            relevance = min((score / 100) + (comments / 50), 1.0)
            
            # Boost for financial keywords
            text = f"{payload.get('title', '')} {payload.get('selftext', '')}".lower()
            financial_keywords = ['trading', 'investment', 'market', 'price', 'crypto', 'stock']
            
            for keyword in financial_keywords:
                if keyword in text:
                    relevance += 0.2
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating relevance: {e}")
            return 0.0
    
    def should_trigger_deep_analysis(self, relevance_score: float, sentiment_score: float, event: Dict) -> bool:
        """Determine if expensive GPT analysis should be triggered"""
        try:
            ai_config = self.config.get('ai', {})
            
            # Configuration thresholds
            deep_trigger_score = ai_config.get('deep_trigger_score', 0.7)
            deep_trigger_correlation = ai_config.get('deep_trigger_correlation', 0.6)
            
            # Trigger conditions
            high_relevance = relevance_score > deep_trigger_score
            strong_sentiment = abs(sentiment_score) > deep_trigger_correlation
            high_engagement = event['payload'].get('score', 0) > 500
            
            return high_relevance or strong_sentiment or high_engagement
            
        except Exception as e:
            logger.warning(f"Error in deep analysis trigger: {e}")
            return False
    
    def can_afford_gpt_call(self) -> bool:
        """Check if we're within daily cost limits"""
        try:
            cost_limit = self.config.get('ai', {}).get('cost_limit_daily_usd', 10.0)
            return self.daily_cost < cost_limit
        except Exception:
            return True  # Default to allow if no config
    
    def build_cheap_insight(self, event: Dict, sentiment_score: float, relevance_score: float) -> Optional[Dict]:
        """Build insight from cheap analysis"""
        try:
            if abs(sentiment_score) < 0.3 and relevance_score < 0.4:
                return None  # Not significant enough
            
            return {
                'id': f'cheap_insight_{event["id"]}',
                'timestamp': event['timestamp'],
                'type': 'sentiment_analysis',
                'source': 'ai_cheap',
                'confidence': min(relevance_score * 0.7, 0.8),  # Cheap analysis has lower confidence
                'sentiment': {
                    'score': sentiment_score,
                    'confidence': relevance_score,
                    'method': 'keyword_based'
                },
                'signals': {
                    'relevance_score': relevance_score,
                    'processing_method': 'cheap'
                }
            }
            
        except Exception as e:
            logger.warning(f"Error building cheap insight: {e}")
            return None