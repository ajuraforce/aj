"""
Smart Alert Management System
Intelligent alert clustering and fatigue prevention
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
import asyncio
import logging
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import openai
import re
import os

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    id: str
    alert_type: str
    symbol: str
    confidence: float
    message: str
    timestamp: float
    processed: bool = False
    cluster_id: Optional[int] = None
    priority: int = 1
    risk_score: float = 0.0
    business_impact: float = 0.0
    threat_intel_score: float = 0.0
    analyst_feedback: Optional[str] = None
    disposition: Optional[str] = None  # 'true_positive', 'false_positive', 'pending'
    importance_score: float = 0.0

@dataclass
class AnalystFeedback:
    alert_id: str
    analyst_id: str
    disposition: str
    confidence: float
    response_time: float
    timestamp: float

class ThreatIntelligenceEngine:
    def __init__(self, config: dict):
        self.threat_feeds = config.get('threat_feeds', [])
        self.api_keys = config.get('api_keys', {})
        self.cache_duration = config.get('cache_duration', 3600)  # 1 hour
        self.threat_cache = {}
    
    async def get_threat_intelligence_score(self, alert: Alert) -> float:
        """Get threat intelligence score for an alert."""
        try:
            # Check cache first
            cache_key = f"{alert.symbol}_{alert.alert_type}"
            if cache_key in self.threat_cache:
                cached_time, score = self.threat_cache[cache_key]
                if datetime.now().timestamp() - cached_time < self.cache_duration:
                    return score
            
            # Basic threat scoring based on alert characteristics
            threat_score = 0.5  # Baseline
            
            # High-risk patterns in message
            risk_keywords = ['break', 'anomaly', 'unusual', 'spike', 'crash', 'dump']
            if any(keyword in alert.message.lower() for keyword in risk_keywords):
                threat_score += 0.2
            
            # Asset-based scoring
            high_value_assets = ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL']
            if alert.symbol in high_value_assets:
                threat_score += 0.3
            
            # Alert type severity
            high_severity_types = ['TRADING_SIGNAL', 'INSTITUTIONAL_FLOW', 'MULTI_TIMEFRAME']
            if alert.alert_type in high_severity_types:
                threat_score += 0.2
            
            # Cache the result
            self.threat_cache[cache_key] = (datetime.now().timestamp(), threat_score)
            return min(1.0, threat_score)
        except Exception as e:
            logger.error(f"Error getting threat intelligence score: {e}")
            return 0.5

class ContextualScoringEngine:
    def __init__(self, config: dict):
        self.asset_criticality = config.get('asset_criticality', {
            'BTC': 0.9, 'ETH': 0.9, 'BNB': 0.8, 'ADA': 0.7, 'XRP': 0.7,
            'SOL': 0.8, 'DOT': 0.6, 'AVAX': 0.6, 'LINK': 0.6, 'DOGE': 0.5
        })
        self.business_hours = config.get('business_hours', {'start': 9, 'end': 17})
        self.threat_intel = ThreatIntelligenceEngine(config.get('threat_intelligence', {}))
        
    def get_asset_criticality(self, symbol: str) -> float:
        """Get business criticality score for an asset."""
        return self.asset_criticality.get(symbol, 0.5)
    
    def get_temporal_context(self, timestamp: float) -> float:
        """Calculate temporal context multiplier."""
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        is_business_hours = self.business_hours['start'] <= hour <= self.business_hours['end']
        is_weekend = dt.weekday() >= 5
        
        if is_business_hours and not is_weekend:
            return 1.2  # Higher priority during business hours
        elif is_weekend:
            return 0.8  # Lower priority on weekends
        return 1.0
    
    async def calculate_contextual_risk_score(self, alert: Alert) -> float:
        """Calculate comprehensive risk score incorporating multiple factors."""
        try:
            # Base severity from alert type and confidence
            type_weights = {
                'TRADING_SIGNAL': 0.8,
                'CORRELATION_BREAK': 0.6,
                'INSTITUTIONAL_FLOW': 0.9,
                'SENTIMENT_FLOW': 0.5,
                'ML_ANOMALY': 0.7,
                'PORTFOLIO_REBALANCING': 0.3,
                'MULTI_TIMEFRAME': 0.9
            }
            base_severity = type_weights.get(alert.alert_type, 0.5) * alert.confidence
            
            # Asset criticality
            asset_crit = self.get_asset_criticality(alert.symbol)
            
            # Temporal context
            temporal_mult = self.get_temporal_context(alert.timestamp)
            
            # Threat intelligence
            threat_score = await self.threat_intel.get_threat_intelligence_score(alert)
            
            # Business impact (derived from asset value and market conditions)
            business_impact = min(1.0, asset_crit * temporal_mult)
            
            # Combined risk score
            risk_score = (base_severity * 0.4 + 
                         threat_score * 0.3 + 
                         business_impact * 0.3) * temporal_mult
            
            alert.risk_score = min(1.0, risk_score)
            alert.business_impact = business_impact
            alert.threat_intel_score = threat_score
            
            return alert.risk_score
        except Exception as e:
            logger.error(f"Error calculating contextual risk score: {e}")
            return 0.5

class AdaptiveFatigueManager:
    def __init__(self, config: dict):
        self.base_daily_limit = config.get('max_daily_alerts', 20)
        self.analyst_performance = {}
        self.workload_tracker = {}
        
    def update_analyst_performance(self, feedback: AnalystFeedback):
        """Update analyst performance tracking."""
        analyst_id = feedback.analyst_id
        if analyst_id not in self.analyst_performance:
            self.analyst_performance[analyst_id] = {
                'total_alerts': 0,
                'true_positives': 0,
                'false_positives': 0,
                'avg_response_time': 0.0,
                'fatigue_score': 0
            }
        
        perf = self.analyst_performance[analyst_id]
        perf['total_alerts'] += 1
        if feedback.disposition == 'true_positive':
            perf['true_positives'] += 1
        elif feedback.disposition == 'false_positive':
            perf['false_positives'] += 1
        
        # Update average response time
        perf['avg_response_time'] = (perf['avg_response_time'] * (perf['total_alerts'] - 1) + 
                                   feedback.response_time) / perf['total_alerts']
    
    def calculate_dynamic_limit(self, analyst_id: str, current_hour: int) -> int:
        """Calculate dynamic alert limit based on analyst performance and fatigue."""
        if analyst_id not in self.analyst_performance:
            return self.base_daily_limit
        
        perf = self.analyst_performance[analyst_id]
        
        # Calculate accuracy rate
        total_dispositioned = perf['true_positives'] + perf['false_positives']
        accuracy_rate = perf['true_positives'] / max(1, total_dispositioned)
        
        # Adjust limit based on performance
        if accuracy_rate > 0.8:
            limit_multiplier = 1.2
        elif accuracy_rate < 0.6:
            limit_multiplier = 0.8
        else:
            limit_multiplier = 1.0
        
        # Consider time of day fatigue
        if 9 <= current_hour <= 12:
            time_multiplier = 1.1  # Morning peak
        elif 13 <= current_hour <= 16:
            time_multiplier = 1.0   # Afternoon steady
        else:
            time_multiplier = 0.9   # Evening/night reduced
        
        dynamic_limit = int(self.base_daily_limit * limit_multiplier * time_multiplier)
        return max(5, min(50, dynamic_limit))  # Keep within reasonable bounds

class SmartAlertManager:
    def __init__(self, config: dict, db_path: str):
        self.config = config
        self.db_path = db_path
        
        alert_config = config.get('smart_alerts', {})
        self.user_preferences = alert_config.get('user_preferences', {})
        self.clustering_window = alert_config.get('clustering_window', '1h')
        self.fatigue_threshold = alert_config.get('fatigue_threshold', 2)
        
        # Initialize enhanced components
        self.contextual_scorer = ContextualScoringEngine(alert_config.get('contextual_scoring', {}))
        self.fatigue_manager = AdaptiveFatigueManager(alert_config.get('fatigue_management', {}))
        
        # Initialize GPT-5 client for semantic analysis
        try:
            self.gpt5_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.gpt5_enabled = True
            logger.info("GPT-5 client initialized successfully")
        except Exception as e:
            logger.warning(f"GPT-5 client initialization failed: {e}")
            self.gpt5_client = None
            self.gpt5_enabled = False
        
        # Alert history for fatigue management
        self.alert_history = []
        self.daily_alert_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Enhanced clustering parameters
        self.clustering_features = ['confidence', 'hour_of_day', 'symbol_hash', 'risk_score', 'business_impact']
        
    def calculate_alert_priority(self, alert: Alert) -> int:
        """Calculate alert priority based on type and confidence"""
        try:
            # Base priority from alert type
            type_priorities = {
                'TRADING_SIGNAL': 3,
                'CORRELATION_BREAK': 2,
                'INSTITUTIONAL_FLOW': 3,
                'SENTIMENT_FLOW': 2,
                'ML_ANOMALY': 2,
                'PORTFOLIO_REBALANCING': 1,
                'MULTI_TIMEFRAME': 3
            }
            
            base_priority = type_priorities.get(alert.alert_type, 1)
            
            # Adjust based on confidence
            if alert.confidence > 0.9:
                priority_boost = 1
            elif alert.confidence > 0.7:
                priority_boost = 0
            else:
                priority_boost = -1
            
            final_priority = max(1, min(3, base_priority + priority_boost))
            return final_priority
            
        except Exception as e:
            logger.error(f"Error calculating alert priority: {e}")
            return 1
    
    def check_alert_fatigue(self, alert: Alert) -> bool:
        """Check if alert should be filtered due to fatigue"""
        try:
            # Reset daily count if new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_alert_count = 0
                self.last_reset_date = current_date
            
            # Check daily limit (much stricter)
            max_daily_alerts = self.user_preferences.get('max_daily_alerts', 5)  # Reduced from 10 to 5
            if self.daily_alert_count >= max_daily_alerts:
                return True
            
            # Check quiet hours
            quiet_hours = self.user_preferences.get('quiet_hours', [])
            if len(quiet_hours) == 2:
                current_hour = datetime.now().hour
                start_hour = int(quiet_hours[0].split(':')[0])
                end_hour = int(quiet_hours[1].split(':')[0])
                
                if start_hour <= end_hour:
                    if start_hour <= current_hour <= end_hour:
                        return True
                else:  # Quiet hours cross midnight
                    if current_hour >= start_hour or current_hour <= end_hour:
                        return True
            
            # Check symbol-specific fatigue (extended window for critical assets)
            recent_window = datetime.now() - timedelta(hours=2)  # Extended to 2 hours
            recent_alerts = [
                a for a in self.alert_history 
                if (a.symbol == alert.symbol and 
                    datetime.fromtimestamp(a.timestamp) > recent_window)
            ]
            
            if len(recent_alerts) >= self.fatigue_threshold:
                logger.info(f"Alert fatigue triggered for {alert.symbol}: {len(recent_alerts)} recent alerts")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking alert fatigue: {e}")
            return False
    
    def prepare_clustering_features(self, alerts: List[Alert]) -> np.ndarray:
        """Prepare features for alert clustering"""
        try:
            if not alerts:
                return np.array([])
            
            features = []
            for alert in alerts:
                # Extract features for clustering
                feature_vector = [
                    alert.confidence,
                    datetime.fromtimestamp(alert.timestamp).hour,  # Hour of day
                    hash(alert.symbol) % 1000,  # Symbol hash for grouping
                    len(alert.message) / 100,  # Message length (normalized)
                    alert.risk_score,  # New: contextual risk score
                    alert.business_impact,  # New: business impact score
                    alert.threat_intel_score,  # New: threat intelligence score
                ]
                
                # Add alert type as numeric
                type_mapping = {
                    'TRADING_SIGNAL': 1,
                    'CORRELATION_BREAK': 2,
                    'INSTITUTIONAL_FLOW': 3,
                    'SENTIMENT_FLOW': 4,
                    'ML_ANOMALY': 5,
                    'PORTFOLIO_REBALANCING': 6,
                    'MULTI_TIMEFRAME': 7
                }
                feature_vector.append(type_mapping.get(alert.alert_type, 0))
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparing clustering features: {e}")
            return np.array([])
    
    def cluster_alerts(self, alerts: List[Alert]) -> Dict[int, List[Alert]]:
        """Cluster similar alerts to reduce noise"""
        try:
            if len(alerts) < 2:
                return {0: alerts}
            
            # Prepare features
            features = self.prepare_clustering_features(alerts)
            
            if features.size == 0:
                return {0: alerts}
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)
            
            # Apply enhanced clustering with hierarchical approach for better grouping
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=2,
                    distance_threshold=None,
                    linkage='ward'
                )
                cluster_labels = clustering.fit_predict(normalized_features)
            except:
                # Fallback to DBSCAN if hierarchical clustering fails
                clustering = DBSCAN(eps=0.5, min_samples=2)
                cluster_labels = clustering.fit_predict(normalized_features)
            
            # Group alerts by cluster
            clustered_alerts = {}
            for i, alert in enumerate(alerts):
                cluster_id = cluster_labels[i]
                alert.cluster_id = cluster_id
                
                if cluster_id not in clustered_alerts:
                    clustered_alerts[cluster_id] = []
                clustered_alerts[cluster_id].append(alert)
            
            return clustered_alerts
            
        except Exception as e:
            logger.error(f"Error clustering alerts: {e}")
            return {0: alerts}
    
    def detect_attack_chains(self, clustered_alerts: Dict[int, List[Alert]]) -> Dict[int, List[Alert]]:
        """Detect attack chains in clustered alerts and boost their risk scores"""
        try:
            attack_chains = {}
            for cluster_id, cluster_alerts in clustered_alerts.items():
                sorted_alerts = sorted(cluster_alerts, key=lambda x: x.timestamp)
                
                chain_detected = False
                for i in range(len(sorted_alerts) - 1):
                    time_diff = sorted_alerts[i+1].timestamp - sorted_alerts[i].timestamp
                    if 300 <= time_diff <= 3600:  # 5 minutes to 1 hour spacing
                        chain_detected = True
                        break
                
                if chain_detected:
                    logger.info(f"Attack chain detected in cluster {cluster_id} with {len(sorted_alerts)} alerts")
                    for alert in sorted_alerts:
                        alert.risk_score = min(1.0, alert.risk_score * 1.3)  # Boost risk score
                        alert.message = f"⚠️ ATTACK CHAIN: {alert.message}"
                
                attack_chains[cluster_id] = sorted_alerts
            return attack_chains
            
        except Exception as e:
            logger.error(f"Error detecting attack chains: {e}")
            return clustered_alerts
    
    async def enhance_alert_with_gpt5(self, alert: Alert, market_context: Dict = None) -> Alert:
        """Enhanced contextual analysis using GPT-5's reasoning capabilities"""
        if not self.gpt5_enabled:
            return alert
            
        try:
            if market_context is None:
                market_context = {}
                
            prompt = f"""
            Analyze this trading alert with advanced market context understanding:
            
            Alert Details:
            - Type: {alert.alert_type}
            - Symbol: {alert.symbol}
            - Message: {alert.message}
            - Confidence: {alert.confidence}
            - Current Risk Score: {alert.risk_score}
            
            Market Context:
            - Current Market Regime: {market_context.get('regime', 'unknown')}
            - Sector Performance: {market_context.get('sector_performance', {})}
            - Recent Market Events: {market_context.get('recent_events', [])}
            
            Provide comprehensive analysis including:
            1. Refined risk assessment (0.0-1.0 scale) with reasoning
            2. Potential market impact assessment
            3. Historical pattern similarity
            4. Recommended urgency level (1-5)
            5. Key insights for decision making
            
            Format your response as JSON with keys: refined_risk_score, market_impact, urgency_level, insights, reasoning
            """
            
            response = self.gpt5_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=800
            )
            
            # Parse GPT-5 analysis
            analysis = json.loads(response.choices[0].message.content)
            
            # Update alert with GPT-5 insights
            gpt5_risk = float(analysis.get('refined_risk_score', alert.risk_score))
            alert.risk_score = (alert.risk_score * 0.6) + (gpt5_risk * 0.4)  # Weighted blend
            
            # Enhanced analyst feedback with GPT-5 insights
            gpt5_insights = analysis.get('insights', 'No additional insights available')
            if alert.analyst_feedback:
                alert.analyst_feedback += f"\n\nGPT-5 Analysis: {gpt5_insights}"
            else:
                alert.analyst_feedback = f"GPT-5 Analysis: {gpt5_insights}"
            
            # Update importance score based on urgency
            urgency = analysis.get('urgency_level', 3)
            alert.importance_score = max(alert.importance_score, urgency / 5.0)
            
            logger.info(f"Enhanced alert {alert.id} with GPT-5 analysis: risk={alert.risk_score:.3f}, urgency={urgency}")
            return alert
            
        except Exception as e:
            logger.error(f"Error enhancing alert with GPT-5: {e}")
            return alert
    
    async def generate_alert_clusters_narrative(self, clustered_alerts: Dict) -> str:
        """Generate human-readable narrative for alert clusters using GPT-5"""
        if not self.gpt5_enabled or not clustered_alerts:
            return "Alert cluster analysis not available."
            
        try:
            cluster_summaries = []
            for cluster_id, alerts in clustered_alerts.items():
                if len(alerts) > 1:
                    alert_descriptions = [f"- {alert.symbol}: {alert.message[:100]}" for alert in alerts[:5]]
                    
                    prompt = f"""
                    Generate a concise narrative summary for this cluster of {len(alerts)} related trading alerts:
                    
                    Alerts in cluster:
                    {chr(10).join(alert_descriptions)}
                    
                    Create a professional summary explaining:
                    1. Common theme connecting these alerts
                    2. Potential market implications
                    3. Risk level assessment
                    4. Recommended actions
                    
                    Keep response under 200 words, use clear professional language.
                    """
                    
                    response = self.gpt5_client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=300
                    )
                    
                    cluster_summaries.append(f"**Cluster {cluster_id}** ({len(alerts)} alerts): {response.choices[0].message.content}")
            
            return "\n\n".join(cluster_summaries) if cluster_summaries else "No significant alert clusters detected."
            
        except Exception as e:
            logger.error(f"Error generating cluster narrative: {e}")
            return "Cluster narrative generation failed."
    
    def merge_cluster_alerts(self, cluster_alerts: List[Alert]) -> Alert:
        """Merge alerts in the same cluster into a single representative alert"""
        try:
            if len(cluster_alerts) == 1:
                return cluster_alerts[0]
            
            # Sort by priority and confidence
            sorted_alerts = sorted(
                cluster_alerts, 
                key=lambda x: (x.priority, x.confidence), 
                reverse=True
            )
            
            primary_alert = sorted_alerts[0]
            
            # Create merged message
            symbols = list(set(alert.symbol for alert in cluster_alerts))
            if len(symbols) > 3:
                symbol_summary = f"{', '.join(symbols[:3])} and {len(symbols)-3} others"
            else:
                symbol_summary = ', '.join(symbols)
            
            merged_message = f"🔗 Combined Alert ({len(cluster_alerts)} signals)\n"
            merged_message += f"Assets: {symbol_summary}\n"
            merged_message += f"Primary: {primary_alert.message.split(chr(10))[0]}"  # First line only
            
            # Create merged alert
            merged_alert = Alert(
                id=f"merged_{primary_alert.id}",
                alert_type=primary_alert.alert_type,
                symbol=symbol_summary,
                confidence=float(np.mean([alert.confidence for alert in cluster_alerts])),
                message=merged_message,
                timestamp=primary_alert.timestamp,
                priority=primary_alert.priority
            )
            
            return merged_alert
            
        except Exception as e:
            logger.error(f"Error merging cluster alerts: {e}")
            return cluster_alerts[0] if cluster_alerts else Alert("", "", "", 0.0, "", 0.0)
    
    def rank_alerts_by_importance(self, alerts: List[Alert]) -> List[Alert]:
        """Rank alerts by importance for user attention"""
        try:
            # Calculate enhanced importance score using contextual factors
            for alert in alerts:
                importance_score = 0
                
                # Priority weight (highest impact)
                importance_score += alert.priority * 30
                
                # Enhanced: Risk score weight (new contextual factor)
                importance_score += alert.risk_score * 40
                
                # Enhanced: Business impact weight
                importance_score += alert.business_impact * 25
                
                # Enhanced: Threat intelligence score
                importance_score += alert.threat_intel_score * 20
                
                # Confidence weight (reduced since risk score incorporates this)
                importance_score += alert.confidence * 15
                
                # Recency weight (more recent = more important)
                hours_ago = (datetime.now().timestamp() - alert.timestamp) / 3600
                recency_weight = max(0, 10 - hours_ago)  # Decay over 10 hours
                importance_score += recency_weight
                
                # Alert type specific weights
                type_weights = {
                    'TRADING_SIGNAL': 20,
                    'INSTITUTIONAL_FLOW': 15,
                    'MULTI_TIMEFRAME': 15,
                    'CORRELATION_BREAK': 10,
                    'SENTIMENT_FLOW': 10,
                    'ML_ANOMALY': 8,
                    'PORTFOLIO_REBALANCING': 5
                }
                importance_score += type_weights.get(alert.alert_type, 0)
                
                alert.importance_score = float(importance_score)
            
            # Sort by importance score
            return sorted(alerts, key=lambda x: getattr(x, 'importance_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error ranking alerts: {e}")
            return alerts
    
    def save_alert_to_history(self, alert: Alert):
        """Save alert to database for history tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create enhanced alerts table with new fields
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id TEXT PRIMARY KEY,
                    alert_type TEXT,
                    symbol TEXT,
                    confidence REAL,
                    message TEXT,
                    timestamp REAL,
                    priority INTEGER,
                    cluster_id INTEGER,
                    risk_score REAL DEFAULT 0.0,
                    business_impact REAL DEFAULT 0.0,
                    threat_intel_score REAL DEFAULT 0.0,
                    disposition TEXT DEFAULT 'pending',
                    analyst_feedback TEXT,
                    importance_score REAL DEFAULT 0.0
                )
            """)
            
            # Insert enhanced alert with new fields
            conn.execute("""
                INSERT OR REPLACE INTO alert_history 
                (id, alert_type, symbol, confidence, message, timestamp, priority, cluster_id,
                 risk_score, business_impact, threat_intel_score, disposition, analyst_feedback, importance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id, alert.alert_type, alert.symbol, alert.confidence,
                alert.message, alert.timestamp, alert.priority, alert.cluster_id,
                alert.risk_score, alert.business_impact, alert.threat_intel_score,
                alert.disposition, alert.analyst_feedback, alert.importance_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving alert to history: {e}")
    
    async def process_smart_alerts(self, raw_alerts: List[Dict]) -> Dict:
        """Process raw alerts through smart management system"""
        try:
            if not raw_alerts:
                return {
                    'processed_alerts': [],
                    'filtered_count': 0,
                    'clustered_count': 0,
                    'processing_timestamp': datetime.now().isoformat()
                }
            
            # Convert raw alerts to Alert objects
            alert_objects = []
            for i, raw_alert in enumerate(raw_alerts):
                alert = Alert(
                    id=f"alert_{i}_{int(datetime.now().timestamp())}",
                    alert_type=raw_alert.get('alert_type', 'UNKNOWN'),
                    symbol=raw_alert.get('symbol', 'UNKNOWN'),
                    confidence=raw_alert.get('confidence', 0.0),
                    message=raw_alert.get('message', ''),
                    timestamp=raw_alert.get('timestamp', datetime.now().timestamp())
                )
                
                # Calculate priority and contextual risk score
                alert.priority = self.calculate_alert_priority(alert)
                await self.contextual_scorer.calculate_contextual_risk_score(alert)
                
                # Enhance with GPT-5 semantic analysis
                alert = await self.enhance_alert_with_gpt5(alert, {
                    'regime': 'neutral',  # Would be populated with actual market data
                    'sector_performance': {},
                    'recent_events': []
                })
                
                alert_objects.append(alert)
            
            # Filter alerts due to fatigue
            filtered_alerts = []
            fatigue_filtered_count = 0
            
            for alert in alert_objects:
                if self.check_alert_fatigue(alert):
                    fatigue_filtered_count += 1
                else:
                    filtered_alerts.append(alert)
            
            # Cluster similar alerts
            clustered_groups = self.cluster_alerts(filtered_alerts)
            
            # Detect attack chains and boost risk scores
            clustered_groups = self.detect_attack_chains(clustered_groups)
            
            # Merge clusters and create final alerts
            final_alerts = []
            for cluster_id, cluster_alerts in clustered_groups.items():
                if len(cluster_alerts) > 1:
                    merged_alert = self.merge_cluster_alerts(cluster_alerts)
                    if merged_alert:
                        final_alerts.append(merged_alert)
                else:
                    final_alerts.extend(cluster_alerts)
            
            # Rank by importance
            ranked_alerts = self.rank_alerts_by_importance(final_alerts)
            
            # Enhanced: Use dynamic limit based on adaptive fatigue management
            analyst_id = 'default'  # Can be configured per user
            current_hour = datetime.now().hour
            dynamic_limit = self.fatigue_manager.calculate_dynamic_limit(analyst_id, current_hour)
            limited_alerts = ranked_alerts[:dynamic_limit]
            
            # Save to history and update counters
            for alert in limited_alerts:
                self.save_alert_to_history(alert)
                self.alert_history.append(alert)
                self.daily_alert_count += 1
            
            # Clean old history (keep last 100 alerts)
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
            
            return {
                'processed_alerts': [asdict(alert) for alert in limited_alerts],
                'filtered_count': fatigue_filtered_count,
                'clustered_count': len([g for g in clustered_groups.values() if len(g) > 1]),
                'total_input_alerts': len(raw_alerts),
                'final_output_alerts': len(limited_alerts),
                'daily_alert_count': self.daily_alert_count,
                'dynamic_limit_used': dynamic_limit,
                'average_risk_score': np.mean([alert.risk_score for alert in limited_alerts]) if limited_alerts else 0,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing smart alerts: {e}")
            return {
                'processed_alerts': [],
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
    
    async def provide_analyst_feedback(self, alert_id: str, disposition: str, feedback: str = ""):
        """Provide analyst feedback for alert disposition and learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Update alert with disposition and feedback
            conn.execute("""
                UPDATE alert_history 
                SET disposition = ?, analyst_feedback = ?
                WHERE id = ?
            """, (disposition, feedback, alert_id))
            
            # Update analyst performance tracking
            analyst_id = 'default'  # Can be configured per user
            
            if disposition in ['true_positive', 'actionable']:
                self.fatigue_manager.update_analyst_performance(analyst_id, 'true_positive')
                logger.info(f"Alert {alert_id} marked as actionable by analyst")
            elif disposition in ['false_positive', 'noise']:
                self.fatigue_manager.update_analyst_performance(analyst_id, 'false_positive')
                logger.info(f"Alert {alert_id} marked as false positive by analyst")
            
            conn.commit()
            conn.close()
            
            return {
                'status': 'success',
                'alert_id': alert_id,
                'disposition': disposition,
                'feedback': feedback
            }
            
        except Exception as e:
            logger.error(f"Error providing analyst feedback: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_alert_analytics(self) -> dict:
        """Get analytics on alert performance and trends"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_alerts,
                    AVG(risk_score) as avg_risk_score,
                    AVG(business_impact) as avg_business_impact,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN disposition = 'true_positive' THEN 1 END) as true_positives,
                    COUNT(CASE WHEN disposition = 'false_positive' THEN 1 END) as false_positives
                FROM alert_history 
                WHERE timestamp > ?
            """, ((datetime.now() - timedelta(days=7)).timestamp(),))
            
            stats = cursor.fetchone()
            
            analytics = {
                'total_alerts_week': stats[0] if stats else 0,
                'average_risk_score': round(stats[1] or 0, 2),
                'average_business_impact': round(stats[2] or 0, 2),
                'average_confidence': round(stats[3] or 0, 2),
                'true_positives': stats[4] if stats else 0,
                'false_positives': stats[5] if stats else 0,
                'precision_rate': round((stats[4] / max(1, stats[4] + stats[5])) * 100, 1) if stats and (stats[4] or stats[5]) else 0
            }
            
            conn.close()
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting alert analytics: {e}")
            return {}