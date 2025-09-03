"""
Alert Sender Module
Implements multi-channel alert delivery with rate limiting and escalation
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import aiohttp
import json
import time
import redis
from utils.signal_schema import Signal

logger = logging.getLogger(__name__)

class AlertSender:
    """Multi-channel alert sending with escalation and rate limiting"""
    
    def __init__(self):
        self.alert_history = []
        self.rate_limits = {}
        self.session = None
        self.channels = self.setup_channels()
        self.recent_alerts = {}  # For deduplication
        self.alert_cooldowns = {}  # Symbol-specific cooldowns
        # Redis connection for signal queue
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()  # Test connection
            logger.info("Redis connection established for signal queue")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Signal queue will use in-memory fallback")
            self.redis_client = None
    
    def process_alert(self, signal_data: Dict):
        """Enhanced alert processing with Redis-based deduplication and cooldowns"""
        try:
            signal = Signal(**signal_data)
        except ValueError as e:
            logger.error(f"Invalid signal data: {e}")
            return False

        # Enhanced Dedup Key
        dedup_key = f"alert:{signal.symbol}:{signal.signal_type}:{int(signal.confidence * 100)}"
        
        # Cooldown based on priority levels from the snippet
        priority = 'High' if signal.confidence > 0.8 else 'Medium' if signal.confidence > 0.5 else 'Low'
        cooldown_map = {'High': 900, 'Medium': 600, 'Low': 300}  # seconds
        cooldown = cooldown_map[priority]
        
        # Check cooldown using Redis
        if self.redis_client:
            try:
                last_sent = self.redis_client.get(dedup_key)
                if last_sent and (datetime.now() - datetime.fromtimestamp(float(last_sent))) < timedelta(seconds=cooldown):
                    logger.info(f"Alert {signal.id} skipped (in cooldown)")
                    return False
                
                # Set cooldown in Redis
                self.redis_client.set(dedup_key, datetime.now().timestamp(), ex=cooldown)
            except Exception as e:
                logger.error(f"Redis cooldown check failed: {e}")
                # Fallback to in-memory cooldown
                return self._fallback_cooldown_check(signal, dedup_key, cooldown)
        else:
            # Fallback to in-memory cooldown
            return self._fallback_cooldown_check(signal, dedup_key, cooldown)

        # Log successful alert processing
        logger.info(f"Alert {signal.id} processed (priority: {priority}, cooldown: {cooldown}s)")
        
        # Continue with existing alert generation logic
        return self.generate_alert(signal_data)

    def _fallback_cooldown_check(self, signal: Signal, dedup_key: str, cooldown: int) -> bool:
        """Fallback cooldown check when Redis is not available"""
        current_time = datetime.utcnow()
        
        if dedup_key in self.alert_cooldowns:
            if current_time < self.alert_cooldowns[dedup_key]:
                logger.info(f"Signal {signal.id} skipped (in-memory cooldown)")
                return False

        # Set cooldown
        self.alert_cooldowns[dedup_key] = current_time + timedelta(seconds=cooldown)
        return True

    def generate_alert(self, signal_data: Dict):
        """Generate standardized alert from signal data and push to queue"""
        try:
            signal = Signal(**signal_data)
        except ValueError as e:
            logger.error(f"Invalid signal data: {e}")
            return

        # Legacy dedup check for backwards compatibility
        dedup_key = f"{signal.symbol}_{signal.signal_type}_{int(signal.confidence * 100)}"
        current_time = datetime.utcnow()
        
        if dedup_key in self.alert_cooldowns:
            if current_time < self.alert_cooldowns[dedup_key]:
                logger.info(f"Signal {signal.id} skipped (legacy cooldown)")
                return

        # Set legacy cooldown (15min for high confidence, 5min for others)
        cooldown_minutes = 15 if signal.confidence > 0.8 else 5
        self.alert_cooldowns[dedup_key] = current_time + timedelta(minutes=cooldown_minutes)

        # Push to Redis queue for consumers (paper trader, notifications)
        if self.redis_client:
            try:
                self.redis_client.rpush('signals_queue', json.dumps(signal.dict()))
                logger.info(f"Signal {signal.id} pushed to Redis queue")
            except Exception as e:
                logger.error(f"Failed to push signal to Redis: {e}")

        # Convert to pattern format for existing alert system
        pattern = self._signal_to_pattern(signal)
        score = signal.confidence * 100
        
        # Send immediate alert via existing channels
        asyncio.create_task(self.send_alert(pattern, score))
        
        # Send web notification via Socket.IO
        try:
            from app import socketio
            socketio.emit('new_alert', signal.dict())
        except Exception as e:
            logger.debug(f"Socket.IO not available: {e}")

        logger.info(f"Alert generated: {signal.id}")

    def _signal_to_pattern(self, signal: Signal) -> Dict:
        """Convert Signal object to pattern format for existing alert system"""
        return {
            'id': signal.id,
            'asset': signal.symbol.split('/')[0] if '/' in signal.symbol else signal.symbol,
            'type': 'trading_signal',
            'source': 'signal_engine',
            'signals': {
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'timeframe': signal.timeframe,
                'reasons': signal.reason,
                'expires_in': signal.expires_in
            }
        }

    def setup_channels(self) -> Dict:
        """Setup available alert channels"""
        return {
            'telegram': {
                'enabled': bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID')),
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
                'rate_limit': 300  # 5 minutes between messages
            },
            'discord': {
                'enabled': bool(os.getenv('DISCORD_WEBHOOK_URL')),
                'webhook_url': os.getenv('DISCORD_WEBHOOK_URL', ''),
                'rate_limit': 180  # 3 minutes between messages
            },
            'reddit': {
                'enabled': bool(os.getenv('REDDIT_USERNAME') and os.getenv('REDDIT_PASSWORD')),
                'rate_limit': 600  # 10 minutes for Reddit posts
            },
            'email': {
                'enabled': bool(os.getenv('SMTP_SERVER') and os.getenv('EMAIL_TO')),
                'smtp_server': os.getenv('SMTP_SERVER', ''),
                'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                'email_from': os.getenv('EMAIL_FROM', ''),
                'email_to': os.getenv('EMAIL_TO', ''),
                'password': os.getenv('EMAIL_PASSWORD', ''),
                'rate_limit': 300  # 5 minutes for email
            },
            'webhook': {
                'enabled': bool(os.getenv('WEBHOOK_URL')),
                'url': os.getenv('WEBHOOK_URL', ''),
                'rate_limit': 120  # 2 minutes between messages
            }
        }
    
    async def send_alert(self, pattern: Dict, score: float):
        """Send alert through appropriate channels based on score and pattern"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30)
                )
            
            # Check for duplicate alerts first
            if self.is_duplicate_alert(pattern, score):
                logger.debug(f"Skipping duplicate alert for {pattern.get('asset', 'unknown')}")
                return
            
            # Determine alert priority and channels
            priority = self.determine_priority(score, pattern)
            channels = self.select_channels(priority, pattern)
            
            if not channels:
                # Reduced logging frequency - only log once per session
                if not hasattr(self, '_no_channels_logged'):
                    logger.info("No alert channels configured - alerts will be skipped")
                    self._no_channels_logged = True
                return
            
            # Generate alert content
            alert_content = self.generate_alert_content(pattern, score, priority)
            
            # Send through selected channels
            tasks = []
            for channel in channels:
                if await self.check_rate_limit(channel):
                    tasks.append(self.send_to_channel(channel, alert_content, pattern))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                logger.info(f"Alert sent to {success_count}/{len(tasks)} channels")
                
                # Record alert
                await self.record_alert(pattern, score, priority, success_count)
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def is_duplicate_alert(self, pattern: Dict, score: float) -> bool:
        """Check if this alert is a duplicate of a recent one"""
        try:
            asset = pattern.get('asset', 'unknown')
            pattern_type = pattern.get('type', 'unknown')
            
            # Create alert signature for deduplication
            alert_signature = f"{asset}_{pattern_type}_{int(score)}"
            
            current_time = time.time()
            
            # Check if we've seen this exact alert recently (within 10 minutes)
            if alert_signature in self.recent_alerts:
                last_time = self.recent_alerts[alert_signature]
                if current_time - last_time < 600:  # 10 minutes cooldown
                    return True
            
            # Check symbol-specific cooldown for critical alerts
            if score >= 90:  # Critical alerts
                symbol_cooldown_key = f"{asset}_critical"
                if symbol_cooldown_key in self.alert_cooldowns:
                    last_critical = self.alert_cooldowns[symbol_cooldown_key]
                    if current_time - last_critical < 900:  # 15 minutes for critical
                        return True
                self.alert_cooldowns[symbol_cooldown_key] = current_time
            
            # Record this alert
            self.recent_alerts[alert_signature] = current_time
            
            # Clean up old entries (older than 1 hour)
            cutoff_time = current_time - 3600
            self.recent_alerts = {k: v for k, v in self.recent_alerts.items() if v > cutoff_time}
            self.alert_cooldowns = {k: v for k, v in self.alert_cooldowns.items() if v > cutoff_time}
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate alert: {e}")
            return False
    
    def determine_priority(self, score: float, pattern: Dict) -> str:
        """Determine alert priority based on score and pattern type"""
        try:
            pattern_type = pattern.get('type', '')
            
            # Critical alerts (score > 90 or high-impact news)
            if score > 90:
                return 'critical'
            
            if pattern_type == 'news_impact':
                impact = pattern.get('signals', {}).get('potential_impact', 'low')
                if impact == 'high':
                    return 'critical'
            
            # High priority alerts (score > 75)
            if score > 75:
                return 'high'
            
            # Medium priority alerts (score > 60)
            if score > 60:
                return 'medium'
            
            # Low priority (score <= 60)
            return 'low'
            
        except Exception as e:
            logger.error(f"Error determining priority: {e}")
            return 'low'
    
    def select_channels(self, priority: str, pattern: Dict) -> List[str]:
        """Select appropriate channels based on priority"""
        try:
            channels = []
            
            if priority == 'critical':
                # Use all available channels for critical alerts
                channels = [name for name, config in self.channels.items() if config['enabled']]
            elif priority == 'high':
                # Telegram + Discord for high priority
                if self.channels['telegram']['enabled']:
                    channels.append('telegram')
                if self.channels['discord']['enabled']:
                    channels.append('discord')
                if self.channels['webhook']['enabled']:
                    channels.append('webhook')
            elif priority == 'medium':
                # Telegram or Discord for medium priority
                if self.channels['telegram']['enabled']:
                    channels.append('telegram')
                elif self.channels['discord']['enabled']:
                    channels.append('discord')
            elif priority == 'community':
                # Community channels for collaborative alerts
                if self.channels['reddit']['enabled']:
                    channels.append('reddit')
                if self.channels['discord']['enabled']:
                    channels.append('discord')
            else:  # low priority
                # Only webhook for low priority
                if self.channels['webhook']['enabled']:
                    channels.append('webhook')
            
            return channels
            
        except Exception as e:
            logger.error(f"Error selecting channels: {e}")
            return []
    
    def generate_alert_content(self, pattern: Dict, score: float, priority: str) -> Dict:
        """Generate alert content for different channels"""
        try:
            asset = pattern.get('asset', 'UNKNOWN')
            pattern_type = pattern.get('type', 'unknown')
            source = pattern.get('source', 'unknown')
            signals = pattern.get('signals', {})
            
            # Base content
            title = f"🚨 {priority.upper()} Alert: {asset}"
            
            # Pattern-specific details
            if pattern_type == 'mention_spike':
                detail = f"Mention spike: {signals.get('mention_count', 0)} mentions/hour"
            elif pattern_type == 'price_movement':
                change = signals.get('price_change_percent', 0)
                detail = f"Price movement: {change:+.2f}%"
            elif pattern_type == 'volume_spike':
                ratio = signals.get('volume_ratio', 1)
                detail = f"Volume spike: {ratio:.1f}x normal"
            elif pattern_type == 'news_impact':
                impact = signals.get('potential_impact', 'medium')
                detail = f"News impact: {impact} severity"
            elif pattern_type == 'cross_source_correlation':
                sources = signals.get('sources', [])
                detail = f"Multi-source signal: {', '.join(sources)}"
            else:
                detail = f"Pattern: {pattern_type}"
            
            # Priority indicators
            priority_emoji = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }
            
            emoji = priority_emoji.get(priority, '⚪')
            
            # Construct messages for different platforms
            telegram_text = f"{emoji} {title}\n\n📊 Score: {score:.1f}/100\n📈 {detail}\n🎯 Source: {source}\n⏰ {datetime.utcnow().strftime('%H:%M UTC')}"
            
            discord_content = {
                "embeds": [{
                    "title": f"{emoji} {asset} Alert",
                    "description": detail,
                    "color": self.get_priority_color(priority),
                    "fields": [
                        {"name": "Score", "value": f"{score:.1f}/100", "inline": True},
                        {"name": "Source", "value": source, "inline": True},
                        {"name": "Priority", "value": priority.title(), "inline": True}
                    ],
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            
            webhook_payload = {
                "alert_type": "trading_signal",
                "priority": priority,
                "asset": asset,
                "score": score,
                "pattern_type": pattern_type,
                "source": source,
                "details": detail,
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "signals": signals
            }
            
            return {
                'telegram': telegram_text,
                'discord': discord_content,
                'email': {
                    'subject': f"{priority.upper()} Alert: {asset} - Score {score:.1f}",
                    'body': f"{title}\n\n{detail}\n\nScore: {score:.1f}/100\nSource: {source}\nTime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\nSignals: {json.dumps(signals, indent=2)}"
                },
                'webhook': webhook_payload
            }
            
        except Exception as e:
            logger.error(f"Error generating alert content: {e}")
            return {}
    
    def get_priority_color(self, priority: str) -> int:
        """Get Discord embed color for priority"""
        colors = {
            'critical': 0xFF0000,  # Red
            'high': 0xFF8C00,      # Orange
            'medium': 0xFFD700,    # Gold
            'low': 0x00FF00        # Green
        }
        return colors.get(priority, 0x808080)  # Gray default
    
    async def check_rate_limit(self, channel: str) -> bool:
        """Check if channel is rate limited"""
        try:
            now = time.time()
            config = self.channels.get(channel, {})
            rate_limit = config.get('rate_limit', 60)
            
            if channel in self.rate_limits:
                last_sent = self.rate_limits[channel]
                if now - last_sent < rate_limit:
                    logger.debug(f"Rate limited for {channel}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    async def send_to_channel(self, channel: str, content: Dict, pattern: Dict) -> bool:
        """Send alert to specific channel"""
        try:
            config = self.channels[channel]
            
            if channel == 'telegram':
                success = await self.send_telegram(config, content['telegram'])
            elif channel == 'discord':
                success = await self.send_discord(config, content['discord'])
            elif channel == 'reddit':
                success = await self.broadcast_to_community(content, 'reddit')
            elif channel == 'email':
                success = await self.send_email(config, content['email'])
            elif channel == 'webhook':
                success = await self.send_webhook(config, content['webhook'])
            else:
                logger.warning(f"Unknown channel: {channel}")
                return False
            
            if success:
                self.rate_limits[channel] = time.time()
                logger.info(f"Alert sent via {channel}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending to {channel}: {e}")
            return False
    
    async def send_telegram(self, config: Dict, text: str) -> bool:
        """Send alert via Telegram"""
        try:
            if not self.session:
                logger.error("HTTP session not available for Telegram")
                return False
                
            url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            payload = {
                'chat_id': config['chat_id'],
                'text': text,
                'parse_mode': 'HTML'
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return True
                else:
                    logger.error(f"Telegram API error: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Telegram: {e}")
            return False
    
    async def send_discord(self, config: Dict, content: Dict) -> bool:
        """Send alert via Discord webhook"""
        try:
            if not self.session:
                logger.error("HTTP session not available for Discord")
                return False
                
            async with self.session.post(config['webhook_url'], json=content) as response:
                if response.status in [200, 204]:
                    return True
                else:
                    logger.error(f"Discord webhook error: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Discord: {e}")
            return False
    
    async def send_email(self, config: Dict, content: Dict) -> bool:
        """Send alert via email (SMTP)"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Run in executor to avoid blocking
            def send_smtp():
                try:
                    msg = MIMEMultipart()
                    msg['From'] = config['email_from']
                    msg['To'] = config['email_to']
                    msg['Subject'] = content['subject']
                    
                    msg.attach(MIMEText(content['body'], 'plain'))
                    
                    server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
                    server.starttls()
                    server.login(config['email_from'], config['password'])
                    server.send_message(msg)
                    server.quit()
                    return True
                except Exception as e:
                    logger.error(f"SMTP error: {e}")
                    return False
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, send_smtp)
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def send_webhook(self, config: Dict, payload: Dict) -> bool:
        """Send alert via custom webhook"""
        try:
            if not self.session:
                logger.error("HTTP session not available for webhook")
                return False
                
            headers = {'Content-Type': 'application/json'}
            
            async with self.session.post(
                config['url'], 
                json=payload,
                headers=headers
            ) as response:
                if 200 <= response.status < 300:
                    return True
                else:
                    logger.error(f"Webhook error: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False
    
    async def record_alert(self, pattern: Dict, score: float, priority: str, success_count: int):
        """Record alert in history"""
        try:
            alert_record = {
                'id': f"alert_{int(time.time())}",
                'pattern_id': pattern.get('id', ''),
                'asset': pattern.get('asset', ''),
                'pattern_type': pattern.get('type', ''),
                'source': pattern.get('source', 'unknown'),
                'source_url': self.generate_source_url(pattern),
                'score': score,
                'priority': priority,
                'channels_sent': success_count,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            self.alert_history.append(alert_record)
            
            # Limit history size
            self.alert_history = self.alert_history[-100:]
            
        except Exception as e:
            logger.error(f"Error recording alert: {e}")
    
    def generate_source_url(self, pattern: Dict) -> str:
        """Generate clickable source URL based on pattern source and data"""
        try:
            source = pattern.get('source', '')
            signals = pattern.get('signals', {})
            
            if source == 'reddit':
                # Check if we have Reddit permalink in signals
                if 'permalink' in signals:
                    return f"https://reddit.com{signals['permalink']}"
                else:
                    # Fallback to subreddit if no specific post
                    subreddit = signals.get('subreddit', 'CryptoCurrency')
                    return f"https://reddit.com/r/{subreddit}"
            elif source == 'news':
                # Check for news URL in signals
                if 'url' in signals:
                    return signals['url']
                elif 'news_source' in signals:
                    return signals['news_source']
                else:
                    return "https://news.google.com"
            elif source == 'binance':
                asset = pattern.get('asset', '')
                if asset:
                    return f"https://www.binance.com/en/trade/{asset}_USDT"
                else:
                    return "https://www.binance.com"
            elif source == 'multi_source':
                return "#"  # Internal correlation, no external link
            else:
                return ""
                
        except Exception as e:
            logger.warning(f"Error generating source URL: {e}")
            return ""
    
    def get_alert_history(self) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history
    
    async def test_channels(self) -> Dict[str, bool]:
        """Test all configured channels"""
        results = {}
        
        test_pattern = {
            'id': 'test_alert',
            'asset': 'TEST',
            'type': 'test',
            'source': 'system',
            'signals': {'test': True}
        }
        
        test_content = self.generate_alert_content(test_pattern, 50.0, 'low')
        
        for channel, config in self.channels.items():
            if config['enabled']:
                try:
                    success = await self.send_to_channel(channel, test_content, test_pattern)
                    results[channel] = success
                except Exception as e:
                    logger.error(f"Error testing {channel}: {e}")
                    results[channel] = False
            else:
                results[channel] = False
        
        return results
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def broadcast_to_community(self, alert_content: Dict, community: str) -> bool:
        """
        Broadcast alerts to community channels.
        """
        try:
            if community == 'reddit':
                # Use reddit_poster for community sharing
                from executor.reddit_poster import RedditPoster
                poster = RedditPoster()
                # Convert alert to pattern format for Reddit posting
                pattern = {
                    'asset': alert_content.get('webhook', {}).get('asset', 'CRYPTO'),
                    'type': alert_content.get('webhook', {}).get('pattern_type', 'alert'),
                    'source': alert_content.get('webhook', {}).get('source', 'platform'),
                    'signals': alert_content.get('webhook', {}).get('signals', {})
                }
                result = await poster.safe_post(pattern, 80.0)  # High score for community posts
                return bool(result)
            elif community == 'discord':
                return await self.send_discord(self.channels['discord'], alert_content['discord'])
            return False
        except Exception as e:
            logger.error(f"Error broadcasting to community {community}: {e}")
            return False
