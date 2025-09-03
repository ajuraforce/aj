#!/usr/bin/env python3
"""
RSS Scheduler Utility
Manages periodic RSS feed scanning and analysis operations
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os

logger = logging.getLogger(__name__)

class RSSScheduler:
    """
    RSS Scheduler for periodic feed scanning and analysis
    Integrates with the RSS analyzer and provides intelligent scheduling
    """
    
    def __init__(self, rss_analyzer=None):
        self.rss_analyzer = rss_analyzer
        self.running = False
        self.task_handle = None
        
        # Configuration
        self.config = self.load_config()
        self.scan_intervals = self.config.get('scan_intervals', {
            'crypto': 300,      # 5 minutes for crypto feeds
            'finance': 600,     # 10 minutes for finance feeds  
            'geopolitics': 900, # 15 minutes for geopolitics feeds
            'default': 600      # 10 minutes default
        })
        
        # Analysis configuration
        self.analysis_config = self.config.get('analysis', {
            'batch_size': 50,
            'max_daily_analyses': 1000,
            'priority_threshold': 0.7,
            'enable_smart_scheduling': True
        })
        
        # State tracking
        self.last_scan_times = {}
        self.analysis_counts = {'daily': 0, 'last_reset': datetime.now().date()}
        self.feed_health = {}
        
        # Performance metrics
        self.metrics = {
            'scans_completed': 0,
            'analyses_completed': 0,
            'articles_processed': 0,
            'errors_encountered': 0,
            'last_successful_scan': None
        }
        
        logger.info("RSS Scheduler initialized")
    
    def load_config(self) -> Dict:
        """Load scheduler configuration"""
        config_path = 'config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('rss_scheduler', {})
            except Exception as e:
                logger.warning(f"Failed to load scheduler config: {e}")
        
        # Return default configuration
        return {
            'scan_intervals': {
                'crypto': 300,
                'finance': 600,
                'geopolitics': 900,
                'default': 600
            },
            'analysis': {
                'batch_size': 50,
                'max_daily_analyses': 1000,
                'priority_threshold': 0.7,
                'enable_smart_scheduling': True
            }
        }
    
    async def start(self):
        """Start the RSS scheduler"""
        if self.running:
            logger.warning("RSS Scheduler is already running")
            return
            
        if not self.rss_analyzer:
            logger.error("RSS Analyzer not provided - cannot start scheduler")
            return
            
        self.running = True
        self.task_handle = asyncio.create_task(self._scheduler_loop())
        logger.info("RSS Scheduler started")
    
    async def stop(self):
        """Stop the RSS scheduler"""
        if not self.running:
            return
            
        self.running = False
        if self.task_handle:
            self.task_handle.cancel()
            try:
                await self.task_handle
            except asyncio.CancelledError:
                pass
        
        logger.info("RSS Scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("RSS Scheduler loop started")
        
        while self.running:
            try:
                # Reset daily counters if needed
                self._reset_daily_counters()
                
                # Check which categories need scanning
                categories_to_scan = self._get_categories_needing_scan()
                
                # Perform scans
                for category in categories_to_scan:
                    await self._scan_category(category)
                
                # Perform analysis if needed
                if self._should_perform_analysis():
                    await self._perform_analysis()
                
                # Update health monitoring
                await self._update_feed_health()
                
                # Log metrics periodically
                self._log_metrics()
                
                # Sleep until next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in RSS scheduler loop: {e}")
                self.metrics['errors_encountered'] += 1
                await asyncio.sleep(30)  # Wait before retrying
    
    def _reset_daily_counters(self):
        """Reset daily analysis counters if needed"""
        current_date = datetime.now().date()
        if self.analysis_counts['last_reset'] != current_date:
            self.analysis_counts['daily'] = 0
            self.analysis_counts['last_reset'] = current_date
            logger.info("Daily analysis counters reset")
    
    def _get_categories_needing_scan(self) -> List[str]:
        """Determine which categories need scanning based on intervals"""
        current_time = time.time()
        categories_to_scan = []
        
        for category, interval in self.scan_intervals.items():
            if category == 'default':
                continue
                
            last_scan = self.last_scan_times.get(category, 0)
            if current_time - last_scan >= interval:
                categories_to_scan.append(category)
        
        return categories_to_scan
    
    async def _scan_category(self, category: str):
        """Scan RSS feeds for a specific category"""
        try:
            logger.info(f"Scanning RSS category: {category}")
            
            # Get feeds for this category from RSS analyzer
            feeds = self._get_feeds_for_category(category)
            
            # Perform scanning (this would integrate with news scanner)
            scan_results = await self._perform_category_scan(category, feeds)
            
            # Update last scan time
            self.last_scan_times[category] = time.time()
            self.metrics['scans_completed'] += 1
            self.metrics['last_successful_scan'] = datetime.now()
            
            logger.info(f"Completed scanning {category}: {scan_results}")
            
        except Exception as e:
            logger.error(f"Failed to scan category {category}: {e}")
            self.metrics['errors_encountered'] += 1
    
    def _get_feeds_for_category(self, category: str) -> List[str]:
        """Get RSS feeds for a specific category"""
        # This would be retrieved from config or RSS analyzer
        category_feeds = {
            'crypto': [
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://cointelegraph.com/rss',
                'https://decrypt.co/feed'
            ],
            'finance': [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://www.marketwatch.com/rss/topstories'
            ],
            'geopolitics': [
                'https://feeds.reuters.com/Reuters/worldNews',
                'https://feeds.reuters.com/reuters/businessNews'
            ]
        }
        
        return category_feeds.get(category, [])
    
    async def _perform_category_scan(self, category: str, feeds: List[str]) -> Dict:
        """Perform actual RSS scanning for a category"""
        # This would integrate with the news scanner or RSS analyzer
        # For now, return a mock result
        return {
            'category': category,
            'feeds_scanned': len(feeds),
            'articles_found': 0,  # Would be actual count
            'timestamp': datetime.now().isoformat()
        }
    
    def _should_perform_analysis(self) -> bool:
        """Determine if analysis should be performed"""
        # Check daily limits
        if self.analysis_counts['daily'] >= self.analysis_config['max_daily_analyses']:
            return False
        
        # Check if RSS analyzer is available
        if not self.rss_analyzer:
            return False
        
        # Check if smart scheduling is enabled
        if self.analysis_config.get('enable_smart_scheduling', True):
            return self._smart_analysis_check()
        
        # Default: analyze every 10 minutes
        last_analysis = getattr(self, '_last_analysis_time', 0)
        return time.time() - last_analysis >= 600
    
    def _smart_analysis_check(self) -> bool:
        """Smart analysis scheduling based on system load and content priority"""
        try:
            # Check system resources (simplified)
            current_hour = datetime.now().hour
            is_market_hours = 9 <= current_hour <= 16  # Market hours
            
            # Prioritize analysis during market hours
            if is_market_hours:
                analysis_interval = 300  # 5 minutes
            else:
                analysis_interval = 900  # 15 minutes
            
            last_analysis = getattr(self, '_last_analysis_time', 0)
            return time.time() - last_analysis >= analysis_interval
            
        except Exception as e:
            logger.warning(f"Smart analysis check failed: {e}")
            return False
    
    async def _perform_analysis(self):
        """Perform RSS article analysis"""
        try:
            logger.info("Starting RSS analysis batch")
            
            # Get batch size
            batch_size = self.analysis_config.get('batch_size', 50)
            
            # Perform analysis using RSS analyzer
            if self.rss_analyzer:
                analyzed_count = await self._run_analysis_batch(batch_size)
                
                # Update counters
                self.analysis_counts['daily'] += analyzed_count
                self.metrics['analyses_completed'] += 1
                self.metrics['articles_processed'] += analyzed_count
                self._last_analysis_time = time.time()
                
                logger.info(f"Analysis completed: {analyzed_count} articles processed")
            else:
                logger.warning("RSS analyzer not available for analysis")
                
        except Exception as e:
            logger.error(f"Failed to perform RSS analysis: {e}")
            self.metrics['errors_encountered'] += 1
    
    async def _run_analysis_batch(self, batch_size: int) -> int:
        """Run analysis batch on RSS analyzer"""
        try:
            # This would call the RSS analyzer's process_batch method
            if self.rss_analyzer and hasattr(self.rss_analyzer, 'process_batch'):
                return await asyncio.to_thread(self.rss_analyzer.process_batch, batch_size)
            else:
                logger.warning("RSS analyzer does not have process_batch method")
                return 0
        except Exception as e:
            logger.error(f"Analysis batch execution failed: {e}")
            return 0
    
    async def _update_feed_health(self):
        """Update feed health monitoring"""
        try:
            # This would check feed connectivity and update health status
            for category in self.scan_intervals.keys():
                if category == 'default':
                    continue
                    
                feeds = self._get_feeds_for_category(category)
                health_status = await self._check_feed_health(feeds)
                self.feed_health[category] = health_status
                
        except Exception as e:
            logger.error(f"Failed to update feed health: {e}")
    
    async def _check_feed_health(self, feeds: List[str]) -> Dict:
        """Check health status of RSS feeds"""
        # Simplified health check
        return {
            'total_feeds': len(feeds),
            'healthy_feeds': len(feeds),  # Would be actual count
            'last_check': datetime.now().isoformat(),
            'status': 'healthy'
        }
    
    def _log_metrics(self):
        """Log performance metrics"""
        current_time = time.time()
        
        # Log every 30 minutes
        if not hasattr(self, '_last_metrics_log') or current_time - self._last_metrics_log >= 1800:
            logger.info(f"RSS Scheduler metrics: {self.metrics}")
            self._last_metrics_log = current_time
    
    def get_status(self) -> Dict:
        """Get scheduler status and metrics"""
        return {
            'running': self.running,
            'last_scan_times': self.last_scan_times,
            'analysis_counts': self.analysis_counts,
            'feed_health': self.feed_health,
            'metrics': self.metrics,
            'config': {
                'scan_intervals': self.scan_intervals,
                'analysis_config': self.analysis_config
            }
        }
    
    def update_config(self, new_config: Dict):
        """Update scheduler configuration"""
        try:
            if 'scan_intervals' in new_config:
                self.scan_intervals.update(new_config['scan_intervals'])
            
            if 'analysis' in new_config:
                self.analysis_config.update(new_config['analysis'])
            
            logger.info("RSS Scheduler configuration updated")
            
        except Exception as e:
            logger.error(f"Failed to update scheduler config: {e}")
    
    async def trigger_immediate_scan(self, category: Optional[str] = None):
        """Trigger an immediate scan for a category or all categories"""
        try:
            if category:
                await self._scan_category(category)
                logger.info(f"Immediate scan completed for category: {category}")
            else:
                for cat in self.scan_intervals.keys():
                    if cat != 'default':
                        await self._scan_category(cat)
                logger.info("Immediate scan completed for all categories")
                
        except Exception as e:
            logger.error(f"Failed to perform immediate scan: {e}")
    
    async def trigger_immediate_analysis(self):
        """Trigger immediate analysis regardless of scheduling"""
        try:
            await self._perform_analysis()
            logger.info("Immediate analysis completed")
        except Exception as e:
            logger.error(f"Failed to perform immediate analysis: {e}")