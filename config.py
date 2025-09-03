"""
Configuration Management
Centralized configuration for the Social Intelligence Trading Platform
"""

import os
from typing import Dict, Any, Optional


class Config:
    """Main configuration class with environment-based settings"""
    
    # Platform Settings
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Flask Settings
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    
    # Scanner Configuration
    SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL', '30'))  # seconds
    VIRAL_THRESHOLD = float(os.getenv('VIRAL_THRESHOLD', '70.0'))
    
    # Reddit API Configuration
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'SocialIntelligenceBot/1.0')
    REDDIT_USERNAME = os.getenv('REDDIT_USERNAME', '')
    REDDIT_PASSWORD = os.getenv('REDDIT_PASSWORD', '')
    
    # Reddit Posting Limits
    REDDIT_POST_THRESHOLD = float(os.getenv('REDDIT_POST_THRESHOLD', '70.0'))
    MAX_DAILY_POSTS = int(os.getenv('MAX_DAILY_POSTS', '5'))
    MAX_POSTS_PER_SUBREDDIT = int(os.getenv('MAX_POSTS_PER_SUBREDDIT', '1'))
    MIN_POST_INTERVAL = int(os.getenv('MIN_POST_INTERVAL', '1800'))  # 30 minutes
    
    # Exchange Configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET = os.getenv('BINANCE_SECRET', '')
    BINANCE_SANDBOX = os.getenv('BINANCE_SANDBOX', 'true').lower() == 'true'
    
    # Trading Configuration
    LIVE_TRADING = os.getenv('LIVE_TRADING', 'false').lower() == 'true'
    TRADE_THRESHOLD = float(os.getenv('TRADE_THRESHOLD', '80.0'))
    MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '5'))
    
    # Risk Management
    BASE_POSITION_PERCENT = float(os.getenv('BASE_POSITION_PERCENT', '2.0'))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '200.0'))
    MIN_POSITION_SIZE = float(os.getenv('MIN_POSITION_SIZE', '10.0'))
    STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', '2.0'))
    BASE_TAKE_PROFIT_PERCENT = float(os.getenv('BASE_TAKE_PROFIT_PERCENT', '4.0'))
    DAILY_LOSS_LIMIT = float(os.getenv('DAILY_LOSS_LIMIT', '-500.0'))
    MIN_TRADING_BALANCE = float(os.getenv('MIN_TRADING_BALANCE', '100.0'))
    
    # Alert Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
    WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')
    
    # Email Configuration
    SMTP_SERVER = os.getenv('SMTP_SERVER', '')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    EMAIL_FROM = os.getenv('EMAIL_FROM', '')
    EMAIL_TO = os.getenv('EMAIL_TO', '')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5')  # Latest model as of Aug 2025
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '1.0'))  # GPT-5 only supports default
    
    # Advanced Intelligence System Configuration
    ENABLE_PATTERN_MEMORY = os.getenv('ENABLE_PATTERN_MEMORY', 'true').lower() == 'true'
    PATTERN_MEMORY_DEPTH = int(os.getenv('PATTERN_MEMORY_DEPTH', '100'))  # Number of patterns to remember
    
    # GitHub Backup Configuration
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')
    GITHUB_REPO_OWNER = os.getenv('GITHUB_REPO_OWNER', '')
    GITHUB_REPO_NAME = os.getenv('GITHUB_REPO_NAME', '')
    GITHUB_BRANCH = os.getenv('GITHUB_BRANCH', 'main')
    BACKUP_INTERVAL = int(os.getenv('BACKUP_INTERVAL', '3600'))  # 1 hour
    
    # Encryption Configuration
    ENCRYPTION_PASSPHRASE = os.getenv('ENCRYPTION_PASSPHRASE', '')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/platform.log')
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Critical validations
        if not cls.REDDIT_CLIENT_ID or not cls.REDDIT_CLIENT_SECRET:
            validation_results['warnings'].append('Reddit API credentials not configured')
        
        if not cls.BINANCE_API_KEY or not cls.BINANCE_SECRET:
            validation_results['warnings'].append('Binance API credentials not configured')
        
        if not cls.GITHUB_TOKEN or not cls.GITHUB_REPO_OWNER or not cls.GITHUB_REPO_NAME:
            validation_results['warnings'].append('GitHub backup not configured')
        
        if not cls.ENCRYPTION_PASSPHRASE:
            validation_results['warnings'].append('Encryption passphrase not set')
        
        # Alert channel validations
        alert_channels = 0
        if cls.TELEGRAM_BOT_TOKEN and cls.TELEGRAM_CHAT_ID:
            alert_channels += 1
        if cls.DISCORD_WEBHOOK_URL:
            alert_channels += 1
        if cls.WEBHOOK_URL:
            alert_channels += 1
        if cls.SMTP_SERVER and cls.EMAIL_TO:
            alert_channels += 1
            
        if alert_channels == 0:
            validation_results['warnings'].append('No alert channels configured')
        
        # Trading validations
        if cls.LIVE_TRADING and cls.BINANCE_SANDBOX:
            validation_results['errors'].append('Live trading enabled but using sandbox mode')
            validation_results['valid'] = False
        
        # Threshold validations
        if cls.TRADE_THRESHOLD <= cls.VIRAL_THRESHOLD:
            validation_results['warnings'].append('Trade threshold should be higher than viral threshold')
        
        if cls.STOP_LOSS_PERCENT >= cls.BASE_TAKE_PROFIT_PERCENT:
            validation_results['warnings'].append('Stop loss should be lower than take profit')
        
        return validation_results
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """Get configuration summary for display"""
        return {
            'platform': {
                'debug_mode': cls.DEBUG,
                'live_trading': cls.LIVE_TRADING,
                'sandbox_mode': cls.BINANCE_SANDBOX
            },
            'scanners': {
                'reddit_configured': bool(cls.REDDIT_CLIENT_ID and cls.REDDIT_CLIENT_SECRET),
                'binance_configured': bool(cls.BINANCE_API_KEY and cls.BINANCE_SECRET),
                'scan_interval': cls.SCAN_INTERVAL
            },
            'thresholds': {
                'viral_threshold': cls.VIRAL_THRESHOLD,
                'reddit_post_threshold': cls.REDDIT_POST_THRESHOLD,
                'trade_threshold': cls.TRADE_THRESHOLD
            },
            'limits': {
                'max_daily_posts': cls.MAX_DAILY_POSTS,
                'max_daily_trades': cls.MAX_DAILY_TRADES,
                'max_position_size': cls.MAX_POSITION_SIZE
            },
            'alerts': {
                'telegram_configured': bool(cls.TELEGRAM_BOT_TOKEN and cls.TELEGRAM_CHAT_ID),
                'discord_configured': bool(cls.DISCORD_WEBHOOK_URL),
                'email_configured': bool(cls.SMTP_SERVER and cls.EMAIL_TO),
                'webhook_configured': bool(cls.WEBHOOK_URL)
            },
            'backup': {
                'github_configured': bool(cls.GITHUB_TOKEN and cls.GITHUB_REPO_OWNER),
                'encryption_configured': bool(cls.ENCRYPTION_PASSPHRASE),
                'backup_interval': cls.BACKUP_INTERVAL
            }
        }


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    BINANCE_SANDBOX = True
    LIVE_TRADING = False
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    # Production settings should come from environment variables


class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    BINANCE_SANDBOX = True
    LIVE_TRADING = False
    SCAN_INTERVAL = 5  # Faster for testing


# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name: Optional[str] = None) -> Config:
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default') or 'default'
    
    return config_map.get(config_name, DevelopmentConfig)
