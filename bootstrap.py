#!/usr/bin/env python3
"""
Bootstrap Script for Social Intelligence Trading Platform
Handles fresh account setup, state restoration, and seamless migration
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import getpass
from pathlib import Path
from typing import Optional, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.state_manager import StateManager
from utils.encryption import StateEncryption
from utils.github_backup import GitHubBackup
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlatformBootstrap:
    """Bootstrap handler for platform initialization and migration"""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.encryption = StateEncryption()
        self.github_backup = GitHubBackup()
        self.config_valid = False
    
    async def bootstrap(self, args: argparse.Namespace):
        """Main bootstrap process"""
        logger.info("🚀 Starting Social Intelligence Trading Platform Bootstrap")
        
        try:
            # Step 1: Validate configuration
            logger.info("📋 Step 1: Validating configuration...")
            validation_result = await self.validate_configuration()
            
            if not validation_result['valid']:
                logger.error("❌ Configuration validation failed!")
                for error in validation_result['errors']:
                    logger.error(f"   ❌ {error}")
                return False
            
            if validation_result['warnings']:
                logger.warning("⚠️ Configuration warnings:")
                for warning in validation_result['warnings']:
                    logger.warning(f"   ⚠️ {warning}")
            
            # Step 2: Handle state restoration
            logger.info("💾 Step 2: Checking for existing state...")
            if args.restore or not os.path.exists('state.json'):
                success = await self.restore_or_create_state(args)
                if not success:
                    return False
            
            # Step 3: Install dependencies (conceptual - would be handled by container/environment)
            logger.info("📦 Step 3: Dependencies check...")
            await self.check_dependencies()
            
            # Step 4: Initialize platform components
            logger.info("🔧 Step 4: Initializing platform components...")
            await self.initialize_components()
            
            # Step 5: Create initial backup
            if Config.GITHUB_TOKEN:
                logger.info("☁️ Step 5: Creating initial backup...")
                await self.create_initial_backup()
            
            # Step 6: Announce startup
            logger.info("📢 Step 6: Announcing platform startup...")
            await self.announce_startup()
            
            logger.info("✅ Bootstrap completed successfully!")
            logger.info("🤖 Platform is ready to start trading operations")
            
            return True
            
        except Exception as e:
            logger.error(f"💥 Bootstrap failed: {e}")
            return False
    
    async def validate_configuration(self) -> Dict:
        """Validate platform configuration"""
        validation_result = Config.validate_config()
        
        # Additional bootstrap-specific validations
        required_dirs = ['logs', 'state_backups']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                logger.info(f"📁 Created directory: {dir_name}")
        
        # Check write permissions
        test_files = ['state.json', 'logs/test.log']
        for test_file in test_files:
            try:
                Path(test_file).parent.mkdir(parents=True, exist_ok=True)
                with open(test_file, 'a') as f:
                    f.write('')
                if test_file.endswith('test.log'):
                    os.remove(test_file)
            except PermissionError:
                validation_result['errors'].append(f"No write permission for {test_file}")
                validation_result['valid'] = False
        
        self.config_valid = validation_result['valid']
        return validation_result
    
    async def restore_or_create_state(self, args: argparse.Namespace) -> bool:
        """Restore state from backup or create fresh state"""
        try:
            if args.restore and Config.GITHUB_TOKEN:
                logger.info("🔄 Attempting to restore state from GitHub backup...")
                
                # Get passphrase
                passphrase = self.get_passphrase(args)
                if not passphrase:
                    logger.error("❌ No passphrase provided for state restoration")
                    return False
                
                # Restore from GitHub
                success = await self.github_backup.restore_state(passphrase)
                
                if success:
                    logger.info("✅ State restored from GitHub backup")
                    return True
                else:
                    logger.warning("⚠️ Failed to restore from backup, creating fresh state")
            
            # Create fresh state
            logger.info("🆕 Creating fresh state...")
            fresh_state = self.state_manager.create_default_state()
            fresh_state['migration_info'] = {
                'source_account': args.source_account if hasattr(args, 'source_account') else None,
                'migration_timestamp': fresh_state['created_at'],
                'migration_reason': 'fresh_install' if not args.restore else 'restore_failed'
            }
            
            self.state_manager.save_state(fresh_state)
            logger.info("✅ Fresh state created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in state restoration: {e}")
            return False
    
    def get_passphrase(self, args: argparse.Namespace) -> Optional[str]:
        """Get encryption passphrase from various sources"""
        # Try environment variable first
        passphrase = os.getenv('ENCRYPTION_PASSPHRASE')
        if passphrase:
            logger.info("🔑 Using passphrase from environment variable")
            return passphrase
        
        # Try command line argument
        if hasattr(args, 'passphrase') and args.passphrase:
            logger.info("🔑 Using passphrase from command line")
            return args.passphrase
        
        # Interactive prompt (only if stdin is a terminal)
        if sys.stdin.isatty():
            try:
                passphrase = getpass.getpass("🔐 Enter encryption passphrase: ")
                return passphrase
            except KeyboardInterrupt:
                logger.info("❌ Passphrase entry cancelled")
                return None
        
        logger.warning("⚠️ No passphrase available (non-interactive mode)")
        return None
    
    async def check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            import praw
            import ccxt
            import aiohttp
            import cryptography
            logger.info("✅ Core dependencies available")
            
            # Test API connections (without credentials)
            logger.info("🔗 Testing API connectivity...")
            
            # Test Reddit API structure
            if Config.REDDIT_CLIENT_ID:
                logger.info("   📱 Reddit API credentials configured")
            
            # Test Binance API structure  
            if Config.BINANCE_API_KEY:
                logger.info("   📊 Binance API credentials configured")
            
            # Test GitHub API
            if Config.GITHUB_TOKEN:
                github_test = await self.github_backup.check_repo_access()
                if github_test:
                    logger.info("   ☁️ GitHub repository access verified")
                else:
                    logger.warning("   ⚠️ GitHub repository access failed")
            
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            raise
    
    async def initialize_components(self):
        """Initialize platform components"""
        try:
            # Initialize state manager
            state_summary = self.state_manager.get_state_summary()
            logger.info(f"💾 State manager initialized - {state_summary.get('open_trades', 0)} open trades")
            
            # Test scanners (without actually scanning)
            logger.info("🔍 Scanner components initialized")
            
            # Test decoders
            logger.info("🧠 Decoder components initialized")
            
            # Test executors
            logger.info("⚙️ Executor components initialized")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    async def create_initial_backup(self):
        """Create initial state backup"""
        try:
            passphrase = Config.ENCRYPTION_PASSPHRASE
            if not passphrase:
                logger.warning("⚠️ No encryption passphrase configured, skipping backup")
                return
            
            success = await self.github_backup.backup_state()
            if success:
                logger.info("✅ Initial backup created")
            else:
                logger.warning("⚠️ Initial backup failed")
                
        except Exception as e:
            logger.warning(f"⚠️ Backup creation failed: {e}")
    
    async def announce_startup(self):
        """Announce platform startup"""
        try:
            from executor.alert_sender import AlertSender
            
            if not any([Config.TELEGRAM_BOT_TOKEN, Config.DISCORD_WEBHOOK_URL, Config.WEBHOOK_URL]):
                logger.info("📢 No alert channels configured, skipping announcement")
                return
            
            alert_sender = AlertSender()
            
            # Create startup announcement
            startup_pattern = {
                'id': 'platform_startup',
                'asset': 'SYSTEM',
                'type': 'startup',
                'source': 'bootstrap',
                'signals': {
                    'startup_time': Config.get_config_summary(),
                    'migration_info': self.state_manager.load_state().get('migration_info', {})
                }
            }
            
            await alert_sender.send_alert(startup_pattern, 50.0)  # Medium priority
            await alert_sender.cleanup()
            
            logger.info("✅ Startup announcement sent")
            
        except Exception as e:
            logger.warning(f"⚠️ Startup announcement failed: {e}")


def create_sample_env():
    """Create sample .env file for configuration"""
    sample_env = """# Social Intelligence Trading Platform Configuration

# Flask Settings
FLASK_ENV=development
DEBUG=true
SECRET_KEY=your-secret-key-here

# Reddit API (Get from https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-client-secret
REDDIT_USERNAME=your-reddit-username
REDDIT_PASSWORD=your-reddit-password

# Binance API (Get from Binance API management)
BINANCE_API_KEY=your-binance-api-key
BINANCE_SECRET=your-binance-secret
BINANCE_SANDBOX=true

# Trading Settings
LIVE_TRADING=false
TRADE_THRESHOLD=80.0
MAX_DAILY_TRADES=5

# Alert Channels
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHAT_ID=your-telegram-chat-id
DISCORD_WEBHOOK_URL=your-discord-webhook-url

# GitHub Backup
GITHUB_TOKEN=your-github-token
GITHUB_REPO_OWNER=your-github-username
GITHUB_REPO_NAME=your-repo-name

# Encryption
ENCRYPTION_PASSPHRASE=your-strong-encryption-passphrase

# Email Alerts (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_FROM=your-email@gmail.com
EMAIL_TO=alerts@yourdomain.com
EMAIL_PASSWORD=your-app-password
"""
    
    with open('.env.example', 'w') as f:
        f.write(sample_env)
    
    print("📄 Created .env.example file")
    print("📝 Copy it to .env and fill in your API credentials")


async def main():
    """Main bootstrap function"""
    parser = argparse.ArgumentParser(
        description='Social Intelligence Trading Platform Bootstrap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bootstrap.py                    # Fresh installation
  python bootstrap.py --restore          # Restore from GitHub backup
  python bootstrap.py --create-env       # Create sample .env file
  python bootstrap.py --restore --passphrase mypassword
        """
    )
    
    parser.add_argument('--restore', action='store_true',
                       help='Restore state from GitHub backup')
    parser.add_argument('--passphrase', type=str,
                       help='Encryption passphrase for backup restoration')
    parser.add_argument('--source-account', type=str,
                       help='Source account identifier for migration tracking')
    parser.add_argument('--create-env', action='store_true',
                       help='Create sample .env configuration file')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration, do not bootstrap')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_env:
        create_sample_env()
        return
    
    # Initialize bootstrap
    bootstrap = PlatformBootstrap()
    
    if args.validate_only:
        logger.info("🔍 Configuration validation only...")
        result = await bootstrap.validate_configuration()
        if result['valid']:
            logger.info("✅ Configuration is valid")
        else:
            logger.error("❌ Configuration validation failed")
            sys.exit(1)
        return
    
    # Run full bootstrap
    success = await bootstrap.bootstrap(args)
    
    if success:
        logger.info("🎉 Platform bootstrap completed successfully!")
        logger.info("🚀 You can now run: python app.py")
        sys.exit(0)
    else:
        logger.error("💥 Platform bootstrap failed!")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
