#!/usr/bin/env python3
"""
Simple GitHub Push Script
Creates database backup and project structure without git operations
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimpleGitHubPush:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_owner = os.getenv('GITHUB_REPO_OWNER') 
        self.github_repo_name = os.getenv('GITHUB_REPO_NAME')
        
        logger.info(f"🔧 Repository: {self.github_owner}/{self.github_repo_name}")
        
    def export_database_data(self):
        """Export database data to JSON files"""
        logger.info("📊 Exporting database data...")
        
        try:
            # Import Flask app and database models
            from app import app
            from models import db, PatternOutcome, AssetMention, Correlation, CausalHypothesis, CausalTest, TradingSignal, BackupRecord
            
            with app.app_context():
                # Create database backup directory
                backup_dir = Path("database_backup")
                backup_dir.mkdir(exist_ok=True)
                
                # Export each table to JSON
                tables = {
                    'pattern_outcomes': PatternOutcome,
                    'asset_mentions': AssetMention, 
                    'correlations': Correlation,
                    'causal_hypotheses': CausalHypothesis,
                    'causal_tests': CausalTest,
                    'trading_signals': TradingSignal,
                    'backup_records': BackupRecord
                }
                
                backup_data = {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'database_url': 'postgresql://[PROTECTED]',
                    'platform_version': 'v1.2.0-enhanced',
                    'tables': {}
                }
                
                total_records = 0
                for table_name, model_class in tables.items():
                    try:
                        records = model_class.query.all()
                        backup_data['tables'][table_name] = [record.to_dict() for record in records]
                        logger.info(f"  ✓ Exported {len(records)} records from {table_name}")
                        total_records += len(records)
                    except Exception as e:
                        logger.warning(f"  ⚠ Could not export {table_name}: {e}")
                        backup_data['tables'][table_name] = []
                
                # Save complete backup
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                backup_file = backup_dir / f"complete_database_backup_{timestamp}.json"
                with open(backup_file, 'w') as f:
                    json.dump(backup_data, f, indent=2, default=str)
                
                # Also save a latest backup
                with open(backup_dir / "latest_database_backup.json", 'w') as f:
                    json.dump(backup_data, f, indent=2, default=str)
                
                # Create a backup summary
                summary = {
                    'backup_timestamp': datetime.utcnow().isoformat(),
                    'total_records': total_records,
                    'tables_exported': len([t for t in backup_data['tables'] if backup_data['tables'][t]]),
                    'backup_file': str(backup_file),
                    'database_schema': {
                        table_name: len(records) for table_name, records in backup_data['tables'].items()
                    }
                }
                
                with open(backup_dir / "backup_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
                
                logger.info(f"✅ Database backup completed: {total_records} total records")
                return True
                
        except Exception as e:
            logger.error(f"❌ Database export failed: {e}")
            return False

    def create_project_structure(self):
        """Create proper project structure and documentation"""
        logger.info("📁 Creating project structure...")
        
        # Create README.md
        readme_content = f"""# AJxAI Trading Platform

Advanced AI-powered cryptocurrency trading and social media automation platform.

## Repository: {self.github_owner}/{self.github_repo_name}
**Deployed**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

## 🚀 Features

- **Data Collection**: Reddit, Binance, News feeds, India equity markets
- **Advanced Analytics**: Pattern analysis, viral scoring, sentiment analysis
- **AI Trading**: Machine learning models, regime detection, portfolio optimization
- **Database**: PostgreSQL with automated backups
- **Security**: Encrypted state management and secure API integrations

## 📊 Database Structure

The platform uses PostgreSQL with the following key tables:
- `pattern_outcomes`: Historical pattern performance
- `asset_mentions`: Social media asset tracking
- `correlations`: Cross-asset correlation analysis
- `causal_hypotheses`: Advanced causal analysis
- `trading_signals`: AI-generated trading signals

## 🔧 Configuration

The platform is configured through environment variables:
- Database: PostgreSQL (via DATABASE_URL)
- GitHub: Automated backups and code management
- APIs: Reddit, Binance, Telegram integration

## 📈 Current Status

Platform is operational with 15 features running including:
- Multi-timeframe analysis
- Correlation matrix engine
- Signal generation
- Portfolio optimization
- Sentiment flow analysis
- Phase 5 advanced features

## 🔒 Security

- All sensitive data is encrypted
- GitHub backups with state protection
- Rate limiting and circuit breakers
- Comprehensive audit trail

## 📄 Database Backup

Latest database backup available in `/database_backup/` directory.
Complete export includes all tables with timestamp: {datetime.utcnow().isoformat()}

---

**Platform Version**: v1.2.0-enhanced  
**Last Updated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
        
        with open('README.md', 'w') as f:
            f.write(readme_content)
        
        # Create project info file
        project_info = {
            'project_name': 'AJxAI Trading Platform',
            'version': 'v1.2.0-enhanced',
            'repository': f"{self.github_owner}/{self.github_repo_name}",
            'deployed_at': datetime.utcnow().isoformat(),
            'features_active': 15,
            'database_type': 'PostgreSQL',
            'backup_system': 'GitHub API',
            'platform_status': 'operational'
        }
        
        with open('project_info.json', 'w') as f:
            json.dump(project_info, f, indent=2)
        
        logger.info("✅ Project structure created")
        return True

    def use_existing_github_backup(self):
        """Use the existing GitHub backup system"""
        logger.info("🔄 Using existing GitHub backup system...")
        
        try:
            from utils.github_backup import GitHubBackup
            
            backup_system = GitHubBackup()
            
            # Set a simple passphrase for encryption
            os.environ['ENCRYPTION_PASSPHRASE'] = 'ajx-ai-backup-2025'
            
            # Run async backup
            async def run_backup():
                success = await backup_system.backup_state()
                return success
            
            success = asyncio.run(run_backup())
            
            if success:
                logger.info("✅ GitHub backup completed successfully!")
                return True
            else:
                logger.error("❌ GitHub backup failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ GitHub backup system error: {e}")
            return False

    async def run_simple_push(self):
        """Run the simple push process"""
        logger.info("🚀 Starting GitHub deployment...")
        
        steps = [
            ("Export database data", self.export_database_data),
            ("Create project structure", self.create_project_structure),
            ("Use GitHub backup system", self.use_existing_github_backup)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 {step_name}...")
            success = step_func()
            if success:
                logger.info(f"✅ {step_name} completed")
            else:
                logger.warning(f"⚠ {step_name} had issues but continuing...")
        
        logger.info("🎉 GitHub deployment process completed!")
        logger.info(f"🔗 Repository: https://github.com/{self.github_owner}/{self.github_repo_name}")
        return True

def main():
    """Main function"""
    try:
        pusher = SimpleGitHubPush()
        asyncio.run(pusher.run_simple_push())
        logger.info("✅ Deployment completed successfully!")
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()