#!/usr/bin/env python3
"""
GitHub Deployment Script
Pushes all source code and database data to GitHub repository
"""

import os
import sys
import json
import subprocess
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class GitHubDeployer:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_owner = os.getenv('GITHUB_REPO_OWNER') 
        self.github_repo_name = os.getenv('GITHUB_REPO_NAME')
        self.branch = 'main'
        
        if not all([self.github_token, self.github_owner, self.github_repo_name]):
            raise ValueError("Missing required GitHub credentials in environment variables")
        
        self.repo_url = f"https://{self.github_token}@github.com/{self.github_owner}/{self.github_repo_name}.git"
        
    def run_command(self, command, cwd=None):
        """Run shell command and return result"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd,
                capture_output=True, 
                text=True, 
                check=True
            )
            logger.info(f"✓ {command}")
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {command}")
            logger.error(f"Error: {e.stderr}")
            return None

    def setup_git_config(self):
        """Configure git with GitHub credentials"""
        logger.info("Setting up Git configuration...")
        
        # Set git credentials
        self.run_command(f'git config --global user.name "{self.github_owner}"')
        self.run_command(f'git config --global user.email "{self.github_owner}@users.noreply.github.com"')
        
        # Configure credential helper
        self.run_command('git config --global credential.helper store')
        
        return True

    def export_database_data(self):
        """Export database data to JSON files"""
        logger.info("Exporting database data...")
        
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
                    'database_url': 'postgresql://[REDACTED]',  # Don't expose actual URL
                    'tables': {}
                }
                
                for table_name, model_class in tables.items():
                    try:
                        records = model_class.query.all()
                        backup_data['tables'][table_name] = [record.to_dict() for record in records]
                        logger.info(f"✓ Exported {len(records)} records from {table_name}")
                    except Exception as e:
                        logger.warning(f"Could not export {table_name}: {e}")
                        backup_data['tables'][table_name] = []
                
                # Save complete backup
                backup_file = backup_dir / f"database_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(backup_data, f, indent=2, default=str)
                
                # Also save a latest backup
                with open(backup_dir / "latest_backup.json", 'w') as f:
                    json.dump(backup_data, f, indent=2, default=str)
                
                logger.info(f"✓ Database backup saved to {backup_file}")
                return True
                
        except Exception as e:
            logger.error(f"Database export failed: {e}")
            return False

    def create_repository_structure(self):
        """Create proper repository structure"""
        logger.info("Creating repository structure...")
        
        # Create .gitignore if it doesn't exist
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/

# Flask
instance/
.webassets-cache

# Environment variables
.env
.env.local
.env.production

# Logs
*.log
logs/

# Database
*.db
*.db-journal

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
state_backups/
temp/
.replit
replit.nix
poetry.lock
uv.lock
"""
        
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content.strip())
        
        logger.info("✓ Repository structure created")
        return True

    def initialize_repository(self):
        """Initialize git repository and connect to GitHub"""
        logger.info("Initializing Git repository...")
        
        # Initialize git if not already initialized
        if not os.path.exists('.git'):
            self.run_command('git init')
            self.run_command(f'git remote add origin {self.repo_url}')
        else:
            # Update remote URL to use token
            self.run_command(f'git remote set-url origin {self.repo_url}')
        
        return True

    def commit_and_push(self):
        """Stage, commit, and push all changes"""
        logger.info("Committing and pushing to GitHub...")
        
        # Add all files
        self.run_command('git add .')
        
        # Check if there are changes to commit
        result = self.run_command('git diff --staged --quiet')
        if result is None:  # Command failed, meaning there are changes
            # Commit changes
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            commit_message = f"Complete platform deployment with database - {timestamp}"
            
            commit_result = self.run_command(f'git commit -m "{commit_message}"')
            if commit_result is not None:
                # Push to GitHub
                push_result = self.run_command(f'git push -u origin {self.branch}')
                if push_result is not None:
                    logger.info("✅ Successfully pushed to GitHub!")
                    return True
                else:
                    logger.error("Failed to push to GitHub")
                    return False
            else:
                logger.error("Failed to commit changes")
                return False
        else:
            logger.info("No changes to commit")
            return True

    async def run_deployment(self):
        """Run the complete deployment process"""
        logger.info("🚀 Starting GitHub deployment...")
        logger.info(f"Repository: {self.github_owner}/{self.github_repo_name}")
        
        steps = [
            ("Setup Git configuration", self.setup_git_config),
            ("Export database data", self.export_database_data),
            ("Create repository structure", self.create_repository_structure),
            ("Initialize repository", self.initialize_repository),
            ("Commit and push changes", self.commit_and_push)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 {step_name}...")
            success = step_func()
            if not success:
                logger.error(f"❌ {step_name} failed")
                return False
            logger.info(f"✅ {step_name} completed")
        
        logger.info("🎉 GitHub deployment completed successfully!")
        logger.info(f"🔗 Repository: https://github.com/{self.github_owner}/{self.github_repo_name}")
        return True

def main():
    """Main deployment function"""
    try:
        deployer = GitHubDeployer()
        success = asyncio.run(deployer.run_deployment())
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()