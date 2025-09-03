"""
GitHub Backup Module
Handles automated state backup to GitHub with encryption
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict
import aiohttp
import json
import base64
from .encryption import StateEncryption
from .state_manager import StateManager

# Try to import database models for enhanced backups
try:
    from flask import current_app
    from models import db, PatternOutcome, AssetMention, Correlation, CausalHypothesis, CausalTest, TradingSignal, BackupRecord
    DATABASE_MODELS_AVAILABLE = True
except ImportError:
    DATABASE_MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)

class GitHubBackup:
    """Automated GitHub backup with encryption"""
    
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN', '')
        self.repo_owner = os.getenv('GITHUB_REPO_OWNER', '')
        self.repo_name = os.getenv('GITHUB_REPO_NAME', '')
        self.branch = os.getenv('GITHUB_BRANCH', 'main')
        
        self.encryption = StateEncryption()
        self.state_manager = StateManager()
        
        self.session = None
        self.api_base = 'https://api.github.com'
    
    async def backup_state(self) -> bool:
        """Backup current state to GitHub"""
        try:
            if not self.validate_config():
                logger.error("GitHub backup configuration invalid")
                return False
            
            # Get encryption passphrase
            passphrase = self.encryption.get_passphrase_from_env()
            if not passphrase:
                logger.error("No encryption passphrase available")
                return False
            
            # Load current state and database data
            current_state = self.state_manager.load_state()
            
            # Add database backup if available
            if DATABASE_MODELS_AVAILABLE:
                try:
                    database_data = await self.export_database_data()
                    current_state['database_backup'] = database_data
                    logger.info("Added database backup to state")
                except Exception as e:
                    logger.warning(f"Could not backup database data: {e}")
            
            # Encrypt state
            encrypted_data = self.encryption.encrypt_data(current_state, passphrase)
            
            # Upload to GitHub
            success = await self.upload_encrypted_state(encrypted_data)
            
            if success:
                # Also backup logs and changelog
                await self.backup_logs()
                await self.backup_changelog()
                logger.info("State backup completed successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error backing up state: {e}")
            return False
    
    async def export_database_data(self) -> Dict:
        """Export database data for backup"""
        if not DATABASE_MODELS_AVAILABLE:
            return {}
            
        try:
            data = {}
            
            with current_app.app_context():
                # Export pattern outcomes
                pattern_outcomes = PatternOutcome.query.all()
                data['pattern_outcomes'] = [outcome.to_dict() for outcome in pattern_outcomes]
                
                # Export asset mentions
                asset_mentions = AssetMention.query.all()
                data['asset_mentions'] = [mention.to_dict() for mention in asset_mentions]
                
                # Export correlations
                correlations = Correlation.query.all()
                data['correlations'] = [corr.to_dict() for corr in correlations]
                
                # Export causal hypotheses
                causal_hypotheses = CausalHypothesis.query.all()
                data['causal_hypotheses'] = [hyp.to_dict() for hyp in causal_hypotheses]
                
                # Export causal tests
                causal_tests = CausalTest.query.all()
                data['causal_tests'] = [test.to_dict() for test in causal_tests]
                
                # Export trading signals
                trading_signals = TradingSignal.query.all()
                data['trading_signals'] = [signal.to_dict() for signal in trading_signals]
                
                # Record this backup
                backup_record = BackupRecord(
                    backup_id=f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    backup_type="FULL",
                    backup_status="SUCCESS"
                )
                db.session.add(backup_record)
                db.session.commit()
                
                logger.info("Database data exported successfully")
                return data
                
        except Exception as e:
            logger.error(f"Error exporting database data: {e}")
            return {}
    
    async def restore_database_data(self, backup_data: Dict):
        """Restore database data from backup"""
        if not DATABASE_MODELS_AVAILABLE or not backup_data:
            return
            
        try:
            with current_app.app_context():
                # Clear existing data
                PatternOutcome.query.delete()
                AssetMention.query.delete()
                Correlation.query.delete()
                CausalHypothesis.query.delete()
                CausalTest.query.delete()
                TradingSignal.query.delete()
                
                # Restore pattern outcomes
                if 'pattern_outcomes' in backup_data:
                    for outcome_data in backup_data['pattern_outcomes']:
                        outcome = PatternOutcome(
                            pattern_id=outcome_data['pattern_id'],
                            outcome=outcome_data['outcome']
                        )
                        db.session.add(outcome)
                
                # Restore asset mentions
                if 'asset_mentions' in backup_data:
                    for mention_data in backup_data['asset_mentions']:
                        mention = AssetMention(
                            asset=mention_data['asset'],
                            mentions=mention_data['mentions']
                        )
                        db.session.add(mention)
                
                # Restore correlations
                if 'correlations' in backup_data:
                    for corr_data in backup_data['correlations']:
                        correlation = Correlation(
                            key=corr_data['key'],
                            value=corr_data['value']
                        )
                        db.session.add(correlation)
                
                # Restore causal hypotheses
                if 'causal_hypotheses' in backup_data:
                    for hyp_data in backup_data['causal_hypotheses']:
                        hypothesis = CausalHypothesis(
                            hypothesis_id=hyp_data['hypothesis_id'],
                            x_variable=hyp_data['x_variable'],
                            y_variable=hyp_data['y_variable'],
                            hypothesis=hyp_data['hypothesis'],
                            granger_p=hyp_data.get('granger_p'),
                            lead_lag_minutes=hyp_data.get('lead_lag_minutes'),
                            effect_size=hyp_data.get('effect_size'),
                            confidence=hyp_data.get('confidence'),
                            regime_dependency=hyp_data.get('regime_dependency'),
                            status=hyp_data.get('status')
                        )
                        db.session.add(hypothesis)
                
                # Restore causal tests
                if 'causal_tests' in backup_data:
                    for test_data in backup_data['causal_tests']:
                        test = CausalTest(
                            test_id=test_data['test_id'],
                            hypothesis_id=test_data['hypothesis_id'],
                            test_type=test_data['test_type'],
                            result=test_data.get('result'),
                            p_value=test_data.get('p_value'),
                            effect_size=test_data.get('effect_size'),
                            sample_size=test_data.get('sample_size')
                        )
                        db.session.add(test)
                
                # Restore trading signals
                if 'trading_signals' in backup_data:
                    for signal_data in backup_data['trading_signals']:
                        signal = TradingSignal(
                            signal_id=signal_data['signal_id'],
                            asset=signal_data['asset'],
                            signal_type=signal_data['signal_type'],
                            strength=signal_data['strength'],
                            confidence=signal_data['confidence'],
                            price_target=signal_data.get('price_target'),
                            stop_loss=signal_data.get('stop_loss'),
                            time_horizon=signal_data.get('time_horizon'),
                            source=signal_data.get('source'),
                            signal_metadata=signal_data.get('signal_metadata'),
                            expires_at=datetime.fromisoformat(signal_data['expires_at']) if signal_data.get('expires_at') else None,
                            is_active=signal_data.get('is_active', True)
                        )
                        db.session.add(signal)
                
                # Record successful restoration
                restore_record = BackupRecord(
                    backup_id=f"restore_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    backup_type="RESTORE",
                    backup_status="SUCCESS"
                )
                db.session.add(restore_record)
                
                db.session.commit()
                logger.info("Database data restored successfully")
                
        except Exception as e:
            logger.error(f"Error restoring database data: {e}")
            if current_app.app_context:
                db.session.rollback()
            raise
    
    async def restore_state(self, passphrase: Optional[str] = None) -> bool:
        """Restore state from GitHub backup"""
        try:
            if not self.validate_config():
                logger.error("GitHub backup configuration invalid")
                return False
            
            # Download encrypted state
            encrypted_data = await self.download_encrypted_state()
            if not encrypted_data:
                return False
            
            # Get passphrase
            if not passphrase:
                passphrase = self.encryption.get_passphrase_from_env()
            
            if not passphrase:
                logger.error("No decryption passphrase available")
                return False
            
            # Decrypt state
            decrypted_state = self.encryption.decrypt_data(encrypted_data, passphrase)
            
            # Restore database backup if available
            if DATABASE_MODELS_AVAILABLE and 'database_backup' in decrypted_state:
                try:
                    await self.restore_database_data(decrypted_state['database_backup'])
                    # Remove database data from state before saving to avoid duplication
                    del decrypted_state['database_backup']
                    logger.info("Restored database data from backup")
                except Exception as e:
                    logger.warning(f"Could not restore database data: {e}")
            
            # Save as current state
            self.state_manager.save_state(decrypted_state)
            
            logger.info("State restored from GitHub backup")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring state: {e}")
            return False
    
    def validate_config(self) -> bool:
        """Validate GitHub configuration"""
        required_vars = [
            ('GITHUB_TOKEN', self.github_token),
            ('GITHUB_REPO_OWNER', self.repo_owner),
            ('GITHUB_REPO_NAME', self.repo_name)
        ]
        
        for var_name, var_value in required_vars:
            if not var_value:
                logger.error(f"Missing required environment variable: {var_name}")
                return False
        
        return True
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if not self.session:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'TradingPlatform-Backup/1.0'
            }
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            )
        
        return self.session
    
    async def upload_encrypted_state(self, encrypted_data: bytes) -> bool:
        """Upload encrypted state data to GitHub"""
        try:
            session = await self.get_session()
            
            # Encode data for GitHub API
            content_b64 = base64.b64encode(encrypted_data).decode()
            
            # Generate filename with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f'state_backup_{timestamp}.enc'
            file_path = f'backups/{filename}'
            
            # Check if file exists to get SHA
            sha = await self.get_file_sha(file_path)
            
            # Prepare commit data
            commit_data = {
                'message': f'Automated state backup - {timestamp}',
                'content': content_b64,
                'branch': self.branch
            }
            
            if sha:
                commit_data['sha'] = sha
            
            # Upload file
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{file_path}"
            
            async with session.put(url, json=commit_data) as response:
                if response.status in [200, 201]:
                    logger.info(f"State uploaded to GitHub: {filename}")
                    
                    # Update latest backup reference
                    await self.update_latest_backup_ref(filename)
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"GitHub upload failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error uploading to GitHub: {e}")
            return False
    
    async def download_encrypted_state(self, filename: Optional[str] = None) -> Optional[bytes]:
        """Download encrypted state from GitHub"""
        try:
            session = await self.get_session()
            
            # Use latest backup if no filename specified
            if not filename:
                filename = await self.get_latest_backup_filename()
                if not filename:
                    logger.error("No backup files found")
                    return None
            
            file_path = f'backups/{filename}'
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{file_path}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    content_b64 = data.get('content', '')
                    
                    # Decode from base64
                    encrypted_data = base64.b64decode(content_b64)
                    
                    logger.info(f"Downloaded backup from GitHub: {filename}")
                    return encrypted_data
                else:
                    logger.error(f"Failed to download from GitHub: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error downloading from GitHub: {e}")
            return None
    
    async def get_file_sha(self, file_path: str) -> Optional[str]:
        """Get SHA of existing file"""
        try:
            session = await self.get_session()
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{file_path}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('sha')
                return None
                
        except Exception as e:
            logger.debug(f"File not found (expected for new files): {e}")
            return None
    
    async def update_latest_backup_ref(self, filename: str) -> bool:
        """Update reference to latest backup"""
        try:
            session = await self.get_session()
            
            ref_data = {
                'latest_backup': filename,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'platform_version': 'v1.2.0'
            }
            
            content_b64 = base64.b64encode(
                json.dumps(ref_data, indent=2).encode()
            ).decode()
            
            ref_path = 'backups/latest.json'
            sha = await self.get_file_sha(ref_path)
            
            commit_data = {
                'message': f'Update latest backup reference to {filename}',
                'content': content_b64,
                'branch': self.branch
            }
            
            if sha:
                commit_data['sha'] = sha
            
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{ref_path}"
            
            async with session.put(url, json=commit_data) as response:
                return response.status in [200, 201]
                
        except Exception as e:
            logger.error(f"Error updating latest backup reference: {e}")
            return False
    
    async def get_latest_backup_filename(self) -> Optional[str]:
        """Get filename of latest backup"""
        try:
            session = await self.get_session()
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/backups/latest.json"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    content_b64 = data.get('content', '')
                    content_str = base64.b64decode(content_b64).decode()
                    ref_data = json.loads(content_str)
                    
                    return ref_data.get('latest_backup')
                    
        except Exception as e:
            logger.error(f"Error getting latest backup filename: {e}")
        
        # Fallback: list backups directory and find latest
        return await self.find_latest_backup_by_listing()
    
    async def find_latest_backup_by_listing(self) -> Optional[str]:
        """Find latest backup by listing directory"""
        try:
            session = await self.get_session()
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/backups"
            
            async with session.get(url) as response:
                if response.status == 200:
                    files = await response.json()
                    
                    backup_files = [
                        f['name'] for f in files 
                        if f['name'].startswith('state_backup_') and f['name'].endswith('.enc')
                    ]
                    
                    if backup_files:
                        # Sort by timestamp in filename
                        backup_files.sort(reverse=True)
                        return backup_files[0]
                        
        except Exception as e:
            logger.error(f"Error listing backup files: {e}")
        
        return None
    
    async def backup_logs(self) -> bool:
        """Backup recent logs to GitHub"""
        try:
            logs_dir = 'logs'
            if not os.path.exists(logs_dir):
                return True  # No logs to backup
            
            # Find recent log files
            recent_logs = []
            for filename in os.listdir(logs_dir):
                if filename.endswith('.log'):
                    filepath = os.path.join(logs_dir, filename)
                    # Only backup logs from last 7 days
                    age_days = (datetime.now() - datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    )).days
                    
                    if age_days <= 7:
                        recent_logs.append(filename)
            
            # Upload each log file
            session = await self.get_session()
            for log_filename in recent_logs:
                try:
                    log_path = os.path.join(logs_dir, log_filename)
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                    
                    content_b64 = base64.b64encode(log_content.encode()).decode()
                    
                    github_path = f'logs/{log_filename}'
                    sha = await self.get_file_sha(github_path)
                    
                    commit_data = {
                        'message': f'Backup log file: {log_filename}',
                        'content': content_b64,
                        'branch': self.branch
                    }
                    
                    if sha:
                        commit_data['sha'] = sha
                    
                    url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{github_path}"
                    
                    async with session.put(url, json=commit_data) as response:
                        if response.status in [200, 201]:
                            logger.debug(f"Uploaded log: {log_filename}")
                        
                except Exception as e:
                    logger.warning(f"Failed to backup log {log_filename}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error backing up logs: {e}")
            return False
    
    async def backup_changelog(self) -> bool:
        """Backup CHANGELOG.md to GitHub"""
        try:
            changelog_path = 'CHANGELOG.md'
            if not os.path.exists(changelog_path):
                return True  # No changelog to backup
            
            with open(changelog_path, 'r') as f:
                changelog_content = f.read()
            
            content_b64 = base64.b64encode(changelog_content.encode()).decode()
            
            session = await self.get_session()
            sha = await self.get_file_sha('CHANGELOG.md')
            
            commit_data = {
                'message': f'Update CHANGELOG.md - {datetime.utcnow().strftime("%Y-%m-%d %H:%M")}',
                'content': content_b64,
                'branch': self.branch
            }
            
            if sha:
                commit_data['sha'] = sha
            
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/CHANGELOG.md"
            
            async with session.put(url, json=commit_data) as response:
                return response.status in [200, 201]
                
        except Exception as e:
            logger.error(f"Error backing up changelog: {e}")
            return False
    
    async def list_backups(self) -> list:
        """List all available backups"""
        try:
            session = await self.get_session()
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/backups"
            
            async with session.get(url) as response:
                if response.status == 200:
                    files = await response.json()
                    
                    backups = []
                    for file_info in files:
                        if file_info['name'].startswith('state_backup_') and file_info['name'].endswith('.enc'):
                            # Extract timestamp from filename
                            timestamp_str = file_info['name'].replace('state_backup_', '').replace('.enc', '')
                            try:
                                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                                backups.append({
                                    'filename': file_info['name'],
                                    'timestamp': timestamp.isoformat() + 'Z',
                                    'size': file_info['size']
                                })
                            except ValueError:
                                continue
                    
                    # Sort by timestamp (newest first)
                    backups.sort(key=lambda x: x['timestamp'], reverse=True)
                    return backups
                    
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
        
        return []
    
    async def cleanup_old_backups(self, keep_count: int = 10) -> bool:
        """Remove old backups, keeping only the most recent ones"""
        try:
            backups = await self.list_backups()
            
            if len(backups) <= keep_count:
                return True  # Nothing to clean up
            
            # Delete old backups
            session = await self.get_session()
            backups_to_delete = backups[keep_count:]
            
            for backup in backups_to_delete:
                try:
                    file_path = f"backups/{backup['filename']}"
                    sha = await self.get_file_sha(file_path)
                    
                    if sha:
                        delete_data = {
                            'message': f"Cleanup old backup: {backup['filename']}",
                            'sha': sha,
                            'branch': self.branch
                        }
                        
                        url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{file_path}"
                        
                        async with session.delete(url, json=delete_data) as response:
                            if response.status == 200:
                                logger.info(f"Deleted old backup: {backup['filename']}")
                            
                except Exception as e:
                    logger.warning(f"Failed to delete backup {backup['filename']}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            return False
    
    async def check_repo_access(self) -> bool:
        """Test GitHub repository access"""
        try:
            session = await self.get_session()
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    logger.info("GitHub repository access verified")
                    return True
                else:
                    logger.error(f"GitHub repository access failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error checking repo access: {e}")
            return False
    
    async def push_source_code(self, commit_message: Optional[str] = None) -> bool:
        """Push entire source code to GitHub for external review"""
        try:
            if not self.validate_config():
                logger.error("GitHub backup configuration invalid")
                return False
            
            session = await self.get_session()
            
            # Default commit message
            if commit_message is None:
                timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                commit_message = f"Code review deployment - {timestamp}"
            
            # List of important files/directories to include
            include_patterns = [
                '*.py',
                'templates/*.html',
                'static/**/*',
                'utils/*.py',
                'scanners/*.py',
                'decoders/*.py',
                'executors/*.py',
                'config.json',
                'permissions.json',
                'requirements.txt',
                'replit.md'
            ]
            
            # Files to exclude
            exclude_patterns = [
                'state.json',
                'state_backups/',
                'logs/',
                '__pycache__/',
                '*.pyc',
                '.env',
                'node_modules/',
                '.git/',
                '.cache/',
                '.pythonlibs/',
                '.uv/',
                'templates/',
                'static/',
                'mazer-theme/'
            ]
            
            uploaded_files = []
            
            # Walk through project directory
            for root, dirs, files in os.walk('.'):
                # Skip excluded directories and all hidden directories
                dirs[:] = [d for d in dirs if not any(
                    d.startswith(pattern.rstrip('/')) for pattern in exclude_patterns
                ) and not d.startswith('.') and d not in ['cache', 'pythonlibs', 'uv']]
                
                for file in files:
                    file_path = os.path.join(root, file).replace('./', '')
                    
                    # Skip excluded files and paths
                    if any(file.endswith(pattern.lstrip('*.')) for pattern in exclude_patterns if pattern.startswith('*.')):
                        continue
                    if any(pattern.rstrip('/') in file_path for pattern in exclude_patterns):
                        continue
                    # Skip any files in hidden directories or cache/lib directories
                    if '/.cache/' in file_path or '/.pythonlibs/' in file_path or '/.uv/' in file_path:
                        continue
                    
                    # Include only matching patterns or important files
                    should_include = (
                        file.endswith('.py') or 
                        file.endswith('.html') or
                        file.endswith('.js') or
                        file.endswith('.css') or
                        file.endswith('.json') or
                        file.endswith('.md') or
                        file.endswith('.txt')
                    )
                    
                    if should_include:
                        try:
                            # Read file content
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Encode for GitHub
                            content_b64 = base64.b64encode(content.encode()).decode()
                            
                            # Get existing file SHA (if any)
                            github_path = file_path
                            sha = await self.get_file_sha(github_path)
                            
                            # Prepare commit
                            commit_data = {
                                'message': commit_message,
                                'content': content_b64,
                                'branch': self.branch
                            }
                            
                            if sha:
                                commit_data['sha'] = sha
                            
                            # Upload to GitHub
                            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{github_path}"
                            
                            async with session.put(url, json=commit_data) as response:
                                if response.status in [200, 201]:
                                    uploaded_files.append(file_path)
                                    logger.debug(f"Uploaded: {file_path}")
                                else:
                                    error_text = await response.text()
                                    logger.warning(f"Failed to upload {file_path}: {response.status} - {error_text}")
                        
                        except Exception as e:
                            logger.warning(f"Error uploading {file_path}: {e}")
            
            logger.info(f"Successfully pushed {len(uploaded_files)} source files to GitHub")
            logger.info(f"Repository: https://github.com/{self.repo_owner}/{self.repo_name}")
            
            return len(uploaded_files) > 0
            
        except Exception as e:
            logger.error(f"Error pushing source code: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
