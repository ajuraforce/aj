"""
State Manager Module
Handles loading, saving, and validation of platform state
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class StateManager:
    """Manages platform state persistence and validation"""
    
    def __init__(self, state_file: str = 'state.json'):
        self.state_file = state_file
        self.backup_dir = 'state_backups'
        self.ensure_backup_dir()
    
    def ensure_backup_dir(self):
        """Ensure backup directory exists"""
        Path(self.backup_dir).mkdir(exist_ok=True)
    
    def load_state(self) -> Dict:
        """Load platform state from file"""
        try:
            if not os.path.exists(self.state_file):
                logger.info("No state file found, creating default state")
                return self.create_default_state()
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Validate state structure
            validated_state = self.validate_and_repair_state(state)
            
            logger.info(f"Loaded state from {self.state_file}")
            return validated_state
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in state file: {e}")
            return self.create_default_state()
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return self.create_default_state()
    
    def save_state(self, state: Dict):
        """Save platform state to file with backup"""
        try:
            # Create backup before saving
            self.backup_current_state()
            
            # Validate state before saving
            validated_state = self.validate_and_repair_state(state)
            
            # Write to temporary file first
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(validated_state, f, indent=2, default=str)
            
            # Atomic replace
            shutil.move(temp_file, self.state_file)
            
            logger.debug(f"State saved to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            # Clean up temp file if it exists
            temp_file = f"{self.state_file}.tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def backup_current_state(self):
        """Create timestamped backup of current state"""
        try:
            if not os.path.exists(self.state_file):
                return
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(self.backup_dir, f"state_{timestamp}.json")
            
            shutil.copy2(self.state_file, backup_file)
            
            # Clean old backups (keep last 10)
            self.cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"Error creating state backup: {e}")
    
    def cleanup_old_backups(self):
        """Keep only the most recent backups"""
        try:
            backup_files = []
            for f in os.listdir(self.backup_dir):
                if f.startswith('state_') and f.endswith('.json'):
                    backup_files.append(os.path.join(self.backup_dir, f))
            
            # Sort by modification time
            backup_files.sort(key=os.path.getmtime, reverse=True)
            
            # Remove old backups (keep 10)
            for old_backup in backup_files[10:]:
                os.remove(old_backup)
                logger.debug(f"Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"Error cleaning old backups: {e}")
    
    def create_default_state(self) -> Dict:
        """Create default state structure"""
        return {
            "last_run_id": datetime.utcnow().isoformat() + "Z",
            "scanner": {
                "sources": ["reddit", "binance", "news"],
                "last_offsets": {
                    "reddit": None,
                    "binance": None,
                    "news": None
                }
            },
            "decoder": {
                "correlation_snapshot": {},
                "recent_alerts": []
            },
            "executor": {
                "open_trades": [],
                "recent_posts": []
            },
            "config_version": "v1.2.0",
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
    
    def validate_and_repair_state(self, state: Dict) -> Dict:
        """Validate state structure and repair if needed"""
        try:
            # Get default state as template
            default_state = self.create_default_state()
            
            # Ensure all required keys exist
            validated_state = self.deep_merge(default_state, state)
            
            # Validate specific structures
            validated_state = self.validate_scanner_state(validated_state)
            validated_state = self.validate_decoder_state(validated_state)
            validated_state = self.validate_executor_state(validated_state)
            
            # Update last_run_id
            validated_state["last_run_id"] = datetime.utcnow().isoformat() + "Z"
            
            return validated_state
            
        except Exception as e:
            logger.error(f"Error validating state: {e}")
            return self.create_default_state()
    
    def deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_scanner_state(self, state: Dict) -> Dict:
        """Validate scanner state structure"""
        try:
            scanner = state.get("scanner", {})
            
            # Ensure sources list
            if not isinstance(scanner.get("sources"), list):
                scanner["sources"] = ["reddit", "binance", "news"]
            
            # Ensure last_offsets dict
            if not isinstance(scanner.get("last_offsets"), dict):
                scanner["last_offsets"] = {"reddit": None, "binance": None, "news": None}
            
            # Ensure all sources have offsets
            for source in scanner["sources"]:
                if source not in scanner["last_offsets"]:
                    scanner["last_offsets"][source] = None
            
            state["scanner"] = scanner
            return state
            
        except Exception as e:
            logger.error(f"Error validating scanner state: {e}")
            return state
    
    def validate_decoder_state(self, state: Dict) -> Dict:
        """Validate decoder state structure"""
        try:
            decoder = state.get("decoder", {})
            
            # Ensure correlation_snapshot is dict
            if not isinstance(decoder.get("correlation_snapshot"), dict):
                decoder["correlation_snapshot"] = {}
            
            # Ensure recent_alerts is list
            if not isinstance(decoder.get("recent_alerts"), list):
                decoder["recent_alerts"] = []
            
            # Limit recent_alerts size
            decoder["recent_alerts"] = decoder["recent_alerts"][-50:]
            
            state["decoder"] = decoder
            return state
            
        except Exception as e:
            logger.error(f"Error validating decoder state: {e}")
            return state
    
    def validate_executor_state(self, state: Dict) -> Dict:
        """Validate executor state structure"""
        try:
            executor = state.get("executor", {})
            
            # Ensure open_trades is list
            if not isinstance(executor.get("open_trades"), list):
                executor["open_trades"] = []
            
            # Ensure recent_posts is list
            if not isinstance(executor.get("recent_posts"), list):
                executor["recent_posts"] = []
            
            # Validate trade structures
            valid_trades = []
            for trade in executor["open_trades"]:
                if isinstance(trade, dict) and self.validate_trade_structure(trade):
                    valid_trades.append(trade)
            executor["open_trades"] = valid_trades
            
            # Limit recent_posts size
            executor["recent_posts"] = executor["recent_posts"][-25:]
            
            state["executor"] = executor
            return state
            
        except Exception as e:
            logger.error(f"Error validating executor state: {e}")
            return state
    
    def validate_trade_structure(self, trade: Dict) -> bool:
        """Validate individual trade structure"""
        required_fields = ['id', 'symbol', 'direction', 'size', 'entry_price', 'status']
        return all(field in trade for field in required_fields)
    
    def get_state_summary(self) -> Dict:
        """Get summary information about current state"""
        try:
            state = self.load_state()
            
            return {
                "last_run_id": state.get("last_run_id"),
                "config_version": state.get("config_version"),
                "scanner_sources": len(state.get("scanner", {}).get("sources", [])),
                "correlation_pairs": len(state.get("decoder", {}).get("correlation_snapshot", {})),
                "recent_alerts": len(state.get("decoder", {}).get("recent_alerts", [])),
                "open_trades": len(state.get("executor", {}).get("open_trades", [])),
                "recent_posts": len(state.get("executor", {}).get("recent_posts", [])),
                "file_size_bytes": os.path.getsize(self.state_file) if os.path.exists(self.state_file) else 0,
                "backup_count": len([f for f in os.listdir(self.backup_dir) 
                                   if f.startswith('state_') and f.endswith('.json')])
            }
            
        except Exception as e:
            logger.error(f"Error getting state summary: {e}")
            return {}
    
    def restore_from_backup(self, backup_filename: Optional[str] = None) -> bool:
        """Restore state from backup"""
        try:
            if backup_filename:
                backup_path = os.path.join(self.backup_dir, backup_filename)
            else:
                # Use most recent backup
                backup_files = []
                for f in os.listdir(self.backup_dir):
                    if f.startswith('state_') and f.endswith('.json'):
                        backup_files.append(os.path.join(self.backup_dir, f))
                
                if not backup_files:
                    logger.error("No backup files found")
                    return False
                
                backup_files.sort(key=os.path.getmtime, reverse=True)
                backup_path = backup_files[0]
            
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Validate backup before restoring
            with open(backup_path, 'r') as f:
                backup_state = json.load(f)
            
            validated_backup = self.validate_and_repair_state(backup_state)
            
            # Save as current state
            self.save_state(validated_backup)
            
            logger.info(f"State restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def export_state(self, export_path: str) -> bool:
        """Export current state to specified path"""
        try:
            current_state = self.load_state()
            
            with open(export_path, 'w') as f:
                json.dump(current_state, f, indent=2, default=str)
            
            logger.info(f"State exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting state: {e}")
            return False
    
    def import_state(self, import_path: str) -> bool:
        """Import state from specified path"""
        try:
            if not os.path.exists(import_path):
                logger.error(f"Import file not found: {import_path}")
                return False
            
            with open(import_path, 'r') as f:
                imported_state = json.load(f)
            
            # Validate imported state
            validated_state = self.validate_and_repair_state(imported_state)
            
            # Save as current state
            self.save_state(validated_state)
            
            logger.info(f"State imported from: {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing state: {e}")
            return False
