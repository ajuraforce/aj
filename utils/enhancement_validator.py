"""
Enhancement Validation Module

Validates and monitors enhancement features
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class EnhancementValidator:
    """Validates and monitors enhancement features"""
    
    def __init__(self):
        self.validation_results = {}
        self.last_validation = None
    
    def validate_feature_flags(self) -> Dict:
        """Validate feature flag configuration"""
        try:
            with open('permissions.json', 'r') as f:
                permissions = json.load(f)
            
            feature_flags = permissions.get('feature_flags', {})
            enhancement_settings = permissions.get('enhancement_settings', {})
            
            validation = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Validate adaptive learning settings
            if feature_flags.get('experimental_features', {}).get('adaptive_learning', False):
                adaptive_settings = enhancement_settings.get('adaptive_learning', {})
                if adaptive_settings.get('learning_rate', 0) <= 0:
                    validation['errors'].append("Invalid learning rate for adaptive learning")
                    validation['valid'] = False
            
            # Validate backtesting settings
            if feature_flags.get('experimental_features', {}).get('backtesting_enabled', False):
                backtest_settings = enhancement_settings.get('backtesting', {})
                if backtest_settings.get('historical_periods', 0) < 50:
                    validation['warnings'].append("Low historical periods for backtesting may reduce accuracy")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating feature flags: {e}")
            return {'valid': False, 'errors': [str(e)], 'warnings': []}
    
    def validate_system_readiness(self) -> Dict:
        """Validate system readiness for enhancements"""
        try:
            readiness = {
                'ready': True,
                'checks': {}
            }
            
            # Check database availability
            try:
                import sqlite3
                conn = sqlite3.connect('patterns.db')
                conn.execute('SELECT 1')
                conn.close()
                readiness['checks']['database'] = True
            except Exception:
                readiness['checks']['database'] = False
                readiness['ready'] = False
            
            # Check permissions file
            try:
                with open('permissions.json', 'r') as f:
                    json.load(f)
                readiness['checks']['permissions'] = True
            except Exception:
                readiness['checks']['permissions'] = False
                readiness['ready'] = False
            
            return readiness
            
        except Exception as e:
            logger.error(f"Error validating system readiness: {e}")
            return {'ready': False, 'checks': {}, 'error': str(e)}

def validate_enhancements() -> Dict:
    """Standalone function to validate all enhancements"""
    validator = EnhancementValidator()
    
    results = {
        'feature_flags': validator.validate_feature_flags(),
        'system_readiness': validator.validate_system_readiness(),
        'validation_timestamp': datetime.now().isoformat()
    }
    
    results['overall_valid'] = (
        results['feature_flags']['valid'] and 
        results['system_readiness']['ready']
    )
    
    return results