"""
Post-Mortem & Learning
Evaluates decisions after their horizon and updates weights
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class Verdicts:
    def __init__(self, db_path='patterns.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()

    def _init_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS verdicts (
              decision_id TEXT PRIMARY KEY,
              result TEXT,
              pnl REAL,
              timestamp TEXT
            )
        ''')
        self.conn.commit()

    def evaluate(self, decision: Dict, market_return: float) -> Dict:
        """
        Evaluate a decision:
        - result: 'TP' if correct, 'SL' if stopped out, else 'no_move'
        - pnl: market_return*size_pct
        """
        size_pct = decision.get('risk', {}).get('size_pct', 0)
        pnl = market_return * size_pct * 100  # percentage
        
        if pnl > 0:
            result = 'TP'
        elif pnl < 0:
            result = 'SL'
        else:
            result = 'no_move'
        
        # Persist verdict
        self.conn.execute('''
            INSERT OR REPLACE INTO verdicts VALUES (?, ?, ?, ?)
        ''', (decision['id'], result, pnl, datetime.now().isoformat()))
        self.conn.commit()
        
        # Return for feedback loops
        return {'decision_id': decision['id'], 'result': result, 'pnl': pnl}

    def get_performance_metrics(self) -> Dict:
        """
        Get overall performance metrics from all verdicts
        """
        cursor = self.conn.execute('''
            SELECT 
                COUNT(*) as total_decisions,
                SUM(CASE WHEN result = 'TP' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN result = 'SL' THEN 1 ELSE 0 END) as failed,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl
            FROM verdicts
        ''')
        
        row = cursor.fetchone()
        if row and row[0] > 0:
            return {
                'total_decisions': row[0],
                'success_rate': row[1] / row[0] if row[0] > 0 else 0,
                'failure_rate': row[2] / row[0] if row[0] > 0 else 0,
                'avg_pnl': row[3] or 0,
                'total_pnl': row[4] or 0
            }
        
        return {
            'total_decisions': 0,
            'success_rate': 0,
            'failure_rate': 0,
            'avg_pnl': 0,
            'total_pnl': 0
        }