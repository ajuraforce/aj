"""
Decision Policy v1
Aggregates signals into one auditable action with confidence and cooldowns
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DecisionPolicy:
    def __init__(self, db_path='patterns.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()
        self.cooldowns = {}  # {hypothesis_id: next_allowed_time}

    def _init_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
              id TEXT PRIMARY KEY,
              asset TEXT,
              view TEXT,
              rationale TEXT,
              plan TEXT,
              risk TEXT,
              confidence REAL,
              timestamp TEXT
            )
        ''')
        self.conn.commit()

    def decide(self, causal_cards: List[Dict], regime: Dict,
               sentiment_score: float, micro_score: float,
               cooldown_minutes: int = 60) -> Dict:
        """
        Returns a decision JSON and persists it if not in cooldown.
        """
        # Weighted score
        if not causal_cards:
            return {'action': 'ignore', 'reason': 'no signals'}

        avg_causal_conf = sum(c['confidence'] for c in causal_cards) / len(causal_cards)
        regime_score = regime.get('confidence', 0.5)
        score = 0.4*avg_causal_conf + 0.25*regime_score + 0.2*sentiment_score + 0.15*micro_score

        # Determine action
        if score >= 0.7:
            action = 'paper_trade'
        elif score >= 0.55:
            action = 'alert_only'
        else:
            action = 'ignore'

        # Build rationale
        rationale = [f"{c['hypothesis']} (conf {c['confidence']:.2f})" for c in causal_cards]
        rationale.append(f"Regime: {regime.get('risk')}/{regime.get('volatility')} (conf {regime_score:.2f})")

        # Plan
        plan = {
            'entry': 'market',
            'tp': 1.2,
            'sl': 0.6,
            'horizon_min': 120
        }

        # Risk
        risk = {'size_pct': min(1.0, score), 'max_dd_stop': 2.0}

        decision_id = f"dec_{datetime.now().strftime('%Y%m%d%H%M%S')}_{int(score*100)}"
        now = datetime.now()
        # Cooldown check
        if decision_id in self.cooldowns and now < self.cooldowns[decision_id]:
            logger.info("Decision in cooldown, ignoring")
            return {'action': 'ignore', 'reason': 'cooldown'}

        # Persist decision
        self.conn.execute('''
            INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (decision_id, causal_cards[0]['hypothesis'].split('->')[-1].strip() if causal_cards else 'UNKNOWN',
              'LONG' if score>0 else 'SHORT',
              json.dumps(rationale), json.dumps(plan),
              json.dumps(risk), score, now.isoformat()))
        self.conn.commit()

        # Set cooldown
        self.cooldowns[decision_id] = now + timedelta(minutes=cooldown_minutes)

        return {
            'id': decision_id,
            'action': action,
            'view': 'LONG' if score>0 else 'SHORT',
            'confidence': score,
            'rationale': rationale,
            'plan': plan,
            'risk': risk
        }