"""
Database Models for AJxAI Trading Platform
SQLAlchemy models for PostgreSQL integration
"""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from datetime import datetime
import json

db = SQLAlchemy()

class PatternOutcome(db.Model):
    __tablename__ = 'pattern_outcomes'
    
    id = Column(Integer, primary_key=True)
    pattern_id = Column(String(255), unique=True, nullable=False)
    outcome = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f'<PatternOutcome {self.pattern_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'pattern_id': self.pattern_id,
            'outcome': self.outcome,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class AssetMention(db.Model):
    __tablename__ = 'asset_mentions'
    
    id = Column(Integer, primary_key=True)
    asset = Column(String(50), nullable=False)
    mentions = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f'<AssetMention {self.asset}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'asset': self.asset,
            'mentions': self.mentions,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class Correlation(db.Model):
    __tablename__ = 'correlations'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(255), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f'<Correlation {self.key}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class CausalHypothesis(db.Model):
    __tablename__ = 'causal_hypotheses'
    
    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(String(255), unique=True, nullable=False)
    x_variable = Column(String(255), nullable=False)
    y_variable = Column(String(255), nullable=False)
    hypothesis = Column(Text, nullable=False)
    granger_p = Column(Float)
    lead_lag_minutes = Column(Float)
    effect_size = Column(Float)
    confidence = Column(Float)
    regime_dependency = Column(JSON)
    status = Column(String(50))
    created_at = Column(DateTime, default=func.now())
    last_tested = Column(DateTime)
    
    def __repr__(self):
        return f'<CausalHypothesis {self.hypothesis_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'hypothesis_id': self.hypothesis_id,
            'x_variable': self.x_variable,
            'y_variable': self.y_variable,
            'hypothesis': self.hypothesis,
            'granger_p': self.granger_p,
            'lead_lag_minutes': self.lead_lag_minutes,
            'effect_size': self.effect_size,
            'confidence': self.confidence,
            'regime_dependency': self.regime_dependency,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_tested': self.last_tested.isoformat() if self.last_tested else None
        }

class CausalTest(db.Model):
    __tablename__ = 'causal_tests'
    
    id = Column(Integer, primary_key=True)
    test_id = Column(String(255), unique=True, nullable=False)
    hypothesis_id = Column(String(255), nullable=False)
    test_type = Column(String(100), nullable=False)
    result = Column(Text)
    p_value = Column(Float)
    effect_size = Column(Float)
    sample_size = Column(Integer)
    test_date = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f'<CausalTest {self.test_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'test_id': self.test_id,
            'hypothesis_id': self.hypothesis_id,
            'test_type': self.test_type,
            'result': self.result,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'sample_size': self.sample_size,
            'test_date': self.test_date.isoformat() if self.test_date else None
        }

class TradingSignal(db.Model):
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    signal_id = Column(String(255), unique=True, nullable=False)
    asset = Column(String(50), nullable=False)
    signal_type = Column(String(50), nullable=False)  # 'BUY', 'SELL', 'HOLD'
    strength = Column(Float, nullable=False)  # 0-100
    confidence = Column(Float, nullable=False)  # 0-100
    price_target = Column(Float)
    stop_loss = Column(Float)
    time_horizon = Column(String(50))  # 'SHORT', 'MEDIUM', 'LONG'
    source = Column(String(100))  # 'PATTERN', 'AI', 'SENTIMENT', etc.
    signal_metadata = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f'<TradingSignal {self.signal_id} {self.asset} {self.signal_type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'signal_id': self.signal_id,
            'asset': self.asset,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'confidence': self.confidence,
            'price_target': self.price_target,
            'stop_loss': self.stop_loss,
            'time_horizon': self.time_horizon,
            'source': self.source,
            'signal_metadata': self.signal_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active
        }

class Pattern(db.Model):
    __tablename__ = 'patterns'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now())
    asset = Column(String(20), nullable=False)
    type = Column(String(50), nullable=False)
    confidence = Column(Float, default=0.0)
    signals = Column(Text)
    price = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f'<Pattern {self.asset} {self.type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'asset': self.asset,
            'type': self.type,
            'confidence': self.confidence,
            'signals': self.signals,
            'price': self.price,
            'volume': self.volume,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class BackupRecord(db.Model):
    __tablename__ = 'backup_records'
    
    id = Column(Integer, primary_key=True)
    backup_id = Column(String(255), unique=True, nullable=False)
    backup_type = Column(String(50), nullable=False)  # 'STATE', 'DATABASE', 'FULL'
    file_path = Column(String(500))
    github_commit_sha = Column(String(255))
    file_size = Column(Integer)
    compression_ratio = Column(Float)
    backup_status = Column(String(50), default='PENDING')  # 'PENDING', 'SUCCESS', 'FAILED'
    error_message = Column(Text)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    
    def __repr__(self):
        return f'<BackupRecord {self.backup_id} {self.backup_status}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'backup_id': self.backup_id,
            'backup_type': self.backup_type,
            'file_path': self.file_path,
            'github_commit_sha': self.github_commit_sha,
            'file_size': self.file_size,
            'compression_ratio': self.compression_ratio,
            'backup_status': self.backup_status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }