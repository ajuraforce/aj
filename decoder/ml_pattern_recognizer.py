"""
Machine Learning Pattern Recognition Engine
Uses ML algorithms to detect complex market patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
import asyncio
import logging
import pickle
import os

logger = logging.getLogger(__name__)

class MLPatternRecognizer:
    def __init__(self, config: dict, db_path: str):
        self.config = config
        self.db_path = db_path
        
        # ML model storage
        self.models = {}
        self.scalers = {}
        self.model_path = "models"
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Pattern detection parameters
        self.anomaly_threshold = self.config.get('ml_pattern', {}).get('anomaly_threshold', 0.6)
        self.min_training_samples = 100
        self.feature_window = 20
        
    def extract_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical analysis features for ML"""
        try:
            if data.empty or len(data) < self.feature_window:
                return pd.DataFrame()
            
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            features['volatility'] = features['returns'].rolling(window=10).std()
            
            # Moving averages
            features['sma_5'] = data['close'].rolling(window=5).mean()
            features['sma_20'] = data['close'].rolling(window=min(20, len(data))).mean()
            features['price_to_sma5'] = data['close'] / features['sma_5']
            features['price_to_sma20'] = data['close'] / features['sma_20']
            
            # Volume features
            if 'volume' in data.columns:
                features['volume_sma'] = data['volume'].rolling(window=10).mean()
                features['volume_ratio'] = data['volume'] / features['volume_sma']
                features['price_volume_trend'] = (features['returns'] * features['volume_ratio']).rolling(window=5).mean()
            else:
                features['volume_ratio'] = 1.0
                features['price_volume_trend'] = features['returns'].rolling(window=5).mean()
            
            # Momentum indicators
            features['rsi'] = self.calculate_rsi(data['close'])
            features['momentum'] = data['close'] / data['close'].shift(10) - 1
            
            # Price patterns
            features['high_low_ratio'] = (data['high'] - data['low']) / data['close'] if 'high' in data.columns else 0
            features['open_close_ratio'] = (data['close'] - data['open']) / data['open'] if 'open' in data.columns else 0
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volatility_lag_{lag}'] = features['volatility'].shift(lag)
            
            # Remove NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting technical features: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # Avoid division by zero
            rs = gain / (loss + 1e-8)  # Add small epsilon to prevent division by zero
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def prepare_training_data(self, symbols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for pattern recognition"""
        try:
            all_features = []
            all_labels = []
            
            for symbol in symbols:
                # Get market data
                conn = sqlite3.connect(self.db_path)
                query = """
                SELECT timestamp, close, volume, open, high, low
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 500
                """
                df = pd.read_sql_query(query, conn, params=[symbol])
                conn.close()
                
                if df.empty or len(df) < self.min_training_samples:
                    continue
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp')
                
                # Extract features
                features = self.extract_technical_features(df)
                
                if features.empty:
                    continue
                
                # Create labels (future returns for supervised learning)
                future_returns = df['close'].shift(-5) / df['close'] - 1  # 5-period forward return
                
                # Classify returns
                labels = pd.Series(index=features.index, dtype=int)
                labels[future_returns > 0.02] = 2  # Strong bullish
                labels[(future_returns > 0.005) & (future_returns <= 0.02)] = 1  # Bullish
                labels[(future_returns >= -0.005) & (future_returns <= 0.005)] = 0  # Neutral
                labels[(future_returns >= -0.02) & (future_returns < -0.005)] = -1  # Bearish
                labels[future_returns < -0.02] = -2  # Strong bearish
                
                # Remove last few rows (no future data)
                valid_features = features.iloc[:-10]
                valid_labels = labels.iloc[:-10]
                
                if len(valid_features) > 0:
                    all_features.append(valid_features)
                    all_labels.append(valid_labels)
            
            if not all_features:
                return pd.DataFrame(), pd.Series()
            
            # Combine all data
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_labels = pd.concat(all_labels, ignore_index=True)
            
            # Remove rows with NaN labels
            valid_mask = ~combined_labels.isna()
            combined_features = combined_features[valid_mask]
            combined_labels = combined_labels[valid_mask]
            
            return combined_features, combined_labels
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train_pattern_classifier(self, symbols: List[str]) -> Dict:
        """Train ML classifier for pattern recognition"""
        try:
            # Prepare training data
            features, labels = self.prepare_training_data(symbols)
            
            if features.empty or len(features) < self.min_training_samples:
                return {'success': False, 'reason': 'Insufficient training data'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest classifier
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = rf_model.score(X_train_scaled, y_train)
            test_score = rf_model.score(X_test_scaled, y_test)
            
            # Save model and scaler
            model_file = os.path.join(self.model_path, 'pattern_classifier.pkl')
            scaler_file = os.path.join(self.model_path, 'pattern_scaler.pkl')
            
            with open(model_file, 'wb') as f:
                pickle.dump(rf_model, f)
            
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Store in memory
            self.models['pattern_classifier'] = rf_model
            self.scalers['pattern_classifier'] = scaler
            
            return {
                'success': True,
                'train_score': train_score,
                'test_score': test_score,
                'training_samples': len(X_train),
                'features_count': X_train.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Error training pattern classifier: {e}")
            return {'success': False, 'reason': str(e)}
    
    def train_anomaly_detector(self, symbols: List[str]) -> Dict:
        """Train anomaly detection model"""
        try:
            # Prepare data for anomaly detection
            all_features = []
            
            for symbol in symbols[:5]:  # Use top 5 symbols
                conn = sqlite3.connect(self.db_path)
                query = """
                SELECT timestamp, close, volume, open, high, low
                FROM market_data
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 200
                """
                df = pd.read_sql_query(query, conn, params=[symbol])
                conn.close()
                
                if df.empty:
                    continue
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp')
                
                features = self.extract_technical_features(df)
                if not features.empty:
                    all_features.append(features)
            
            if not all_features:
                return {'success': False, 'reason': 'No data for anomaly detection'}
            
            # Combine features
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_features = combined_features.dropna()
            
            if len(combined_features) < 50:
                return {'success': False, 'reason': 'Insufficient data for anomaly detection'}
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(combined_features)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_estimators=100
            )
            iso_forest.fit(scaled_features)
            
            # Save model and scaler
            model_file = os.path.join(self.model_path, 'anomaly_detector.pkl')
            scaler_file = os.path.join(self.model_path, 'anomaly_scaler.pkl')
            
            with open(model_file, 'wb') as f:
                pickle.dump(iso_forest, f)
            
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Store in memory
            self.models['anomaly_detector'] = iso_forest
            self.scalers['anomaly_detector'] = scaler
            
            return {
                'success': True,
                'training_samples': len(scaled_features),
                'features_count': scaled_features.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            return {'success': False, 'reason': str(e)}
    
    def load_models(self) -> bool:
        """Load pre-trained models from disk"""
        try:
            model_files = {
                'pattern_classifier': ('pattern_classifier.pkl', 'pattern_scaler.pkl'),
                'anomaly_detector': ('anomaly_detector.pkl', 'anomaly_scaler.pkl')
            }
            
            loaded_count = 0
            for model_name, (model_file, scaler_file) in model_files.items():
                model_path = os.path.join(self.model_path, model_file)
                scaler_path = os.path.join(self.model_path, scaler_file)
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    
                    with open(scaler_path, 'rb') as f:
                        self.scalers[model_name] = pickle.load(f)
                    
                    loaded_count += 1
            
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_pattern(self, symbol: str) -> Dict:
        """Predict pattern for a single symbol"""
        try:
            # Load models if not in memory
            if 'pattern_classifier' not in self.models:
                if not self.load_models():
                    return {'prediction': 'NEUTRAL', 'confidence': 0.0, 'error': 'No trained model available'}
            
            # Get recent data
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT timestamp, close, volume, open, high, low
            FROM market_data
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 50
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if df.empty or len(df) < self.feature_window:
                return {'prediction': 'NEUTRAL', 'confidence': 0.0, 'error': 'Insufficient data'}
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')
            
            # Extract features
            features = self.extract_technical_features(df)
            
            # Online update: add z-score of returns as new feature
            returns = features.get('returns', pd.Series())
            if not returns.empty:
                z = (returns.iloc[-1] - returns.mean()) / (returns.std() + 1e-8)
                features['returns_zscore'] = z
                # Append to latest_features
            
            if features.empty:
                return {'prediction': 'NEUTRAL', 'confidence': 0.0, 'error': 'Feature extraction failed'}
            
            # Use latest features
            latest_features = features.iloc[-1:].values.reshape(1, -1)
            
            # Scale features
            scaler = self.scalers['pattern_classifier']
            scaled_features = scaler.transform(latest_features)
            
            # Make prediction
            model = self.models['pattern_classifier']
            prediction = model.predict(scaled_features)[0]
            probabilities = model.predict_proba(scaled_features)[0]
            confidence = max(probabilities)
            
            # Convert numeric prediction to label
            prediction_labels = {
                -2: 'STRONG_BEARISH',
                -1: 'BEARISH',
                0: 'NEUTRAL',
                1: 'BULLISH',
                2: 'STRONG_BULLISH'
            }
            
            prediction_label = prediction_labels.get(prediction, 'NEUTRAL')
            
            return {
                'prediction': prediction_label,
                'confidence': confidence,
                'probabilities': dict(zip(prediction_labels.values(), probabilities))
            }
            
        except Exception as e:
            logger.error(f"Error predicting pattern for {symbol}: {e}")
            return {'prediction': 'NEUTRAL', 'confidence': 0.0, 'error': str(e)}
    
    def detect_anomaly(self, symbol: str) -> Dict:
        """Detect anomalies for a single symbol"""
        try:
            # Load models if not in memory
            if 'anomaly_detector' not in self.models:
                if not self.load_models():
                    return {'anomaly': False, 'score': 0.0, 'error': 'No trained model available'}
            
            # Get recent data
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT timestamp, close, volume, open, high, low
            FROM market_data
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT 50
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if df.empty or len(df) < self.feature_window:
                return {'anomaly': False, 'score': 0.0, 'error': 'Insufficient data'}
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')
            
            # Extract features
            features = self.extract_technical_features(df)
            
            if features.empty:
                return {'anomaly': False, 'score': 0.0, 'error': 'Feature extraction failed'}
            
            # Use latest features
            latest_features = features.iloc[-1:].values.reshape(1, -1)
            
            # Scale features
            scaler = self.scalers['anomaly_detector']
            scaled_features = scaler.transform(latest_features)
            
            # Detect anomaly
            model = self.models['anomaly_detector']
            anomaly_prediction = model.predict(scaled_features)[0]
            anomaly_score = model.decision_function(scaled_features)[0]
            
            # Convert to more interpretable score
            normalized_score = max(0, min(1, (anomaly_score + 0.5) / 1.0))
            
            return {
                'anomaly': anomaly_prediction == -1,
                'score': normalized_score,
                'raw_score': anomaly_score
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomaly for {symbol}: {e}")
            return {'anomaly': False, 'score': 0.0, 'error': str(e)}
    
    async def run_ml_pattern_recognition(self, symbols: List[str]) -> Dict:
        """Run ML pattern recognition across multiple symbols"""
        try:
            # Ensure models are trained
            if not self.models:
                logger.info("Training ML models...")
                pattern_training = self.train_pattern_classifier(symbols)
                anomaly_training = self.train_anomaly_detector(symbols)
                
                if not pattern_training['success'] and not anomaly_training['success']:
                    return {'error': 'Failed to train any ML models'}
            
            # Run predictions for all symbols
            pattern_predictions = {}
            anomaly_detections = {}
            high_anomaly_symbols = []
            
            for symbol in symbols:
                # Pattern prediction
                pattern_result = self.predict_pattern(symbol)
                pattern_predictions[symbol] = pattern_result
                
                # Anomaly detection
                anomaly_result = self.detect_anomaly(symbol)
                anomaly_detections[symbol] = anomaly_result
                
                # Collect high anomaly symbols
                if anomaly_result.get('anomaly', False) and anomaly_result.get('score', 0) > self.anomaly_threshold:
                    high_anomaly_symbols.append(symbol)
            
            return {
                'pattern_predictions': pattern_predictions,
                'anomaly_detections': anomaly_detections,
                'high_anomaly_symbols': high_anomaly_symbols,
                'models_available': list(self.models.keys()),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in ML pattern recognition: {e}")
            return {
                'error': str(e),
                'pattern_predictions': {},
                'anomaly_detections': {},
                'high_anomaly_symbols': [],
                'analysis_timestamp': datetime.now().isoformat()
            }