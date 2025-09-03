"""
Sentiment Flow Analysis Engine
Tracks sentiment changes and predicts market movements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sqlite3
import os
import psycopg2
from datetime import datetime, timedelta
from scipy import stats
import asyncio
import logging
from prophet import Prophet

logger = logging.getLogger(__name__)

class SentimentFlowAnalyzer:
    def __init__(self, config: dict, db_path: str):
        self.config = config
        self.db_path = db_path
        self.sentiment_sources = ['reddit', 'twitter', 'news', 'trends']
        self.time_windows = ['1h', '4h', '24h', '7d']
        
    def fetch_twitter_sentiment(self, symbol: str, hours_back: int) -> pd.Series:
        """Fetch Twitter sentiment data (placeholder implementation)"""
        try:
            # Placeholder implementation - would integrate with Twitter API
            # For now, return empty series
            return pd.Series(dtype=float)
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment for {symbol}: {e}")
            return pd.Series(dtype=float)
        
    def get_sentiment_data(self, symbol: str, hours_back: int = 168) -> pd.DataFrame:
        """Fetch sentiment data for analysis"""
        try:
            # Use PostgreSQL connection from environment
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                conn = psycopg2.connect(database_url)
            else:
                # Fallback to SQLite if no PostgreSQL available
                conn = sqlite3.connect(self.db_path)
            
            if database_url:
                # PostgreSQL query with INTERVAL
                query = """
                SELECT 
                    timestamp,
                    asset as symbol,
                    confidence as sentiment_score,
                    COALESCE(signals, '') as mention_count,
                    type as source,
                    confidence,
                    COALESCE(price, 0) as news_impact,
                    COALESCE(volume, 0) as social_engagement
                FROM patterns 
                WHERE asset = %s 
                AND timestamp >= NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
                """
                df = pd.read_sql_query(query, conn, params=[symbol, hours_back])
            else:
                # SQLite query with datetime
                query = """
                SELECT 
                    timestamp,
                    asset as symbol,
                    confidence as sentiment_score,
                    COALESCE(signals, '') as mention_count,
                    type as source,
                    confidence,
                    COALESCE(price, 0) as news_impact,
                    COALESCE(volume, 0) as social_engagement
                FROM patterns 
                WHERE asset = ? 
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
                """.format(hours_back)
                df = pd.read_sql_query(query, conn, params=[symbol])
            
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Convert string mention_count to numeric
            df['mention_count'] = pd.to_numeric(df['mention_count'], errors='coerce').fillna(0).astype(float)
            
            # Fetch Twitter sentiment if enabled
            if self.config.get('sentiment_sources', {}).get('twitter', False):
                tweet_scores = self.fetch_twitter_sentiment(symbol, hours_back)
                if not tweet_scores.empty:
                    df = pd.concat([df, pd.DataFrame({
                        'timestamp': tweet_scores.index,
                        'sentiment_score': tweet_scores.values,
                        'source': 'twitter'
                    })])
                    df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_sentiment_momentum(self, sentiment_series: pd.Series, window: int = 12) -> float:
        """Calculate sentiment momentum over time"""
        try:
            if len(sentiment_series) < window:
                return 0.0
            
            # Calculate rolling average
            rolling_avg = sentiment_series.rolling(window=window).mean()
            
            # Calculate momentum as the slope of the trend line
            if len(rolling_avg.dropna()) < 2:
                return 0.0
            
            y = rolling_avg.dropna().values
            x = np.arange(len(y))
            
            # Linear regression to find slope
            slope, _, r_value, _, _ = stats.linregress(x, y)
            
            # Weight by R-squared for confidence
            momentum = slope * (r_value ** 2)
            
            return momentum
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment momentum: {e}")
            return 0.0
    
    def calculate_sentiment_acceleration(self, sentiment_series: pd.Series, window: int = 6) -> float:
        """Calculate sentiment acceleration (second derivative)"""
        try:
            if len(sentiment_series) < window * 2:
                return 0.0
            
            # Calculate first derivative (momentum)
            momentum_series = sentiment_series.rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
            
            # Calculate second derivative (acceleration)
            if len(momentum_series.dropna()) < 2:
                return 0.0
            
            acceleration = momentum_series.diff().mean()
            
            return acceleration
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment acceleration: {e}")
            return 0.0
    
    def analyze_cross_source_divergence(self, symbol: str) -> Dict:
        """Analyze sentiment divergence across different sources"""
        try:
            sentiment_data = self.get_sentiment_data(symbol, 24)  # Last 24 hours
            
            if sentiment_data.empty:
                return {'divergence_detected': False, 'divergence_score': 0.0, 'source_sentiments': {}}
            
            # Group by source and calculate average sentiment
            source_sentiments = {}
            for source in self.sentiment_sources:
                source_data = sentiment_data[sentiment_data['source'].str.contains(source, case=False, na=False)]
                if not source_data.empty:
                    avg_sentiment = source_data['sentiment_score'].mean()
                    source_sentiments[source] = avg_sentiment
            
            if len(source_sentiments) < 2:
                return {'divergence_detected': False, 'divergence_score': 0.0, 'source_sentiments': source_sentiments}
            
            # Calculate divergence score
            sentiment_values = list(source_sentiments.values())
            divergence_score = np.std(sentiment_values)
            
            # Detect significant divergence
            divergence_threshold = 0.3
            divergence_detected = divergence_score > divergence_threshold
            
            return {
                'divergence_detected': divergence_detected,
                'divergence_score': divergence_score,
                'source_sentiments': source_sentiments,
                'max_difference': max(sentiment_values) - min(sentiment_values) if sentiment_values else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-source divergence for {symbol}: {e}")
            return {'divergence_detected': False, 'divergence_score': 0.0, 'source_sentiments': {}}
    
    def analyze_sentiment_price_correlation(self, symbol: str) -> Dict:
        """Analyze correlation between sentiment and price movements"""
        try:
            # Get sentiment data
            sentiment_data = self.get_sentiment_data(symbol, 168)  # 7 days
            
            # Get price data
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                conn = psycopg2.connect(database_url)
                query = """
                SELECT timestamp, close
                FROM market_data 
                WHERE symbol = %s 
                AND timestamp >= NOW() - INTERVAL '7 days'
                ORDER BY timestamp
                """
                price_data = pd.read_sql_query(query, conn, params=[symbol])
            else:
                conn = sqlite3.connect(self.db_path)
                query = """
                SELECT timestamp, close
                FROM market_data 
                WHERE symbol = ? 
                AND timestamp >= strftime('%s', datetime('now', '-7 days'))
                ORDER BY timestamp
                """
                price_data = pd.read_sql_query(query, conn, params=[symbol])
            conn.close()
            
            if sentiment_data.empty or price_data.empty:
                return {'correlation': 0.0, 'lead_lag': 0, 'significance': 0.0, 'sample_size': 0}
            
            # Align timestamps (hourly buckets)
            sentiment_data['hour'] = sentiment_data['timestamp'].dt.floor('H')
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], unit='s')
            price_data['hour'] = price_data['timestamp'].dt.floor('H')
            
            # Aggregate sentiment by hour
            hourly_sentiment = sentiment_data.groupby('hour')['sentiment_score'].mean()
            
            # Calculate price returns
            price_data = price_data.sort_values('timestamp')
            price_data['returns'] = price_data['close'].pct_change()
            hourly_returns = price_data.groupby('hour')['returns'].mean()
            
            # Merge data
            combined_data = pd.DataFrame({
                'sentiment': hourly_sentiment,
                'returns': hourly_returns
            }).dropna()
            
            if len(combined_data) < 10:
                return {'correlation': 0.0, 'lead_lag': 0, 'significance': 0.0, 'sample_size': len(combined_data)}
            
            # Calculate correlation
            correlation = combined_data['sentiment'].corr(combined_data['returns'])
            
            # Test lead-lag relationships
            max_lag = min(12, len(combined_data) // 3)  # Up to 12 hours or 1/3 of data
            best_correlation = correlation
            best_lag = 0
            
            for lag in range(1, max_lag + 1):
                # Sentiment leads price
                lagged_corr = combined_data['sentiment'].corr(combined_data['returns'].shift(-lag))
                if abs(lagged_corr) > abs(best_correlation):
                    best_correlation = lagged_corr
                    best_lag = lag
                
                # Price leads sentiment
                lagged_corr = combined_data['sentiment'].shift(-lag).corr(combined_data['returns'])
                if abs(lagged_corr) > abs(best_correlation):
                    best_correlation = lagged_corr
                    best_lag = -lag
            
            # Calculate statistical significance
            n = len(combined_data)
            if abs(best_correlation) < 1 and n > 2:
                t_stat = best_correlation * np.sqrt((n - 2) / (1 - best_correlation**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                significance = 1 - p_value
            else:
                significance = 0.0
            
            return {
                'correlation': best_correlation,
                'lead_lag': best_lag,  # Positive: sentiment leads, Negative: price leads
                'significance': significance,
                'sample_size': n
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment-price correlation for {symbol}: {e}")
            return {'correlation': 0.0, 'lead_lag': 0, 'significance': 0.0, 'sample_size': 0}
    
    def detect_viral_content_velocity(self, symbol: str) -> Dict:
        """Detect viral content and engagement velocity"""
        try:
            sentiment_data = self.get_sentiment_data(symbol, 24)  # Last 24 hours
            
            if sentiment_data.empty:
                return {'viral_detected': False, 'velocity_score': 0.0, 'max_velocity': 0, 'avg_engagement': 0, 'total_mentions_24h': 0}
            
            # Sort by timestamp
            sentiment_data = sentiment_data.sort_values('timestamp')
            
            # Calculate engagement velocity (mentions per hour)
            sentiment_data['hour'] = sentiment_data['timestamp'].dt.floor('H')
            hourly_mentions = sentiment_data.groupby('hour')['mention_count'].sum()
            
            if len(hourly_mentions) < 2:
                return {'viral_detected': False, 'velocity_score': 0.0, 'max_velocity': 0, 'avg_engagement': 0, 'total_mentions_24h': 0}
            
            # Calculate acceleration in mentions
            mention_velocity = hourly_mentions.diff().rolling(window=3).mean()
            max_velocity = mention_velocity.max() if not mention_velocity.empty else 0
            
            # Calculate engagement score
            avg_engagement = sentiment_data['social_engagement'].mean()
            
            # Viral detection criteria
            velocity_threshold = hourly_mentions.mean() * 2  # 2x average
            viral_detected = max_velocity > velocity_threshold and avg_engagement > 0.5
            
            velocity_score = min(1.0, max_velocity / (hourly_mentions.mean() + 1)) if hourly_mentions.mean() > 0 else 0
            
            return {
                'viral_detected': viral_detected,
                'velocity_score': velocity_score,
                'max_velocity': max_velocity,
                'avg_engagement': avg_engagement,
                'total_mentions_24h': sentiment_data['mention_count'].sum()
            }
            
        except Exception as e:
            logger.error(f"Error detecting viral content velocity for {symbol}: {e}")
            return {'viral_detected': False, 'velocity_score': 0.0, 'max_velocity': 0, 'avg_engagement': 0, 'total_mentions_24h': 0}
    
    async def analyze_sentiment_flow(self, symbol: str) -> Dict:
        """Perform complete sentiment flow analysis"""
        try:
            sentiment_data = self.get_sentiment_data(symbol)
            
            if sentiment_data.empty:
                return {
                    'symbol': symbol,
                    'error': 'No sentiment data available',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate sentiment trajectory
            sentiment_series = sentiment_data.set_index('timestamp')['sentiment_score']
            momentum = self.calculate_sentiment_momentum(sentiment_series)
            acceleration = self.calculate_sentiment_acceleration(sentiment_series)
            
            # Analyze cross-source divergence
            divergence_analysis = self.analyze_cross_source_divergence(symbol)
            
            # Analyze sentiment-price correlation
            correlation_analysis = self.analyze_sentiment_price_correlation(symbol)
            
            # Detect viral content
            viral_analysis = self.detect_viral_content_velocity(symbol)
            
            # Sentiment trend classification
            current_sentiment = sentiment_series.iloc[-1] if len(sentiment_series) > 0 else 0
            prev_sentiment = sentiment_series.iloc[-24] if len(sentiment_series) > 24 else current_sentiment
            
            if momentum > 0.01:
                trend = "Bullish"
            elif momentum < -0.01:
                trend = "Bearish"
            else:
                trend = "Neutral"
            
            # Generate prediction
            confidence = min(0.95, abs(momentum) * 10 + correlation_analysis['significance'])
            
            if momentum > 0 and acceleration > 0:
                prediction = "Continued bullish momentum"
            elif momentum < 0 and acceleration < 0:
                prediction = "Continued bearish momentum"
            elif momentum > 0 and acceleration < 0:
                prediction = "Bullish momentum slowing"
            elif momentum < 0 and acceleration > 0:
                prediction = "Bearish momentum slowing"
            else:
                prediction = "Neutral trend expected"
            
            # Create alert message
            alert_message = f"""📈 Sentiment Flow Alert: {symbol}
├── Reddit: {divergence_analysis['source_sentiments'].get('reddit', 'N/A'):.2f} → {trend} (2h trend)
├── News: {divergence_analysis['source_sentiments'].get('news', 'N/A'):.2f} (lag indicator)
├── Price Action: Following sentiment ({correlation_analysis['correlation']:+.1f} correlation)
└── Prediction: {prediction} ({confidence:.0%} confidence)"""
            
            return {
                'symbol': symbol,
                'sentiment_trajectory': {
                    'current_sentiment': current_sentiment,
                    'previous_sentiment': prev_sentiment,
                    'momentum': momentum,
                    'acceleration': acceleration,
                    'trend': trend
                },
                'cross_source_analysis': divergence_analysis,
                'price_correlation': correlation_analysis,
                'viral_content': viral_analysis,
                'prediction': {
                    'direction': prediction,
                    'confidence': confidence
                },
                'alert_message': alert_message,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment flow analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def enhanced_sentiment_aggregation(self, sentiment_scores: List[float], confidence_weights: List[float]) -> Dict:
        """Enhanced sentiment aggregation with confidence weighting."""
        if not sentiment_scores or not confidence_weights:
            return {"aggregated_sentiment": 0.5, "confidence": 0.0}
        
        total_weight = sum(confidence_weights)
        if total_weight == 0:
            return {"aggregated_sentiment": 0.5, "confidence": 0.0}
        
        weighted_sentiment = sum(score * weight for score, weight in zip(sentiment_scores, confidence_weights)) / total_weight
        avg_confidence = sum(confidence_weights) / len(confidence_weights)
        
        return {"aggregated_sentiment": weighted_sentiment, "confidence": avg_confidence}

    def sentiment_regime_classification(self, sentiment_history: List[float]) -> Dict:
        """Classify sentiment regime based on recent history."""
        if len(sentiment_history) < 10:
            return {"regime": "UNKNOWN", "stability": 0.0}
        
        recent_avg = sum(sentiment_history[-10:]) / 10
        volatility = np.std(sentiment_history[-10:]) if len(sentiment_history) >= 10 else 0
        
        if recent_avg > 0.7 and volatility < 0.1:
            regime = "STRONG_BULLISH"
        elif recent_avg > 0.6 and volatility < 0.15:
            regime = "BULLISH"
        elif recent_avg < 0.3 and volatility < 0.1:
            regime = "STRONG_BEARISH"
        elif recent_avg < 0.4 and volatility < 0.15:
            regime = "BEARISH"
        else:
            regime = "VOLATILE_MIXED"
        
        stability = 1.0 / (1.0 + volatility)
        return {"regime": regime, "stability": stability, "recent_average": recent_avg, "volatility": volatility}

    def calculate_sentiment_lead_lag(self, sentiment_ts: pd.Series, price_ts: pd.Series) -> Dict:
        """Calculate lead/lag relationship between sentiment and price."""
        try:
            correlations = {}
            for lag in range(-5, 6):  # Test lags from -5 to +5 periods
                if lag == 0:
                    corr = sentiment_ts.corr(price_ts)
                elif lag > 0:
                    corr = sentiment_ts.shift(lag).corr(price_ts)
                else:
                    corr = sentiment_ts.corr(price_ts.shift(-lag))
                correlations[lag] = corr if not pd.isna(corr) else 0
            
            max_lag = max(correlations.items(), key=lambda x: abs(x[1]))
            return {"optimal_lag": max_lag[0], "max_correlation": max_lag[1], "all_correlations": correlations}
        except Exception:
            return {"optimal_lag": 0, "max_correlation": 0, "all_correlations": {}}


class ForecastingModule:
    """AI Predictive Forecasting for geopolitics, markets, stocks, and companies"""
    
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer

    def get_sentiment_and_price_data(self, symbol: str) -> pd.DataFrame:
        """Get historical sentiment and price data for forecasting"""
        try:
            # Get sentiment data from existing analyzer
            sentiment_data = self.sentiment_analyzer.get_sentiment_data(symbol, hours_back=720)  # 30 days
            
            if sentiment_data.empty:
                return pd.DataFrame()
            
            # Prepare data for Prophet forecasting
            df = sentiment_data[['timestamp', 'sentiment_score']].copy()
            df = df.sort_values('timestamp')
            df = df.dropna()
            
            # Add price data if available
            try:
                conn = sqlite3.connect(self.sentiment_analyzer.db_path)
                price_query = """
                SELECT timestamp, close as price
                FROM market_data 
                WHERE symbol = ? 
                AND timestamp >= strftime('%s', datetime('now', '-30 days'))
                ORDER BY timestamp
                """
                price_data = pd.read_sql_query(price_query, conn, params=[symbol])
                conn.close()
                
                if not price_data.empty:
                    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], unit='s')
                    # Merge with sentiment data
                    df = pd.merge(df, price_data, on='timestamp', how='left')
                    
            except Exception as e:
                logger.warning(f"Could not fetch price data for {symbol}: {e}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting data for forecasting {symbol}: {e}")
            return pd.DataFrame()

    def predict_events(self, symbol: str, horizon_days: int = 30, event_type: str = 'market') -> Dict:
        """
        Predict future events with probabilities and major dates.
        - event_type: 'market', 'geopolitics', 'stocks', 'companies'
        """
        try:
            data = self.get_sentiment_and_price_data(symbol)
            if data.empty:
                return {'error': 'No data available for prediction'}

            # Prepare data for Prophet (time-series forecasting)
            df = pd.DataFrame({
                'ds': data['timestamp'], 
                'y': data['sentiment_score']
            }).dropna()
            
            if len(df) < 10:
                return {'error': 'Insufficient data for prediction'}

            # Configure Prophet model based on event type
            if event_type == 'geopolitics':
                # More conservative for geopolitical events
                model = Prophet(
                    daily_seasonality=True, 
                    yearly_seasonality=True,
                    uncertainty_samples=100,
                    changepoint_prior_scale=0.01  # Lower for more stability
                )
            else:
                # Standard configuration for market/stocks
                model = Prophet(
                    daily_seasonality=True, 
                    yearly_seasonality=True,
                    uncertainty_samples=100
                )
            
            # Add special event holidays for geopolitics
            if event_type == 'geopolitics':
                model.add_country_holidays(country_name='US')
            
            # Fit the model
            model.fit(df)

            # Generate forecast
            future = model.make_future_dataframe(periods=horizon_days)
            forecast = model.predict(future)

            # Calculate probabilistic outcomes
            current_avg = df['y'].tail(7).mean()  # Last week average
            future_predictions = forecast['yhat'][-horizon_days:]
            
            prob_up = (future_predictions > current_avg).mean()
            prob_down = 1 - prob_up

            # Find major dates (peaks and troughs)
            future_dates = forecast['ds'][-horizon_days:]
            peaks_idx = []
            troughs_idx = []
            
            # Simple peak/trough detection
            for i in range(1, len(future_predictions) - 1):
                if (future_predictions.iloc[i] > future_predictions.iloc[i-1] and 
                    future_predictions.iloc[i] > future_predictions.iloc[i+1]):
                    peaks_idx.append(i)
                elif (future_predictions.iloc[i] < future_predictions.iloc[i-1] and 
                      future_predictions.iloc[i] < future_predictions.iloc[i+1]):
                    troughs_idx.append(i)
            
            peaks = [future_dates.iloc[-horizon_days:].iloc[i].date() for i in peaks_idx[:3]]
            troughs = [future_dates.iloc[-horizon_days:].iloc[i].date() for i in troughs_idx[:3]]

            # Adjust confidence for geopolitics (higher uncertainty)
            if event_type == 'geopolitics':
                prob_up *= 0.7
                prob_down = 1 - prob_up

            # Generate trend analysis
            trend_slope = (future_predictions.iloc[-1] - future_predictions.iloc[0]) / horizon_days
            trend_direction = "upward" if trend_slope > 0 else "downward"
            
            # Calculate confidence based on model uncertainty
            uncertainty = forecast['yhat_upper'][-horizon_days:] - forecast['yhat_lower'][-horizon_days:]
            avg_uncertainty = uncertainty.mean()
            confidence = max(0.1, min(0.95, 1 - (avg_uncertainty / future_predictions.std())))

            return {
                'event_type': event_type,
                'symbol': symbol,
                'horizon_days': horizon_days,
                'probability_positive': prob_up,
                'probability_negative': prob_down,
                'major_positive_dates': peaks,
                'major_negative_dates': troughs,
                'trend_direction': trend_direction,
                'trend_strength': abs(trend_slope),
                'confidence': confidence,
                'summary': f'{horizon_days}-day {event_type} forecast for {symbol}: {prob_up*100:.1f}% chance of positive outcome. Trend: {trend_direction} (confidence: {confidence:.1%})',
                'forecast_data': {
                    'dates': future_dates[-horizon_days:].dt.date.tolist(),
                    'predictions': future_predictions.tolist(),
                    'upper_bound': forecast['yhat_upper'][-horizon_days:].tolist(),
                    'lower_bound': forecast['yhat_lower'][-horizon_days:].tolist()
                }
            }

        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {e}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'event_type': event_type,
                'symbol': symbol
            }

    def predict_geopolitical_events(self, region: str = 'global', horizon_days: int = 90) -> Dict:
        """Specialized geopolitical event prediction"""
        try:
            # Use proxy symbols for geopolitical analysis
            proxy_symbols = {
                'global': ['BTC', 'SPY', 'VIX'],
                'asia': ['NIFTY', 'HSI'],
                'europe': ['DAX', 'FTSE'],
                'americas': ['SPY', 'QQQ']
            }
            
            symbols = proxy_symbols.get(region, ['BTC'])
            all_predictions = []
            
            for symbol in symbols:
                prediction = self.predict_events(symbol, horizon_days, 'geopolitics')
                if 'error' not in prediction:
                    all_predictions.append(prediction)
            
            if not all_predictions:
                return {'error': 'No geopolitical data available for analysis'}
            
            # Aggregate predictions
            avg_prob_positive = np.mean([p['probability_positive'] for p in all_predictions])
            avg_confidence = np.mean([p['confidence'] for p in all_predictions])
            
            # Collect all significant dates
            all_positive_dates = []
            all_negative_dates = []
            for p in all_predictions:
                all_positive_dates.extend(p['major_positive_dates'])
                all_negative_dates.extend(p['major_negative_dates'])
            
            return {
                'region': region,
                'event_type': 'geopolitics',
                'horizon_days': horizon_days,
                'overall_stability_probability': avg_prob_positive,
                'confidence': avg_confidence,
                'key_positive_periods': sorted(set(all_positive_dates))[:5],
                'key_risk_periods': sorted(set(all_negative_dates))[:5],
                'detailed_predictions': all_predictions,
                'summary': f'Geopolitical forecast for {region}: {avg_prob_positive*100:.1f}% probability of stability over {horizon_days} days (confidence: {avg_confidence:.1%})'
            }
            
        except Exception as e:
            logger.error(f"Error in geopolitical prediction for {region}: {e}")
            return {'error': f'Geopolitical prediction failed: {str(e)}'}

    async def generate_prediction_alerts(self, symbols: List[str]) -> List[Dict]:
        """Generate prediction-based alerts for multiple symbols"""
        alerts = []
        
        for symbol in symbols:
            try:
                # Market prediction
                market_prediction = self.predict_events(symbol, 7, 'market')
                
                if 'error' not in market_prediction:
                    if market_prediction['confidence'] > 0.7:
                        alert = {
                            'type': 'prediction_alert',
                            'symbol': symbol,
                            'prediction': market_prediction,
                            'alert_level': 'high' if market_prediction['confidence'] > 0.8 else 'medium',
                            'message': f"🔮 {symbol} Prediction: {market_prediction['summary']}"
                        }
                        alerts.append(alert)
                
                # Add slight delay to avoid overwhelming
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating prediction alert for {symbol}: {e}")
        
        return alerts