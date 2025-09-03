"""
Advanced Trading System Orchestrator
Integrates all advanced trading features into the main platform
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the advanced analyzers
from decoder.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from decoder.correlation_matrix_engine import CorrelationMatrixEngine
from decoder.signal_engine import SignalEngine
from decoder.portfolio_optimizer import PortfolioOptimizer
from decoder.sentiment_flow_analyzer import SentimentFlowAnalyzer, ForecastingModule
from decoder.institutional_flow_detector import InstitutionalFlowDetector
from decoder.ml_pattern_recognizer import MLPatternRecognizer
from decoder.smart_alert_manager import SmartAlertManager

# Import Phase 5 advanced features
from executor.community_simulator import CommunitySimulator
from executor.reddit_poster import RedditPoster
from utils.user_acquisition import UserAcquisition
from utils.telegram_bot import TelegramEngagementBot

# Import Phase 1-4 AI Strategist features
from utils.event_normalizer import EventNormalizer
from decoder.knowledge_graph import KG, AutoLinker
from decoder.regime_engine import RegimeEngine
from decoder.causal_engine import CausalEngine

# Import Phases 5-8 features
from decoder.decision_policy import DecisionPolicy
from utils.post_mortem import Verdicts
from decoder.gpt_integration import GPTIntegrator
from decoder.advanced_models import AdvancedModels

logger = logging.getLogger(__name__)

class AdvancedTradingOrchestrator:
    def __init__(self, config_path: str = 'config.json', db_path: str = 'patterns.db'):
        """Initialize the advanced trading orchestrator"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.db_path = db_path
        
        # Initialize all advanced analyzers
        self.multi_timeframe = MultiTimeframeAnalyzer(self.config, db_path)
        self.correlation_engine = CorrelationMatrixEngine(self.config, db_path)
        self.signal_engine = SignalEngine(self.config, db_path)
        self.portfolio_optimizer = PortfolioOptimizer(self.config, db_path)
        self.sentiment_analyzer = SentimentFlowAnalyzer(self.config, db_path)
        self.institutional_detector = InstitutionalFlowDetector(self.config, db_path)
        self.ml_recognizer = MLPatternRecognizer(self.config, db_path)
        self.alert_manager = SmartAlertManager(self.config, db_path)
        
        # Initialize Phase 5 advanced features
        self.forecasting_module = ForecastingModule(self.sentiment_analyzer)
        self.reddit_poster = RedditPoster()
        self.community_simulator = CommunitySimulator(self.reddit_poster)
        self.user_acquisition = UserAcquisition(self.reddit_poster)
        self.telegram_bot = TelegramEngagementBot()
        
        # Initialize Phase 1-4 AI Strategist features
        self.event_normalizer = EventNormalizer(db_path)
        self.knowledge_graph = KG(db_path)
        self.auto_linker = AutoLinker(self.knowledge_graph)
        self.regime_engine = RegimeEngine(db_path, equity_scanner=None, binance_scanner=None)
        self.causal_engine = CausalEngine(db_path, equity_scanner=None, binance_scanner=None)
        
        # Initialize Phases 5-8 features
        self.decision_policy = DecisionPolicy(db_path)
        self.verdicts = Verdicts(db_path)
        
        # Check for OpenAI API key and initialize GPT integration
        openai_key = os.getenv('OPENAI_API_KEY')
        self.gpt_integrator = GPTIntegrator(openai_key or "")
        
        self.advanced_models = AdvancedModels()
        
        # Load assets configuration
        try:
            with open('assets-config.json', 'r') as f:
                self.assets_config = json.load(f)
            
            with open('sectors-config.json', 'r') as f:
                self.sectors_config = json.load(f)
        except FileNotFoundError:
            # Create default configurations if they don't exist
            self.create_default_configs()
        
        # Get symbol list (top assets for analysis)
        self.symbols = self.get_analysis_symbols()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('advanced_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Advanced Trading Orchestrator initialized successfully")
        
        # Initialize Phase 5 features status
        self.phase5_features = {
            'ai_forecasting': True,
            'community_simulation': True,
            'user_acquisition': True,
            'telegram_engagement': True
        }
        
        # Initialize Phase 1-4 AI Strategist features status
        self.ai_strategist_features = {
            'event_normalization': True,
            'knowledge_graph': True,
            'regime_detection': True,
            'causal_analysis': True
        }
        
        # Initialize enhancements
        self.initialize_enhancements()
    
    def create_default_configs(self):
        """Create default asset and sector configurations"""
        # Default assets configuration
        self.assets_config = {
            "assets": {
                "BTCUSDT": "Bitcoin",
                "ETHUSDT": "Ethereum", 
                "ADAUSDT": "Cardano",
                "DOTUSDT": "Polkadot",
                "LINKUSDT": "Chainlink",
                "NIFTY_50": "Nifty 50",
                "BANKNIFTY": "Bank Nifty",
                "RELIANCE": "Reliance Industries",
                "TCS": "Tata Consultancy Services",
                "INFY": "Infosys"
            }
        }
        
        # Default sectors configuration
        self.sectors_config = {
            "sectors": {
                "cryptocurrency": {
                    "assets": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"],
                    "weight": 0.4
                },
                "indian_equity": {
                    "assets": ["NIFTY_50", "BANKNIFTY", "RELIANCE", "TCS", "INFY"],
                    "weight": 0.6
                }
            }
        }
        
        # Save default configurations
        with open('assets-config.json', 'w') as f:
            json.dump(self.assets_config, f, indent=2)
        
        with open('sectors-config.json', 'w') as f:
            json.dump(self.sectors_config, f, indent=2)
    
    def get_analysis_symbols(self) -> List[str]:
        """Get list of symbols for analysis"""
        try:
            all_symbols = list(self.assets_config.get('assets', {}).keys())
            # Return top 10 symbols for analysis to avoid overwhelming the system
            return all_symbols[:10]
        except Exception as e:
            self.logger.error(f"Error getting analysis symbols: {e}")
            return ["BTCUSDT", "ETHUSDT", "NIFTY_50", "BANKNIFTY", "RELIANCE"]
    
    def initialize_enhancements(self):
        """Initialize enhancement features based on feature flags"""
        try:
            logger.info("🚀 Initializing Phase 1 enhancements...")
            
            # Load feature flags
            with open('permissions.json', 'r') as f:
                permissions = json.load(f)
            
            feature_flags = permissions.get('feature_flags', {}).get('experimental_features', {})
            enhancement_settings = permissions.get('enhancement_settings', {})
            
            # Initialize adaptive learning
            if feature_flags.get('adaptive_learning', False):
                self.adaptive_learning_enabled = True
                self.learning_rate = enhancement_settings.get('adaptive_learning', {}).get('learning_rate', 0.01)
                logger.info("✅ Adaptive learning initialized")
            else:
                self.adaptive_learning_enabled = False
            
            # Initialize backtesting
            if feature_flags.get('backtesting_enabled', False):
                self.backtesting_enabled = True
                self.backtesting_periods = enhancement_settings.get('backtesting', {}).get('historical_periods', 252)
                logger.info("✅ Backtesting capabilities initialized")
            else:
                self.backtesting_enabled = False
            
            # Initialize regime detection
            if feature_flags.get('regime_detection', False):
                self.regime_detection_enabled = True
                self.volatility_threshold = enhancement_settings.get('regime_detection', {}).get('volatility_threshold', 0.25)
                logger.info("✅ Regime detection initialized")
            else:
                self.regime_detection_enabled = False
            
            # Initialize enhancement tracking
            self.enhancement_metrics = {
                'initialization_time': datetime.now().isoformat(),
                'features_enabled': sum([
                    self.adaptive_learning_enabled,
                    self.backtesting_enabled,
                    self.regime_detection_enabled
                ]),
                'total_features': 3
            }
            
            logger.info(f"✅ Phase 1 enhancements initialized: {self.enhancement_metrics['features_enabled']}/3 features enabled")
            
        except Exception as e:
            logger.error(f"❌ Error initializing enhancements: {e}")
            self.adaptive_learning_enabled = False
            self.backtesting_enabled = False
            self.regime_detection_enabled = False

    def get_enhancement_status(self) -> Dict:
        """Get current enhancement status and metrics"""
        try:
            return {
                'adaptive_learning_enabled': getattr(self, 'adaptive_learning_enabled', False),
                'backtesting_enabled': getattr(self, 'backtesting_enabled', False),
                'regime_detection_enabled': getattr(self, 'regime_detection_enabled', False),
                'enhancement_metrics': getattr(self, 'enhancement_metrics', {}),
                'orchestrator_version': '1.3.0-enhanced',
                'status_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting enhancement status: {e}")
            return {'error': str(e)}
    
    async def run_full_analysis(self) -> Dict:
        """Run complete advanced analysis across all systems"""
        self.logger.info("Starting comprehensive advanced market analysis...")
        
        results = {
            'analysis_start_time': datetime.now().isoformat(),
            'symbols_analyzed': self.symbols,
            'features_analyzed': []
        }
        
        try:
            # 1. Multi-timeframe analysis
            self.logger.info("Running multi-timeframe analysis...")
            timeframe_results = {}
            for symbol in self.symbols[:5]:  # Analyze top 5 symbols
                try:
                    timeframe_results[symbol] = await self.multi_timeframe.analyze_multi_timeframe(symbol)
                except Exception as e:
                    self.logger.error(f"Multi-timeframe analysis failed for {symbol}: {e}")
                    timeframe_results[symbol] = {'error': str(e)}
            
            results['multi_timeframe_analysis'] = timeframe_results
            results['features_analyzed'].append('multi_timeframe')
            
            # 2. Correlation analysis
            self.logger.info("Running correlation matrix analysis...")
            try:
                correlation_results = await self.correlation_engine.run_correlation_analysis(
                    self.symbols, self.sectors_config
                )
                results['correlation_analysis'] = correlation_results
                results['features_analyzed'].append('correlation_matrix')
            except Exception as e:
                self.logger.error(f"Correlation analysis failed: {e}")
                results['correlation_analysis'] = {'error': str(e)}
            
            # 3. Signal generation
            self.logger.info("Generating advanced trading signals...")
            try:
                signal_results = await self.signal_engine.run_signal_generation(self.symbols)
                results['trading_signals'] = signal_results
                results['features_analyzed'].append('signal_generation')
            except Exception as e:
                self.logger.error(f"Signal generation failed: {e}")
                results['trading_signals'] = {'error': str(e)}
            
            # 4. Portfolio optimization
            self.logger.info("Running portfolio optimization...")
            try:
                portfolio_results = await self.portfolio_optimizer.run_portfolio_optimization(
                    self.symbols[:8], self.sectors_config
                )
                results['portfolio_optimization'] = portfolio_results
                results['features_analyzed'].append('portfolio_optimization')
            except Exception as e:
                self.logger.error(f"Portfolio optimization failed: {e}")
                results['portfolio_optimization'] = {'error': str(e)}
            
            # 5. Sentiment flow analysis
            self.logger.info("Analyzing sentiment flows...")
            sentiment_results = {}
            for symbol in self.symbols[:3]:  # Top 3 symbols for sentiment analysis
                try:
                    sentiment_results[symbol] = await self.sentiment_analyzer.analyze_sentiment_flow(symbol)
                except Exception as e:
                    self.logger.error(f"Sentiment analysis failed for {symbol}: {e}")
                    sentiment_results[symbol] = {'error': str(e)}
            
            results['sentiment_analysis'] = sentiment_results
            results['features_analyzed'].append('sentiment_flow')
            
            # 6. Institutional flow detection
            self.logger.info("Detecting institutional flows...")
            try:
                institutional_results = await self.institutional_detector.run_institutional_detection(
                    self.symbols[:8]
                )
                results['institutional_analysis'] = institutional_results
                results['features_analyzed'].append('institutional_flow')
            except Exception as e:
                self.logger.error(f"Institutional analysis failed: {e}")
                results['institutional_analysis'] = {'error': str(e)}
            
            # 7. ML pattern recognition
            self.logger.info("Running ML pattern recognition...")
            try:
                ml_results = await self.ml_recognizer.run_ml_pattern_recognition(self.symbols[:6])
                results['ml_pattern_analysis'] = ml_results
                results['features_analyzed'].append('ml_patterns')
            except Exception as e:
                self.logger.error(f"ML pattern recognition failed: {e}")
                results['ml_pattern_analysis'] = {'error': str(e)}
            
        except Exception as e:
            self.logger.error(f"Critical error in analysis pipeline: {e}")
            results['critical_error'] = str(e)
        
        results['analysis_end_time'] = datetime.now().isoformat()
        
        analysis_duration = (
            datetime.fromisoformat(results['analysis_end_time']) - 
            datetime.fromisoformat(results['analysis_start_time'])
        ).total_seconds()
        
        results['analysis_duration_seconds'] = analysis_duration
        results['features_completed'] = len(results['features_analyzed'])
        
        # 8. AI Strategist Features (Phase 1-4)
        self.logger.info("Running AI Strategist features...")
        try:
            ai_strategist_results = await self.run_ai_strategist_features()
            results['ai_strategist_features'] = ai_strategist_results
            results['features_analyzed'].append('ai_strategist')
        except Exception as e:
            self.logger.error(f"AI Strategist features failed: {e}")
            results['ai_strategist_features'] = {'error': str(e)}
        
        # 9. Phase 5 Advanced Features
        self.logger.info("Running Phase 5 advanced features...")
        try:
            phase5_results = await self.run_phase5_features()
            results['phase5_features'] = phase5_results
            results['features_analyzed'].append('phase5_advanced')
        except Exception as e:
            self.logger.error(f"Phase 5 features failed: {e}")
            results['phase5_features'] = {'error': str(e)}
        
        # 10. Phases 5-8 Enhanced Features
        self.logger.info("Running Phases 5-8 enhanced features...")
        try:
            phases_5_8_results = await self.run_phases_5_8_features()
            results['phases_5_8_features'] = phases_5_8_results
            results['features_analyzed'].append('phases_5_8')
        except Exception as e:
            self.logger.error(f"Phases 5-8 features failed: {e}")
            results['phases_5_8_features'] = {'error': str(e)}
        
        self.logger.info(f"Advanced analysis completed in {analysis_duration:.2f} seconds")
        self.logger.info(f"Features analyzed: {', '.join(results['features_analyzed'])}")
        
        return results
    
    async def process_advanced_alerts(self, analysis_results: Dict) -> Dict:
        """Process and send alerts based on advanced analysis results"""
        raw_alerts = []
        
        try:
            # Extract alerts from different analyzers
            
            # Multi-timeframe alerts
            for symbol, tf_result in analysis_results.get('multi_timeframe_analysis', {}).items():
                if isinstance(tf_result, dict) and tf_result.get('overall_confidence', 0) > 0.7:
                    raw_alerts.append({
                        'alert_type': 'MULTI_TIMEFRAME',
                        'symbol': symbol,
                        'confidence': tf_result['overall_confidence'],
                        'message': f"🎯 Multi-timeframe confirmation for {symbol}: {tf_result.get('overall_signal', 'UNKNOWN')}\\n"
                                  f"Confidence: {tf_result['overall_confidence']:.0%}\\n"
                                  f"Timeframes aligned: {tf_result.get('confirmation_ratio', 0):.0%}",
                        'timestamp': datetime.now().timestamp()
                    })
            
            # Correlation break alerts
            corr_analysis = analysis_results.get('correlation_analysis', {})
            if isinstance(corr_analysis, dict):
                corr_alerts = corr_analysis.get('correlation_alerts', [])
                raw_alerts.extend(corr_alerts)
            
            # Trading signal alerts
            trading_signals = analysis_results.get('trading_signals', {})
            if isinstance(trading_signals, dict):
                signals = trading_signals.get('signals', [])
                for signal in signals:
                    if signal.get('confidence', 0) > 0.6:
                        raw_alerts.append({
                            'alert_type': 'TRADING_SIGNAL',
                            'symbol': signal['symbol'],
                            'confidence': signal['confidence'],
                            'message': f"⚡ Trading signal for {signal['symbol']}: {signal['action']}\\n"
                                      f"Confidence: {signal['confidence']:.0%}\\n"
                                      f"Entry: ${signal.get('entry_price', 'TBD')}",
                            'timestamp': datetime.now().timestamp()
                        })
            
            # Portfolio rebalancing alerts
            portfolio_result = analysis_results.get('portfolio_optimization', {})
            if isinstance(portfolio_result, dict) and portfolio_result.get('rebalancing_needed', False):
                raw_alerts.append({
                    'alert_type': 'PORTFOLIO_REBALANCING',
                    'symbol': 'PORTFOLIO',
                    'confidence': 0.8,
                    'message': portfolio_result.get('alert_message', '📊 Portfolio rebalancing recommended'),
                    'timestamp': datetime.now().timestamp()
                })
            
            # Sentiment flow alerts
            for symbol, sentiment_result in analysis_results.get('sentiment_analysis', {}).items():
                if isinstance(sentiment_result, dict) and sentiment_result.get('prediction', {}).get('confidence', 0) > 0.7:
                    raw_alerts.append({
                        'alert_type': 'SENTIMENT_FLOW',
                        'symbol': symbol,
                        'confidence': sentiment_result['prediction']['confidence'],
                        'message': sentiment_result.get('alert_message', f'💬 Sentiment alert for {symbol}'),
                        'timestamp': datetime.now().timestamp()
                    })
            
            # Institutional flow alerts
            institutional_result = analysis_results.get('institutional_analysis', {})
            if isinstance(institutional_result, dict):
                for symbol in institutional_result.get('high_activity_symbols', []):
                    symbol_data = institutional_result.get('institutional_flow_analysis', {}).get(symbol, {})
                    if symbol_data.get('alert_generated', False):
                        raw_alerts.append({
                            'alert_type': 'INSTITUTIONAL_FLOW',
                            'symbol': symbol,
                            'confidence': symbol_data.get('institutional_score', 0.8),
                            'message': symbol_data.get('alert_message', f'🏛️ Institutional activity detected for {symbol}'),
                            'timestamp': datetime.now().timestamp()
                        })
            
            # ML pattern alerts
            ml_result = analysis_results.get('ml_pattern_analysis', {})
            if isinstance(ml_result, dict):
                for symbol in ml_result.get('high_anomaly_symbols', []):
                    raw_alerts.append({
                        'alert_type': 'ML_ANOMALY',
                        'symbol': symbol,
                        'confidence': 0.75,
                        'message': f'🤖 ML anomaly detected for {symbol}\\nPattern recognition suggests unusual market behavior',
                        'timestamp': datetime.now().timestamp()
                    })
            
            # **NEW: Pattern-based alerts from generated patterns**
            # This is the missing link - convert detected patterns into actionable alerts
            try:
                import sqlite3
                from datetime import datetime, timedelta
                
                # Connect to patterns database
                conn = sqlite3.connect(self.db_path)
                
                # Get recent high-confidence patterns (last 1 hour)
                one_hour_ago = (datetime.now() - timedelta(hours=1)).timestamp()
                query = """
                SELECT asset, type, confidence, signals, timestamp 
                FROM patterns 
                WHERE timestamp > ? AND confidence > 0.6
                ORDER BY timestamp DESC, confidence DESC
                LIMIT 20
                """
                
                cursor = conn.execute(query, (one_hour_ago,))
                recent_patterns = cursor.fetchall()
                conn.close()
                
                for pattern in recent_patterns:
                    asset, pattern_type, confidence, signals_json, timestamp = pattern
                    
                    try:
                        import json
                        signals = json.loads(signals_json) if signals_json else {}
                    except:
                        signals = {}
                    
                    # Convert pattern to alert
                    action = signals.get('action', 'WATCH')
                    entry_price = signals.get('entry_price', 'TBD')
                    
                    # Create alert message based on pattern type
                    alert_messages = {
                        'BREAKOUT_CONFIRMATION': f'🚀 Breakout confirmed for {asset}',
                        'VOLUME_SPIKE': f'📈 Volume spike detected for {asset}',
                        'MOMENTUM_SHIFT': f'⚡ Momentum shift in {asset}',
                        'SUPPORT_RESISTANCE': f'🎯 Key level test for {asset}',
                        'TREND_REVERSAL': f'🔄 Trend reversal signal for {asset}'
                    }
                    
                    message = alert_messages.get(pattern_type, f'📊 Pattern detected for {asset}')
                    message += f'\\nAction: {action}\\nConfidence: {confidence:.0%}'
                    if entry_price != 'TBD':
                        message += f'\\nEntry: ${entry_price}'
                    
                    raw_alerts.append({
                        'alert_type': 'PATTERN_DETECTION',
                        'symbol': asset,
                        'confidence': confidence,
                        'message': message,
                        'timestamp': timestamp,
                        'pattern_type': pattern_type,
                        'action': action,
                        'entry_price': entry_price
                    })
                    
                self.logger.info(f"Added {len(recent_patterns)} pattern-based alerts from recent patterns")
                    
            except Exception as e:
                self.logger.error(f"Error fetching patterns for alerts: {e}")
            
        except Exception as e:
            self.logger.error(f"Error extracting alerts: {e}")
        
        # Process alerts through smart alert manager
        try:
            processed_alerts = await self.alert_manager.process_smart_alerts(raw_alerts)
            self.logger.info(f"Generated {len(processed_alerts.get('processed_alerts', []))} processed alerts from {len(raw_alerts)} raw alerts")
            
            # **NEW: Automatic paper trading execution from high-confidence alerts**
            # Execute paper trades for alerts with confidence > 0.7 and clear buy/sell actions
            try:
                high_confidence_alerts = [
                    alert for alert in processed_alerts.get('processed_alerts', [])
                    if alert.get('confidence', 0) > 0.7 and 
                    alert.get('action') in ['BUY', 'SELL'] and
                    alert.get('symbol', '').upper() in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']  # Only crypto for now
                ]
                
                if high_confidence_alerts:
                    from executor.trade_executor import TradeExecutor
                    
                    # Initialize trade executor if not exists
                    if not hasattr(self, 'trade_executor'):
                        self.trade_executor = TradeExecutor()
                    
                    # Execute paper trades
                    executed_trades = 0
                    for alert in high_confidence_alerts:
                        try:
                            # Convert alert to pattern format expected by trade executor
                            trade_pattern = {
                                'asset': alert['symbol'],
                                'type': alert.get('pattern_type', 'ALERT_BASED'),
                                'confidence': alert['confidence'],
                                'signals': {
                                    'action': alert.get('action', 'WATCH'),
                                    'entry_price': alert.get('entry_price'),
                                    'reasoning': alert.get('message', '')
                                },
                                'timestamp': alert.get('timestamp', datetime.now().timestamp())
                            }
                            
                            # Execute the trade
                            await self.trade_executor.execute_trade(trade_pattern, alert['confidence'])
                            executed_trades += 1
                            
                            self.logger.info(f"Executed paper trade for {alert['symbol']} based on {alert.get('alert_type')} alert")
                            
                        except Exception as e:
                            self.logger.error(f"Error executing paper trade for {alert['symbol']}: {e}")
                    
                    if executed_trades > 0:
                        self.logger.info(f"Auto-executed {executed_trades} paper trades from high-confidence alerts")
                
            except Exception as e:
                self.logger.error(f"Error in automatic paper trading: {e}")
            
            return processed_alerts
        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")
            return {
                'processed_alerts': [],
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }

    async def run_phase5_features(self) -> Dict:
        """Run Phase 5 advanced features: AI Forecasting, Community, User Acquisition, Telegram"""
        try:
            results = {}
            
            # 1. AI Predictive Forecasting
            if self.phase5_features.get('ai_forecasting'):
                try:
                    forecast_results = {}
                    
                    # Generate market predictions for top symbols
                    for symbol in self.symbols[:3]:  # Limit to top 3 for performance
                        prediction = self.forecasting_module.predict_events(symbol, 7, 'market')
                        if 'error' not in prediction:
                            forecast_results[symbol] = prediction
                    
                    # Generate geopolitical forecast
                    geo_forecast = self.forecasting_module.predict_geopolitical_events('global', 30)
                    if 'error' not in geo_forecast:
                        forecast_results['geopolitical'] = geo_forecast
                    
                    results['forecasting'] = {
                        'success': True,
                        'predictions_generated': len(forecast_results),
                        'results': forecast_results
                    }
                    
                except Exception as e:
                    logger.error(f"Error in AI forecasting: {e}")
                    results['forecasting'] = {'success': False, 'error': str(e)}
            
            # 2. Community Simulation
            if self.phase5_features.get('community_simulation'):
                try:
                    # Generate community insights
                    community_insights = self.community_simulator.get_community_insights()
                    
                    # Simulate engagement for recent analysis
                    if community_insights.get('overview', {}).get('total_posts', 0) < 5:
                        # Create some sample community content
                        sample_content = "Latest AI analysis shows strong bullish signals across major assets. What are your thoughts on the current market sentiment?"
                        simulation_result = await self.community_simulator.simulate_post(sample_content, 'CryptoCurrency')
                        
                        results['community'] = {
                            'success': True,
                            'insights': community_insights,
                            'simulation_result': simulation_result
                        }
                    else:
                        results['community'] = {
                            'success': True,
                            'insights': community_insights
                        }
                        
                except Exception as e:
                    logger.error(f"Error in community simulation: {e}")
                    results['community'] = {'success': False, 'error': str(e)}
            
            # 3. User Acquisition
            if self.phase5_features.get('user_acquisition'):
                try:
                    # Get acquisition analytics
                    acquisition_analytics = self.user_acquisition.get_acquisition_analytics()
                    
                    # Analyze funnel performance
                    funnel_analysis = await self.user_acquisition.analyze_funnel_performance()
                    
                    results['user_acquisition'] = {
                        'success': True,
                        'analytics': acquisition_analytics,
                        'funnel_analysis': funnel_analysis
                    }
                    
                except Exception as e:
                    logger.error(f"Error in user acquisition: {e}")
                    results['user_acquisition'] = {'success': False, 'error': str(e)}
            
            # 4. Telegram Engagement
            if self.phase5_features.get('telegram_engagement'):
                try:
                    # Get engagement analytics
                    engagement_analytics = self.telegram_bot.get_engagement_analytics()
                    
                    results['telegram'] = {
                        'success': True,
                        'analytics': engagement_analytics,
                        'bot_status': 'operational' if self.telegram_bot.bot else 'simulation'
                    }
                    
                except Exception as e:
                    logger.error(f"Error in Telegram engagement: {e}")
                    results['telegram'] = {'success': False, 'error': str(e)}
            
            # Generate Phase 5 summary
            successful_features = sum(1 for feature_result in results.values() if feature_result.get('success', False))
            
            results['summary'] = {
                'features_operational': successful_features,
                'total_features': len(self.phase5_features),
                'success_rate': successful_features / len(self.phase5_features) * 100,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Phase 5 features completed: {successful_features}/{len(self.phase5_features)} operational")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Phase 5 features: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def run_ai_strategist_features(self) -> Dict:
        """Run AI Strategist features (Phase 1-4)"""
        logger = self.logger
        try:
            results = {}
            
            # 1. Event Normalization
            if self.ai_strategist_features.get('event_normalization'):
                try:
                    # Simulate event normalization with latest news/market events
                    sample_event = {
                        'title': 'Market volatility increases amid policy uncertainty',
                        'summary': 'Central bank policies creating market instability',
                        'timestamp': datetime.now().isoformat(),
                        'source': 'ai_strategist',
                        'confidence': 0.75
                    }
                    
                    normalized_event = self.event_normalizer.normalize(sample_event)
                    
                    results['event_normalization'] = {
                        'success': True,
                        'events_processed': 1,
                        'sample_event': normalized_event,
                        'status': 'operational'
                    }
                    
                except Exception as e:
                    logger.error(f"Error in event normalization: {e}")
                    results['event_normalization'] = {'success': False, 'error': str(e)}
            
            # 2. Knowledge Graph Analysis
            if self.ai_strategist_features.get('knowledge_graph'):
                try:
                    # Get knowledge graph statistics and insights
                    kg_stats = self.knowledge_graph.get_stats()
                    
                    # Perform graph decay maintenance
                    self.knowledge_graph.decay_and_prune()
                    
                    results['knowledge_graph'] = {
                        'success': True,
                        'graph_stats': kg_stats,
                        'maintenance': 'decay_completed',
                        'status': 'operational'
                    }
                    
                except Exception as e:
                    logger.error(f"Error in knowledge graph: {e}")
                    results['knowledge_graph'] = {'success': False, 'error': str(e)}
            
            # 3. Market Regime Detection
            if self.ai_strategist_features.get('regime_detection'):
                try:
                    # Detect current market regime
                    current_regime = self.regime_engine.detect_current_regime()
                    
                    # Check regime stability
                    regime_stable = self.regime_engine.is_regime_stable()
                    
                    results['regime_detection'] = {
                        'success': True,
                        'current_regime': {
                            'risk': current_regime.risk,
                            'liquidity': current_regime.liquidity,
                            'volatility': current_regime.volatility,
                            'trend': current_regime.trend,
                            'confidence': current_regime.confidence
                        },
                        'stability': regime_stable,
                        'status': 'operational'
                    }
                    
                except Exception as e:
                    logger.error(f"Error in regime detection: {e}")
                    results['regime_detection'] = {'success': False, 'error': str(e)}
            
            # 4. Causal Analysis
            if self.ai_strategist_features.get('causal_analysis'):
                try:
                    # Get active causal hypotheses
                    active_hypotheses = self.causal_engine.get_active_hypotheses(min_confidence=0.6)
                    
                    # Test sample asset pair relationships
                    test_pairs = [('BTC', 'ETH'), ('NIFTY', 'BANKNIFTY')]
                    causal_tests = self.causal_engine.test_hypothesis_batch(test_pairs)
                    
                    results['causal_analysis'] = {
                        'success': True,
                        'active_hypotheses_count': len(active_hypotheses),
                        'sample_hypotheses': active_hypotheses[:3],  # First 3 for demo
                        'recent_tests': causal_tests,
                        'status': 'operational'
                    }
                    
                except Exception as e:
                    logger.error(f"Error in causal analysis: {e}")
                    results['causal_analysis'] = {'success': False, 'error': str(e)}
            
            # Generate AI Strategist summary
            successful_features = sum(1 for feature_result in results.values() if feature_result.get('success', False))
            
            results['summary'] = {
                'features_operational': successful_features,
                'total_features': len(self.ai_strategist_features),
                'success_rate': successful_features / len(self.ai_strategist_features) * 100,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"AI Strategist features completed: {successful_features}/{len(self.ai_strategist_features)} operational")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in AI Strategist features: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def run_phases_5_8_features(self) -> Dict:
        """Run Phases 5-8 features: Decision Policy, Post-Mortem, GPT Integration, Advanced Models"""
        logger = self.logger
        try:
            results = {}
            
            # 1. Decision Policy (Phase 5)
            try:
                # Generate sample causal cards and data for decision making
                causal_cards = [
                    {'hypothesis': 'Market sentiment -> BTC price increase', 'confidence': 0.75},
                    {'hypothesis': 'Fed policy -> Market volatility', 'confidence': 0.65}
                ]
                
                regime = {'risk': 'moderate', 'volatility': 'normal', 'confidence': 0.7}
                sentiment_score = 0.6
                micro_score = 0.55
                
                decision = self.decision_policy.decide(causal_cards, regime, sentiment_score, micro_score)
                
                results['decision_policy'] = {
                    'success': True,
                    'latest_decision': decision,
                    'status': 'operational'
                }
                
            except Exception as e:
                logger.error(f"Error in decision policy: {e}")
                results['decision_policy'] = {'success': False, 'error': str(e)}
            
            # 2. Post-Mortem Analysis (Phase 6)
            try:
                # Get performance metrics
                performance_metrics = self.verdicts.get_performance_metrics()
                
                results['post_mortem'] = {
                    'success': True,
                    'performance_metrics': performance_metrics,
                    'status': 'operational'
                }
                
            except Exception as e:
                logger.error(f"Error in post-mortem analysis: {e}")
                results['post_mortem'] = {'success': False, 'error': str(e)}
            
            # 3. GPT Integration (Phase 7)
            try:
                # Test event typing
                sample_title = "Central Bank raises interest rates"
                sample_summary = "Federal Reserve increases rates by 0.25% amid inflation concerns"
                
                event_type = self.gpt_integrator.type_event(sample_title, sample_summary)
                
                # Test narrative building with sample decision
                sample_decision = {
                    'action': 'paper_trade',
                    'confidence': 0.75,
                    'rationale': ['Market sentiment positive', 'Technical indicators bullish']
                }
                narrative = self.gpt_integrator.build_narrative(sample_decision)
                
                results['gpt_integration'] = {
                    'success': True,
                    'event_typing_sample': event_type,
                    'narrative_sample': narrative,
                    'status': 'operational' if self.gpt_integrator.client else 'simulation'
                }
                
            except Exception as e:
                logger.error(f"Error in GPT integration: {e}")
                results['gpt_integration'] = {'success': False, 'error': str(e)}
            
            # 4. Advanced Models (Phase 8)
            try:
                # Test correlation analysis with sample data
                import pandas as pd
                import numpy as np
                
                # Create sample time series data
                dates = pd.date_range('2024-01-01', periods=50, freq='D')
                sample_data = pd.DataFrame({
                    'BTC': np.random.randn(50).cumsum() + 100,
                    'ETH': np.random.randn(50).cumsum() + 50,
                }, index=dates)
                
                correlation_results = self.advanced_models.correlation_analysis(sample_data)
                
                # Test regime change detection
                regime_changes = self.advanced_models.detect_regime_changes(sample_data['BTC'])
                
                results['advanced_models'] = {
                    'success': True,
                    'correlation_analysis': correlation_results,
                    'regime_changes': regime_changes,
                    'svar_available': self.advanced_models.var_available,
                    'kalman_available': self.advanced_models.kalman_available,
                    'status': 'operational'
                }
                
            except Exception as e:
                logger.error(f"Error in advanced models: {e}")
                results['advanced_models'] = {'success': False, 'error': str(e)}
            
            # Generate Phases 5-8 summary
            successful_features = sum(1 for feature_result in results.values() if feature_result.get('success', False))
            
            results['summary'] = {
                'features_operational': successful_features,
                'total_features': 4,  # Phases 5-8
                'success_rate': successful_features / 4 * 100,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Phases 5-8 features completed: {successful_features}/4 operational")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Phases 5-8 features: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_phase5_status(self) -> Dict:
        """Get comprehensive Phase 5 features status"""
        try:
            return {
                'phase5_features': self.phase5_features,
                'feature_details': {
                    'ai_forecasting': {
                        'status': 'operational' if self.phase5_features.get('ai_forecasting') else 'disabled',
                        'description': 'AI-powered market and geopolitical predictions using Prophet'
                    },
                    'community_simulation': {
                        'status': 'operational' if self.phase5_features.get('community_simulation') else 'disabled',
                        'description': 'Advanced community engagement simulation and analytics'
                    },
                    'user_acquisition': {
                        'status': 'operational' if self.phase5_features.get('user_acquisition') else 'disabled',
                        'description': 'Automated cross-platform user acquisition and funnel optimization'
                    },
                    'telegram_engagement': {
                        'status': 'operational' if self.phase5_features.get('telegram_engagement') else 'disabled',
                        'description': 'Advanced Telegram bot for community engagement and monetization'
                    }
                },
                'total_features_operational': sum(self.phase5_features.values()),
                'total_features': len(self.phase5_features),
                'phase5_version': '1.0.0',
                'status_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting Phase 5 status: {e}")
            return {'error': str(e)}
    
    async def run_analysis_cycle(self) -> Dict:
        """Run complete analysis cycle with alert processing"""
        try:
            self.logger.info("🚀 Starting advanced trading analysis cycle...")
            
            # Run full analysis
            analysis_results = await self.run_full_analysis()
            
            # Process alerts
            alert_results = await self.process_advanced_alerts(analysis_results)
            
            # Combine results
            cycle_results = {
                'cycle_timestamp': datetime.now().isoformat(),
                'analysis_results': analysis_results,
                'alert_results': alert_results,
                'system_status': {
                    'features_operational': analysis_results.get('features_completed', 0),
                    'total_features': 7,
                    'alerts_generated': len(alert_results.get('processed_alerts', [])),
                    'analysis_duration': analysis_results.get('analysis_duration_seconds', 0)
                }
            }
            
            self.logger.info(f"✅ Analysis cycle completed successfully")
            self.logger.info(f"Features operational: {cycle_results['system_status']['features_operational']}/7")
            self.logger.info(f"Alerts generated: {cycle_results['system_status']['alerts_generated']}")
            
            return cycle_results
            
        except Exception as e:
            self.logger.error(f"❌ Analysis cycle failed: {e}")
            return {
                'cycle_timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_status': {'operational': False}
            }

# Global orchestrator instance
orchestrator = None

def get_orchestrator():
    """Get or create the global orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = AdvancedTradingOrchestrator()
    return orchestrator

async def run_advanced_analysis():
    """Standalone function to run advanced analysis"""
    orch = get_orchestrator()
    return await orch.run_analysis_cycle()

# Integration function for the main platform
def integrate_advanced_features():
    """Integration function that can be called from the main platform"""
    return get_orchestrator()

if __name__ == "__main__":
    # CLI mode for testing
    async def main():
        orch = AdvancedTradingOrchestrator()
        results = await orch.run_analysis_cycle()
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(main())