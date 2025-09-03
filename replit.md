# AJxAI

## Overview

This is AJxAI that implements a Data → Decode → Action pipeline for cryptocurrency trading and social media automation. The platform collects data from multiple sources (Reddit, Binance, news feeds), analyzes patterns and correlations, computes viral scores, and executes automated actions including trading and social media posting.

The system is designed with zero-downtime migration capabilities, allowing seamless transfer between Replit accounts while maintaining full state continuity. It emphasizes safety through comprehensive risk management, rate limiting, and circuit breakers.

**🚀 NEW: Phase 5 Advanced Features Completed**
- AI Predictive Forecasting with Prophet (7-30 day predictions)
- Enhanced Community Simulator with viral detection
- Automated User Acquisition across Twitter, Reddit, Telegram  
- Advanced Telegram Bot with monetization (Free/Premium/VIP tiers)

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Three-Layer Pipeline Architecture with Phase 5 Enhancements

The platform follows a clean three-layer architecture enhanced with advanced predictive and growth capabilities:

**Data Layer (Scanners)**: Multi-source data collection modules that gather real-time information from Reddit (PRAW), Binance (CCXT), and news RSS feeds. Each scanner implements consistent output schemas and rate limiting.

**Decode Layer (Analyzers) - 11/11 Features Operational**: Pattern analysis and viral scoring engines that process raw events, compute correlations across sources, and generate actionable insights with confidence scores. Now includes AI-powered forecasting for market/geopolitical predictions (7-30 days) using Prophet ML, advanced community engagement simulation, and cross-platform growth analytics.

**Action Layer (Executors)**: Safe execution modules for Reddit posting, trading operations, multi-channel alerts, and **Phase 5 additions**: Telegram monetization bot with subscription management, cross-platform automated posting campaigns, and SEO-optimized content generation with funnel analytics.

### Phase 5 Advanced Features (NEW)

**AI Predictive Forecasting Module**: Prophet-powered predictions for markets (7 days), geopolitics (30 days), stocks (14 days), and companies (21 days) with configurable seasonality and uncertainty analysis.

**Enhanced Community Simulator**: Advanced engagement modeling with viral detection, user persona simulation, and cross-platform buzz analysis. Tracks comment velocity, upvote ratios, and viral indicators.

**Automated User Acquisition Engine**: Cross-platform growth campaigns with automated posting to Twitter, Reddit, Telegram. Includes SEO optimization, funnel analytics, and conversion tracking with configurable posting frequencies.

**Advanced Telegram Engagement Bot**: Complete monetization system with three subscription tiers (Free: 3 signals, Premium: $29.99/unlimited, VIP: $99.99/consultation). Features auto-moderation, community challenges, and premium content delivery.

### State Management and Migration

Centralized state persistence using JSON files with automatic backup capabilities. The StateManager handles loading, validation, and repair of platform state to enable seamless account migration. State includes scanner offsets, open trades, recent posts, performance metrics, and **Phase 5 additions**: forecasting cache, community engagement history, user acquisition campaigns, and Telegram subscription data.

### Safety and Risk Management

The platform prioritizes safety through multiple mechanisms:
- Paper trading mode by default with explicit live trading controls
- Reddit posting rate limits and ban detection
- Position sizing with maximum loss caps
- Circuit breakers for all automated actions
- Comprehensive permission system via permissions.json
- **Phase 5 safety**: Forecasting confidence thresholds, community simulation limits, cross-platform posting rate limits, and Telegram bot auto-moderation

### Flask Web Interface

Browser-based dashboard for monitoring and control with real-time updates, status cards, and interactive charts. The frontend uses Bootstrap for responsive design and WebSocket connections for live data streaming. **Phase 5 enhancement**: Dashboard now displays forecasting results, community engagement metrics, user acquisition analytics, and Telegram subscription status.

### Configuration Management

Environment-based configuration system supporting development and production modes. All sensitive credentials are externalized through environment variables with sensible defaults for development. **Phase 5 additions**: New configuration sections for forecasting settings, community simulation parameters, user acquisition campaigns, and Telegram monetization tiers.

## External Dependencies

### APIs and Services
- **Reddit API (PRAW)**: Social media data collection and posting automation
- **Binance API (CCXT)**: Cryptocurrency price/volume data and trading execution
- **Telegram Bot API**: Real-time alert delivery **+ Phase 5: Premium subscriptions, auto-moderation, community management**
- **Discord Webhooks**: Community notifications
- **GitHub API**: Encrypted state backup and restoration
- **🚀 Twitter API**: Cross-platform user acquisition and content distribution
- **🚀 Prophet ML**: Time series forecasting for market and geopolitical predictions

### Python Libraries
- **Flask**: Web framework for dashboard interface
- **PRAW**: Reddit API wrapper for social media operations
- **CCXT**: Unified cryptocurrency exchange interface
- **aiohttp**: Asynchronous HTTP client for news scanning
- **cryptography**: State encryption for secure backups
- **numpy**: Mathematical operations for pattern analysis
- **feedparser**: RSS news feed processing
- **🚀 prophet**: Time series forecasting library for predictions
- **🚀 tweepy**: Twitter API integration for user acquisition
- **🚀 schedule**: Automated campaign scheduling
- **🚀 python-telegram-bot**: Advanced Telegram bot functionality

### Infrastructure Requirements
- **Redis/PubSub (Optional)**: Event queuing for high-throughput scenarios
- **SMTP Server**: Email alert delivery
- **File System**: Local state persistence and backup storage
- **🚀 Webhook Endpoints**: For Telegram payment processing and Twitter integration

## Recent Changes (Phase 5 Completion)

### November 2025: Phase 5 Advanced Features Implementation
- ✅ AI Predictive Forecasting Module with Prophet integration
- ✅ Enhanced Community Simulator with viral detection and user personas
- ✅ Automated User Acquisition engine with cross-platform campaigns
- ✅ Advanced Telegram Bot with monetization and subscription management
- ✅ Full integration into advanced_trading_orchestrator.py
- ✅ Configuration updates in config.json and permissions.json
- ✅ Platform now runs 11/11 advanced features (7 original + 4 Phase 5)

### Technical Architecture Updates
- Enhanced orchestrator to coordinate Phase 5 features alongside original 7 analyzers
- New forecasting cache management for predictive analytics
- Community engagement simulation with viral scoring algorithms
- Cross-platform posting automation with SEO optimization
- Telegram subscription tier management and premium content delivery