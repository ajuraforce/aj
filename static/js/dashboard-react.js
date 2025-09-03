// Dashboard React Components as Vanilla JavaScript
class DashboardApp {
    constructor() {
        this.state = {
            status: {},
            alerts: { total: 0, new: 0, recent: [] },
            trades: { open: 0, labels: [], data: [], recent: [] },
            portfolio: { pnl: 0, labels: [], data: [] },
            narratives: { top_narratives: [], breaking_news: [], last_updated: '' },
            systemInsights: {
                resonance: { nodes: [], edges: [] },
                scenarios: {},
                timeline: []
            },
            ajxaiChat: {
                isOpen: false,
                messages: [],
                isLoading: false
            }
        };
        
        this.socket = null;
        this.init();
    }

    async init() {
        await this.loadData();
        this.setupWebSocket();
        this.render();
        
        // Ensure Bootstrap Icons font loads
        this.ensureIconsLoad();
        
        // Refresh data every 30 seconds
        setInterval(() => this.loadData(), 30000);
    }

    ensureIconsLoad() {
        // First, add the CSS to ensure proper Bootstrap Icons display
        this.addIconStyles();
        
        // Load Bootstrap Icons font with multiple attempts
        this.loadBootstrapIconsFont();
        
        // Check font loading periodically and retry if needed
        let attempts = 0;
        const checkInterval = setInterval(() => {
            attempts++;
            if (this.areIconsDisplaying() || attempts > 10) {
                clearInterval(checkInterval);
                if (!this.areIconsDisplaying() && attempts > 10) {
                    console.warn('Bootstrap Icons failed to load after multiple attempts');
                }
            } else {
                this.loadBootstrapIconsFont();
            }
        }, 500);
    }

    loadBootstrapIconsFont() {
        // Try multiple CDN sources for Bootstrap Icons
        const fontSources = [
            'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff2',
            'https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.11.3/fonts/bootstrap-icons.woff2',
            'https://unpkg.com/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff2'
        ];
        
        // Try each source
        fontSources.forEach((src, index) => {
            setTimeout(() => {
                const iconFont = new FontFace('bootstrap-icons', `url(${src})`);
                iconFont.load().then(() => {
                    document.fonts.add(iconFont);
                    console.log(`Bootstrap Icons loaded from source ${index + 1}`);
                    this.render();
                }).catch(err => {
                    // Silent fallback - icons will use hardcoded content values
                });
            }, index * 200); // Stagger attempts
        });
    }

    areIconsDisplaying() {
        // Check if any Bootstrap Icon is actually displaying content
        const testIcon = document.createElement('i');
        testIcon.className = 'bi bi-speedometer2';
        testIcon.style.position = 'absolute';
        testIcon.style.top = '-9999px';
        document.body.appendChild(testIcon);
        
        const computed = window.getComputedStyle(testIcon, '::before');
        const hasContent = computed.content && computed.content !== 'none' && computed.content !== '""';
        
        document.body.removeChild(testIcon);
        return hasContent;
    }

    addIconStyles() {
        // Remove any existing Bootstrap Icons styles first
        const existingStyles = document.querySelectorAll('style[data-bootstrap-icons]');
        existingStyles.forEach(style => style.remove());
        
        // Add comprehensive Bootstrap Icons styles
        const style = document.createElement('style');
        style.setAttribute('data-bootstrap-icons', 'true');
        style.textContent = `
            /* Bootstrap Icons - Comprehensive Font Loading */
            @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css');
            
            @font-face {
                font-family: "bootstrap-icons";
                src: url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff2') format('woff2'),
                     url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff') format('woff');
                font-display: block;
                font-weight: normal;
                font-style: normal;
            }
            
            /* Force Bootstrap Icons to display correctly */
            .bi {
                font-family: "bootstrap-icons" !important;
                font-style: normal !important;
                font-weight: normal !important;
                font-variant: normal !important;
                text-transform: none !important;
                line-height: 1 !important;
                vertical-align: -.125em !important;
                -webkit-font-smoothing: antialiased !important;
                -moz-osx-font-smoothing: grayscale !important;
                display: inline-block !important;
                speak: never !important;
            }
            
            .bi::before {
                font-family: "bootstrap-icons" !important;
                display: inline-block !important;
            }
            
            /* Specific icon classes */
            .bi-speedometer2::before { content: "\\f5d9"; }
            .bi-bell::before { content: "\\f1f3"; }
            .bi-arrow-left-right::before { content: "\\f1d4"; }
            .bi-briefcase::before { content: "\\f1f9"; }
            .bi-graph-up::before { content: "\\f384"; }
            .bi-funnel::before { content: "\\f33d"; }
            .bi-clock-history::before { content: "\\f292"; }
            .bi-people::before { content: "\\f4f2"; }
            .bi-gear::before { content: "\\f33e"; }
            .bi-person-circle::before { content: "\\f4fd"; }
            .bi-graph-up-arrow::before { content: "\\f386"; }
            .bi-cpu::before { content: "\\f2a4"; }
            .bi-currency-dollar::before { content: "\\f2a6"; }
            
            /* Navbar and dashboard specific styling */
            .navbar .bi {
                font-size: 1rem !important;
                margin-right: 0.25rem !important;
            }
            
            .kpi-icon.bi {
                font-size: 2.5rem !important;
                margin-bottom: 1rem !important;
            }
        `;
        document.head.appendChild(style);
    }

    async loadData() {
        try {
            const [statusRes, alertsRes, tradesRes, portfolioRes, narrativesRes, resonanceRes, scenariosRes, timelineRes] = await Promise.all([
                this.fetchAPI('/api/status'),
                this.fetchAPI('/api/alerts'),
                this.fetchAPI('/api/trades'),
                this.fetchAPI('/api/portfolio'),
                this.fetchAPI('/api/dashboard/narratives'),
                this.fetchAPI('/api/patterns/resonance'),
                this.fetchAPI('/api/patterns/scenarios'),
                this.fetchAPI('/api/patterns/timeline')
            ]);
            
            // Safely update state with fallback values
            this.state.status = statusRes || {};
            this.state.alerts = alertsRes || { total: 0, new: 0, recent: [] };
            this.state.trades = tradesRes || { open: 0, labels: [], data: [], recent: [] };
            this.state.portfolio = portfolioRes || { pnl: 0, labels: [], data: [] };
            this.state.narratives = narrativesRes || { top_narratives: [], breaking_news: [], last_updated: '' };
            
            // Handle system insights with proper fallbacks
            this.state.systemInsights.resonance = resonanceRes || { nodes: [], edges: [] };
            this.state.systemInsights.scenarios = scenariosRes || {};
            this.state.systemInsights.timeline = timelineRes || [];
            
            console.log('Data loaded successfully:', {
                status: this.state.status,
                resonance: this.state.systemInsights.resonance,
                scenarios: this.state.systemInsights.scenarios,
                timeline: this.state.systemInsights.timeline
            });
            
            this.render();
        } catch (e) {
            console.error('Error loading dashboard data:', e);
            // Show user-friendly error message
            this.showToast('Failed to load dashboard data. Retrying...', 'error');
            // Retry after 5 seconds
            setTimeout(() => this.loadData(), 5000);
        }
    }

    async fetchAPI(endpoint) {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return await response.json();
        } catch (e) {
            console.error(`Error fetching ${endpoint}:`, e);
            return {};
        }
    }

    setupWebSocket() {
        // WebSocket functionality disabled - using polling updates instead
        console.log('Real-time updates via 30-second polling');
    }

    getCurrentActiveClass(path) {
        // Get current pathname from window location
        const currentPath = window.location.pathname;
        return currentPath === path ? 'active' : '';
    }

    render() {
        console.log('🔧 Dashboard render() called - System Features buttons should be visible!');
        const root = document.getElementById('root');
        if (!root) return;

        root.innerHTML = `
            <!-- Navigation -->
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark" role="navigation" aria-label="Main navigation">
                <div class="container-fluid">
                    <!-- Brand/Logo on the left -->
                    <a class="navbar-brand fw-bold" href="/" aria-label="AjxAI Home">
                        <i class="bi bi-graph-up-arrow me-2" aria-hidden="true"></i>AjxAI
                    </a>
                    
                    <!-- Hamburger menu toggle for mobile -->
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" 
                            aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    
                    <div class="collapse navbar-collapse" id="navbarNav">
                        <ul class="navbar-nav me-auto">
                            <li class="nav-item">
                                <a class="nav-link" href="/"><i class="bi bi-house"></i>Dashboard</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/trades"><i class="bi bi-arrow-left-right"></i>Trades</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/portfolio"><i class="bi bi-briefcase"></i>Portfolio</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/analysis"><i class="bi bi-bar-chart"></i>Analysis</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/live-alerts"><i class="bi bi-bell"></i>Alerts</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link active" href="/health"><i class="bi bi-activity"></i>Health</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/community"><i class="bi bi-people"></i>Community</a>
                            </li>
                        </ul>
                        <ul class="navbar-nav">
                            <li class="nav-item dropdown">
                                <a class="nav-link dropdown-toggle d-flex align-items-center" href="#" id="navbarDropdown" 
                                   role="button" data-bs-toggle="dropdown" aria-expanded="false">
                                    <i class="bi bi-person-circle me-1"></i>Account
                                </a>
                                <ul class="dropdown-menu dropdown-menu-end">
                                    <li><a class="dropdown-item" href="/profile"><i class="bi bi-person me-2"></i>Profile</a></li>
                                    <li><a class="dropdown-item" href="/settings"><i class="bi bi-gear me-2"></i>Settings</a></li>
                                    <li><hr class="dropdown-divider"></li>
                                    <li><a class="dropdown-item" href="/screening"><i class="bi bi-funnel me-2"></i>Screening</a></li>
                                    <li><a class="dropdown-item" href="/backtesting"><i class="bi bi-clock-history me-2"></i>Backtesting</a></li>
                                </ul>
                            </li>
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link ${this.getCurrentActiveClass('/analysis')}" href="/analysis" role="menuitem" aria-label="Analysis">
                                    <i class="bi bi-graph-up me-1" aria-hidden="true"></i>Analysis
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link ${this.getCurrentActiveClass('/screening')}" href="/screening" role="menuitem" aria-label="Screening">
                                    <i class="bi bi-funnel me-1" aria-hidden="true"></i>Screening
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link ${this.getCurrentActiveClass('/backtesting')}" href="/backtesting" role="menuitem" aria-label="Backtesting">
                                    <i class="bi bi-clock-history me-1" aria-hidden="true"></i>Backtesting
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link ${this.getCurrentActiveClass('/community')}" href="/community" role="menuitem" aria-label="Community">
                                    <i class="bi bi-people me-1" aria-hidden="true"></i>Community
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link ${this.getCurrentActiveClass('/settings')}" href="/settings" role="menuitem" aria-label="Settings">
                                    <i class="bi bi-gear me-1" aria-hidden="true"></i>Settings
                                </a>
                            </li>
                        </ul>
                        
                        <!-- User profile dropdown on the right -->
                        <ul class="navbar-nav ms-auto">
                            <li class="nav-item dropdown">
                                <a class="nav-link dropdown-toggle" href="#" id="navbarDropdown" role="button" 
                                   data-bs-toggle="dropdown" aria-expanded="false" aria-label="User menu">
                                    <i class="bi bi-person-circle me-1" aria-hidden="true"></i>
                                    <span class="d-md-inline d-none">User</span>
                                </a>
                                <ul class="dropdown-menu dropdown-menu-end" aria-labelledby="navbarDropdown">
                                    <li>
                                        <a class="dropdown-item" href="/profile" aria-label="View Profile">
                                            <i class="bi bi-person me-2" aria-hidden="true"></i>Profile
                                        </a>
                                    </li>
                                    <li><hr class="dropdown-divider" role="separator"></li>
                                    <li>
                                        <a class="dropdown-item" href="/logout" aria-label="Logout">
                                            <i class="bi bi-box-arrow-right me-2" aria-hidden="true"></i>Logout
                                        </a>
                                    </li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
            </nav>
            
            <div class="dashboard-container container-fluid p-4">
                <!-- Header -->
                <div class="row mb-4">
                    <div class="col">
                        <h2>Trading Dashboard</h2>
                        <p class="text-muted">AI-Powered Social Intelligence Platform</p>
                    </div>
                </div>

                <!-- KPI Cards -->
                <div class="row g-4 mb-4">
                    <!-- Platform Status -->
                    <div class="col-12 col-md-6 col-lg-3">
                        <div class="card kpi-card text-center">
                            <div class="card-body">
                                <i class="bi bi-cpu text-primary kpi-icon"></i>
                                <h5 class="card-title">Platform Status</h5>
                                <div class="kpi-value">${this.state.status.status || 'Loading...'}</div>
                                <span class="badge status-badge ${this.state.status.status === 'Running' ? 'bg-success' : 'bg-danger'}">
                                    ${this.state.status.status || '...'}
                                </span>
                                <div class="small text-muted mt-2">${this.state.status.last_backup || ''}</div>
                            </div>
                        </div>
                    </div>

                    <!-- Open Trades -->
                    <div class="col-12 col-md-6 col-lg-3">
                        <div class="card kpi-card text-center">
                            <div class="card-body">
                                <i class="bi bi-graph-up text-success kpi-icon"></i>
                                <h5 class="card-title">Open Trades</h5>
                                <div class="kpi-value">${this.state.trades.open || 0}</div>
                                <div class="chart-container">
                                    <canvas id="trades-chart" width="100" height="30"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Recent Alerts -->
                    <div class="col-12 col-md-6 col-lg-3">
                        <div class="card kpi-card text-center">
                            <div class="card-body">
                                <i class="bi bi-bell text-warning kpi-icon"></i>
                                <h5 class="card-title">Recent Alerts</h5>
                                <div class="kpi-value">${this.state.alerts.total || 0}</div>
                                <span class="badge bg-warning">+${this.state.alerts.new || 0}</span>
                            </div>
                        </div>
                    </div>

                    <!-- Total P&L -->
                    <div class="col-12 col-md-6 col-lg-3">
                        <div class="card kpi-card text-center">
                            <div class="card-body">
                                <i class="bi bi-currency-dollar text-info kpi-icon"></i>
                                <h5 class="card-title">Total P&L</h5>
                                <div class="kpi-value">$${(this.state.portfolio.pnl || 0).toFixed(2)}</div>
                                <div class="chart-container">
                                    <canvas id="portfolio-chart" width="100" height="30"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- News Widget Section -->
                ${this.renderNewsWidget()}

                <!-- System-Level Insights -->
                ${this.renderSystemInsights()}

                <!-- Recent Activity -->
                <div class="row mb-4">
                    <div class="col-lg-8">
                        <div class="card">
                            <div class="card-header">Recent Activity</div>
                            <div class="card-body p-0">
                                <div class="table-responsive">
                                    <table class="table table-hover activity-table mb-0">
                                        <thead class="table-light">
                                            <tr>
                                                <th>Time</th>
                                                <th>Type</th>
                                                <th>Details</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${this.renderRecentActivity()}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            <div class="card-footer text-end">
                                <a href="/live-alerts" class="btn btn-link btn-sm">View All Alerts</a>
                            </div>
                        </div>
                    </div>

                </div>

                <!-- Platform Controls -->
                <div class="card controls-card">
                    <div class="card-header">Platform Controls</div>
                    <div class="card-body text-center">
                        <button class="btn btn-success btn-control" onclick="dashboard.controlPlatform('start')">
                            <i class="bi bi-play-fill me-2"></i>Start
                        </button>
                        <button class="btn btn-danger btn-control" onclick="dashboard.controlPlatform('stop')">
                            <i class="bi bi-stop-fill me-2"></i>Stop
                        </button>
                        <button class="btn btn-primary btn-control" onclick="dashboard.controlPlatform('backup')">
                            <i class="bi bi-cloud-upload me-2"></i>Backup
                        </button>
                    </div>
                </div>
            </div>
            </div>
            
            <!-- AJxAI Floating Chat Button -->
            <div class="ajxai-chat-container" style="position: fixed; bottom: 30px; right: 30px; z-index: 9998;">
                <button class="ajxai-chat-button" id="ajxai-chat-toggle" aria-label="Open AJxAI Chat" 
                        style="position: relative; width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; color: white; box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4); transition: all 0.3s ease; display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer;"
                        onclick="dashboard.toggleAJxAIChat()">
                    💬
                    <span style="font-size: 0.6rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">AJxAI</span>
                </button>
                
                <div class="ajxai-chat-interface" id="ajxai-chat-interface" style="display: none; position: absolute; bottom: 80px; right: 0; width: 380px; height: 500px; background: white; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.15); flex-direction: column; overflow: hidden;">
                    <!-- Header -->
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px 20px; color: white; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h6 class="mb-0 fw-bold text-white">AJxAI Multi-Domain Strategist</h6>
                            <small class="text-white-50">Geopolitics • Markets • Crypto • Sentiment</small>
                        </div>
                        <button class="btn btn-outline-light btn-sm" onclick="dashboard.clearAJxAIChat()" title="Clear conversation">
                            🔄
                        </button>
                    </div>
                    
                    <!-- Messages -->
                    <div id="ajxai-chat-messages" style="flex: 1; padding: 15px; overflow-y: auto; background: #f8f9fa;">
                        <div class="chat-empty-state" style="text-align: center; color: #6c757d; padding: 30px 20px;">
                            💡
                            <p class="mb-1">Ask me about:</p>
                            <small>Market correlations, geopolitical impacts, crypto trends, or trading signals</small>
                        </div>
                    </div>
                    
                    <!-- Input -->
                    <div style="padding: 15px; border-top: 1px solid #e9ecef; background: white;">
                        <div class="input-group">
                            <textarea class="form-control" id="ajxai-message-input" placeholder="Ask AJxAI about market patterns, correlations, or trading signals..." rows="2" style="resize: none; border-radius: 20px; border: 1px solid #e9ecef; padding: 10px 15px; font-size: 0.9rem;"></textarea>
                            <button class="btn btn-primary" id="ajxai-send-btn" onclick="dashboard.sendAJxAIMessage()" style="border-radius: 20px; padding: 10px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none;">
                                ➤
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        console.log('✅ AJxAI Chat button added to dashboard');

        // Initialize charts after rendering
        setTimeout(() => {
            this.initCharts();
            this.initSystemInsights();
        }, 100);
    }

    renderNewsWidget() {
        const narratives = this.state.narratives || {};
        const topNarratives = narratives.top_narratives || [];
        const breakingNews = narratives.breaking_news || [];
        const lastUpdated = narratives.last_updated ? new Date(narratives.last_updated).toLocaleTimeString() : '';

        return `
        <div class="row mb-4">
            <div class="col-12">
                <div class="card news-intelligence-card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h6 class="mb-0">📰 News Intelligence Analytics</h6>
                        <div class="d-flex align-items-center">
                            <span class="badge bg-success me-2" id="newsConnectionStatus">Live</span>
                            <small class="text-muted">Updated: ${lastUpdated}</small>
                        </div>
                    </div>
                    <div class="card-body p-3">
                        <!-- News Trend Chart Section -->
                        <div class="row mb-4">
                            <div class="col-md-8">
                                <div class="chart-container" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 20px;">
                                    <h6 class="text-white mb-3">📈 24-Hour News Activity Trend</h6>
                                    <canvas id="newsTrendChart" width="400" height="200" style="background: rgba(255,255,255,0.1); border-radius: 8px;"></canvas>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="news-metrics-panel">
                                    <div class="metric-card mb-2 p-3" style="background: linear-gradient(45deg, #ff6b6b, #ee5a24); border-radius: 10px; color: white;">
                                        <div class="metric-value" id="totalNewsCount">0</div>
                                        <div class="metric-label">Total Articles</div>
                                    </div>
                                    <div class="metric-card mb-2 p-3" style="background: linear-gradient(45deg, #4ecdc4, #44a08d); border-radius: 10px; color: white;">
                                        <div class="metric-value" id="trendingTopics">0</div>
                                        <div class="metric-label">Topic Clusters</div>
                                    </div>
                                    <div class="metric-card p-3" style="background: linear-gradient(45deg, #45b7d1, #96c93d); border-radius: 10px; color: white;">
                                        <div class="metric-value" id="avgSentiment">0.0</div>
                                        <div class="metric-label">Avg Sentiment</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Topic Clustering Cards Section -->
                        <div class="row mb-3">
                            <div class="col-12">
                                <h6 class="text-primary mb-3">🎯 Topic Clusters & Analysis</h6>
                                <div id="newsTopicCards" class="topic-cards-container">
                                    <div class="text-center text-muted py-4">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Loading news clusters...</span>
                                        </div>
                                        <p class="mt-2">Analyzing news topics with AI clustering...</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Legacy Breaking News Ticker (Enhanced) -->
                        ${breakingNews.length > 0 ? `
                            <div class="row">
                                <div class="col-12">
                                    <div class="breaking-news-panel" style="background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 12px; padding: 15px;">
                                        <h6 class="text-white mb-3">⚡ Real-time News Stream</h6>
                                        <div class="breaking-news-container" style="max-height: 250px; overflow-y: auto;">
                                            ${breakingNews.slice(0, 5).map((news, index) => `
                                                <div class="breaking-news-item mb-2 p-3" style="background: rgba(255,255,255,0.1); border-radius: 8px; backdrop-filter: blur(10px);">
                                                    <div class="d-flex justify-content-between align-items-start">
                                                        <a href="${news.link}" target="_blank" rel="noopener noreferrer" class="text-decoration-none text-white flex-grow-1">
                                                            <h6 class="mb-1">${news.title}</h6>
                                                        </a>
                                                        <span class="sentiment-indicator ms-2 fs-5">
                                                            ${news.sentiment_score > 0.3 ? '📈' : news.sentiment_score < -0.3 ? '📉' : '➡️'}
                                                        </span>
                                                    </div>
                                                    <div class="d-flex justify-content-between align-items-center mt-2">
                                                        <small class="text-white-50">${news.source} • ${new Date(news.published).toLocaleString()}</small>
                                                        <span class="badge ${news.sentiment_score > 0.3 ? 'bg-success' : news.sentiment_score < -0.3 ? 'bg-danger' : 'bg-secondary'} small">
                                                            ${news.sentiment_score.toFixed(2)}
                                                        </span>
                                                    </div>
                                                </div>
                                            `).join('')}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        </div>
        `;
    }

    renderSystemInsights() {
        const insights = this.state.systemInsights || {};
        const scenarios = insights.scenarios || {};
        const timeline = insights.timeline || [];
        const resonance = insights.resonance || { nodes: [], edges: [] };

        return `
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">🎯 System-Level Intelligence</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <!-- Scenario Radar -->
                            <div class="col-lg-4 mb-3">
                                <h6 class="text-primary mb-3">Scenario Probabilities</h6>
                                <div id="scenario-radar-container" style="position: relative; height: 350px;">
                                    ${Object.keys(scenarios).length > 0 ? `
                                        <canvas id="scenario-radar"></canvas>
                                    ` : `
                                        <div class="text-center text-muted py-4">
                                            <i class="bi bi-radar" style="font-size: 2rem;"></i>
                                            <p class="mt-2">No scenario data available</p>
                                        </div>
                                    `}
                                </div>
                            </div>
                            
                            <!-- System Resonance -->
                            <div class="col-lg-4 mb-3">
                                <h6 class="text-success mb-3">System Resonance Map</h6>
                                <div id="resonance-map-container">
                                    ${this.renderResonanceMapCards(resonance)}
                                </div>
                            </div>
                            
                            <!-- Timeline of Echoes -->
                            <div class="col-lg-4 mb-3">
                                <h6 class="text-warning mb-3">Timeline of Echoes</h6>
                                <div class="timeline-echo-container" style="max-height: 300px; overflow-y: auto;">
                                    ${timeline.length > 0 ? timeline.map(entry => `
                                        <div class="timeline-entry mb-2 p-2 border-start border-warning border-3 bg-light">
                                            <div class="d-flex justify-content-between align-items-start">
                                                <span class="fw-bold text-uppercase small">${entry.concept}</span>
                                                <span class="text-muted small">${new Date(entry.time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                                            </div>
                                            <div class="small text-muted">${entry.source}</div>
                                            ${entry.title ? `<div class="small">${entry.title}</div>` : ''}
                                        </div>
                                    `).join('') : `
                                        <div class="text-center text-muted py-4">
                                            <i class="bi bi-clock-history" style="font-size: 2rem;"></i>
                                            <p class="mt-2">No recent concept echoes</p>
                                        </div>
                                    `}
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer bg-light">
                        <div class="d-flex justify-content-between align-items-center">
                            <small class="text-muted">
                                <i class="bi bi-info-circle me-1"></i>
                                AI analysis based on ${this.state.status.features_operational || 15} active features
                            </small>
                            <button class="btn btn-outline-primary btn-sm" onclick="window.location.href='/features'">
                                <i class="bi bi-grid me-1"></i>
                                View All Features
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        `;
    }

    renderRecentActivity() {
        if (!this.state.alerts.recent || this.state.alerts.recent.length === 0) {
            return '<tr><td colspan="3" class="text-center text-muted py-3">No recent activity</td></tr>';
        }

        return this.state.alerts.recent.map(alert => `
            <tr>
                <td>${alert.time || 'N/A'}</td>
                <td><span class="badge bg-secondary">${alert.type || 'Alert'}</span></td>
                <td>${alert.details || alert.message || 'No details'}</td>
            </tr>
        `).join('');
    }

    initCharts() {
        if (window.Chart) {
            // Initialize trades chart
            const tradesCtx = document.getElementById('trades-chart');
            if (tradesCtx && this.state.trades.labels && this.state.trades.data) {
                new Chart(tradesCtx, {
                    type: 'line',
                    data: {
                        labels: this.state.trades.labels,
                        datasets: [{
                            data: this.state.trades.data,
                            borderColor: '#198754',
                            fill: false,
                            pointRadius: 0,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: { x: { display: false }, y: { display: false } },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            // Initialize portfolio chart
            const portfolioCtx = document.getElementById('portfolio-chart');
            if (portfolioCtx && this.state.portfolio.labels && this.state.portfolio.data) {
                new Chart(portfolioCtx, {
                    type: 'line',
                    data: {
                        labels: this.state.portfolio.labels,
                        datasets: [{
                            data: this.state.portfolio.data,
                            borderColor: '#0d6efd',
                            backgroundColor: 'rgba(13,110,253,0.2)',
                            fill: true,
                            pointRadius: 0,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: { x: { display: false }, y: { display: false } },
                        plugins: { legend: { display: false } }
                    }
                });
            }
        }
    }

    initSystemInsights() {
        try {
            console.log('Initializing system insights...');
            
            // Initialize Scenario Radar Chart
            this.initScenarioRadar();
            
            // Resonance Map is now rendered as cards in the template - no D3.js needed
            console.log('System Resonance Map: Using card-based layout');
            
        } catch (error) {
            console.error('Error initializing system insights:', error);
        }
    }
    
    
    renderResonanceMapCards(resonance) {
        if (!resonance || !resonance.nodes || resonance.nodes.length === 0) {
            return `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-diagram-3" style="font-size: 2rem;"></i>
                    <p class="mt-2">No resonance patterns detected</p>
                    <small>System learning from market patterns</small>
                </div>
            `;
        }

        // Sort nodes by weight (highest resonance first)
        const sortedNodes = resonance.nodes.sort((a, b) => (b.weight || 0) - (a.weight || 0));
        
        return `
            <div class="resonance-cards-container">
                <div class="row g-2">
                    ${sortedNodes.map((node, index) => {
                        const connections = resonance.edges.filter(e => 
                            e.source === node.id || e.target === node.id
                        ).length;
                        const resonanceStrength = Math.round((node.weight || 0) * 100);
                        
                        // Create gradient background based on sector/group
                        const gradients = {
                            'financial': 'linear-gradient(135deg, #fbab7e 0%, #f7ce68 100%)',
                            'technology': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
                            'policy': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                            'economic': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                            'default': 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)'
                        };
                        const gradient = gradients[node.group] || gradients['default'];
                        
                        return `
                            <div class="col-6 mb-2">
                                <div class="resonance-card p-3 border-0 rounded-3 shadow-sm" style="
                                    background: ${gradient};
                                    color: #333;
                                    min-height: 100px;
                                    display: flex;
                                    flex-direction: column;
                                    justify-content: space-between;
                                ">
                                    <div>
                                        <div class="fw-bold text-capitalize" style="font-size: 1.1em;">${node.id}</div>
                                        <div class="text-muted small text-capitalize">${node.group || 'Unknown'}</div>
                                    </div>
                                    <div class="mt-2">
                                        <div class="d-flex justify-content-between align-items-center">
                                            <span class="badge text-white fw-bold px-3 py-2" style="
                                                background: linear-gradient(90deg, #2196f3, #4caf50);
                                                font-size: 0.9em;
                                                border-radius: 8px;
                                            ">${resonanceStrength}%</span>
                                            <span class="small text-muted">${connections} links</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
                <div class="text-center mt-3">
                    <small class="text-muted">
                        <i class="bi bi-info-circle me-1"></i>
                        ${resonance.edges.length} connections between ${resonance.nodes.length} concepts
                    </small>
                </div>
            </div>
        `;
    }

    showResonanceMapFallback() {
        // This method is no longer used - cards are now the primary display
        const resonance = this.state.systemInsights.resonance || { nodes: [], edges: [] };
        const container = document.getElementById('resonance-map-container');
        if (container) {
            container.innerHTML = this.renderResonanceMapCards(resonance);
        }
    }

    initScenarioRadar() {
        const scenarios = this.state.systemInsights.scenarios || {};
        const radarCanvas = document.getElementById('scenario-radar');
        
        if (!radarCanvas || !window.Chart || Object.keys(scenarios).length === 0) {
            return;
        }

        const labels = Object.keys(scenarios).map(key => 
            key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
        );
        const values = Object.values(scenarios).map(v => v * 100);
        
        // Find highest probability for highlighting
        const maxValue = Math.max(...values);
        const maxIndex = values.indexOf(maxValue);
        
        // Create gradient background
        const ctx = radarCanvas.getContext('2d');
        const gradient = ctx.createRadialGradient(
            radarCanvas.width / 2, radarCanvas.height / 2, 0,
            radarCanvas.width / 2, radarCanvas.height / 2, radarCanvas.width / 2
        );
        gradient.addColorStop(0, 'rgba(60, 130, 246, 0.4)');
        gradient.addColorStop(0.7, 'rgba(60, 130, 246, 0.2)');
        gradient.addColorStop(1, 'rgba(60, 130, 246, 0.05)');

        // Destroy existing chart if it exists
        if (this.scenarioRadarChart) {
            this.scenarioRadarChart.destroy();
        }

        this.scenarioRadarChart = new Chart(radarCanvas, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Scenario Probability (%)',
                    data: values,
                    backgroundColor: gradient,
                    borderColor: 'rgba(60, 130, 246, 1)',
                    borderWidth: 3,
                    pointBackgroundColor: values.map((_, index) => 
                        index === maxIndex ? 'rgba(236, 64, 122, 0.9)' : 'rgba(60, 130, 246, 0.8)'
                    ),
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: values.map((_, index) => index === maxIndex ? 8 : 6),
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(236, 64, 122, 1)',
                    pointHoverRadius: 10,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1200,
                    easing: 'easeInOutQuart'
                },
                interaction: {
                    intersect: false,
                    mode: 'point'
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        min: 0,
                        angleLines: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)',
                            lineWidth: 1
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)',
                            lineWidth: 1
                        },
                        pointLabels: {
                            font: {
                                size: 14,
                                weight: 'bold',
                                family: 'system-ui, -apple-system, sans-serif'
                            },
                            color: '#2d3748',
                            padding: 8
                        },
                        ticks: {
                            stepSize: 20,
                            color: '#718096',
                            font: {
                                size: 12
                            },
                            backdropColor: 'rgba(255, 255, 255, 0.8)',
                            backdropPadding: 4
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: {
                                size: 14,
                                weight: 'bold'
                            },
                            color: '#2d3748',
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(60, 130, 246, 1)',
                        borderWidth: 2,
                        cornerRadius: 8,
                        padding: 12,
                        titleFont: {
                            size: 14,
                            weight: 'bold'
                        },
                        bodyFont: {
                            size: 13
                        },
                        callbacks: {
                            title: function(context) {
                                return context[0].label;
                            },
                            label: function(context) {
                                const value = context.parsed.r;
                                const isHighest = context.dataIndex === maxIndex;
                                return `Probability: ${value.toFixed(1)}%${isHighest ? ' (Highest Risk)' : ''}`;
                            },
                            afterLabel: function(context) {
                                const value = context.parsed.r;
                                if (value >= 80) return 'Risk Level: Critical';
                                if (value >= 60) return 'Risk Level: High';
                                if (value >= 40) return 'Risk Level: Medium';
                                return 'Risk Level: Low';
                            }
                        }
                    }
                }
            }
        });
    }


    async controlPlatform(action) {
        try {
            const response = await fetch(`/api/${action}`, { method: 'POST' });
            const result = await response.json();
            
            if (result.success) {
                this.showToast(`Platform ${action} successful`, 'success');
                // Refresh data after control action
                setTimeout(() => this.loadData(), 2000);
            } else {
                this.showToast(`Platform ${action} failed: ${result.message}`, 'error');
            }
        } catch (e) {
            this.showToast(`Error: ${e.message}`, 'error');
        }
    }

    showToast(message, type = 'info') {
        // Simple toast notification
        const toast = document.createElement('div');
        toast.className = `alert alert-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
        toast.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span>${message}</span>
                <button type="button" class="btn-close" onclick="this.parentElement.parentElement.remove()"></button>
            </div>
        `;
        document.body.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) toast.remove();
        }, 5000);
    }

    // AJxAI Chat Methods
    toggleAJxAIChat() {
        const chatInterface = document.getElementById('ajxai-chat-interface');
        const chatButton = document.getElementById('ajxai-chat-toggle');
        
        if (chatInterface && chatButton) {
            const isOpen = chatInterface.style.display === 'flex';
            
            if (isOpen) {
                chatInterface.style.display = 'none';
                chatButton.innerHTML = `
                    <i class="bi bi-chat-dots" style="font-size: 1.2rem; margin-bottom: 2px;"></i>
                    <span style="font-size: 0.6rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">AJxAI</span>
                `;
            } else {
                chatInterface.style.display = 'flex';
                chatButton.innerHTML = `
                    <i class="bi bi-x-lg" style="font-size: 1.2rem; margin-bottom: 2px;"></i>
                    <span style="font-size: 0.6rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Close</span>
                `;
                
                // Focus on input
                setTimeout(() => {
                    const input = document.getElementById('ajxai-message-input');
                    if (input) input.focus();
                }, 100);
            }
        }
    }

    async sendAJxAIMessage() {
        const messageInput = document.getElementById('ajxai-message-input');
        const sendBtn = document.getElementById('ajxai-send-btn');
        
        if (!messageInput || !sendBtn) return;
        
        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to chat
        this.addChatMessage('user', message);
        messageInput.value = '';
        
        // Show loading
        sendBtn.innerHTML = '<div class="spinner-border spinner-border-sm"></div>';
        sendBtn.disabled = true;
        messageInput.disabled = true;
        
        try {
            const response = await fetch('/api/ajxai/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    context: 'dashboard'
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.addChatMessage('assistant', data.response || 'I apologize, but I cannot respond right now.');
            } else {
                this.addChatMessage('assistant', 'Sorry, I encountered an error. Please try again.', true);
            }
        } catch (error) {
            console.error('Chat request failed:', error);
            this.addChatMessage('assistant', 'Connection error. Please check your network and try again.', true);
        } finally {
            // Reset UI
            sendBtn.innerHTML = '<i class="bi bi-send"></i>';
            sendBtn.disabled = false;
            messageInput.disabled = false;
        }
    }

    addChatMessage(role, content, isError = false) {
        const messagesContainer = document.getElementById('ajxai-chat-messages');
        if (!messagesContainer) return;
        
        // Clear empty state if it exists
        const emptyState = messagesContainer.querySelector('.chat-empty-state');
        if (emptyState) {
            emptyState.remove();
        }
        
        const timestamp = new Date().toLocaleTimeString();
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role} ${isError ? 'error' : ''}`;
        messageDiv.style.marginBottom = '15px';
        messageDiv.style.display = 'flex';
        messageDiv.style.flexDirection = 'column';
        messageDiv.style.alignItems = role === 'user' ? 'flex-end' : 'flex-start';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.style.maxWidth = '80%';
        contentDiv.style.padding = '10px 15px';
        contentDiv.style.borderRadius = '18px';
        contentDiv.style.fontSize = '0.9rem';
        contentDiv.style.lineHeight = '1.4';
        contentDiv.style.whiteSpace = 'pre-wrap';
        
        if (role === 'user') {
            contentDiv.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            contentDiv.style.color = 'white';
            contentDiv.style.borderBottomRightRadius = '5px';
        } else {
            contentDiv.style.background = isError ? '#f8d7da' : 'white';
            contentDiv.style.border = '1px solid #e9ecef';
            contentDiv.style.boxShadow = '0 2px 5px rgba(0,0,0,0.05)';
            contentDiv.style.borderBottomLeftRadius = '5px';
            contentDiv.style.color = isError ? '#842029' : 'inherit';
        }
        
        contentDiv.textContent = content;
        
        const timeDiv = document.createElement('small');
        timeDiv.style.margin = '5px 15px 0';
        timeDiv.style.color = '#6c757d';
        timeDiv.style.fontSize = '0.75rem';
        timeDiv.textContent = timestamp;
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    clearAJxAIChat() {
        const messagesContainer = document.getElementById('ajxai-chat-messages');
        if (messagesContainer) {
            messagesContainer.innerHTML = `
                <div class="chat-empty-state" style="text-align: center; color: #6c757d; padding: 30px 20px;">
                    <i class="bi bi-lightbulb" style="font-size: 3rem; margin-bottom: 15px; color: #667eea;"></i>
                    <p class="mb-1">Ask me about:</p>
                    <small>Market correlations, geopolitical impacts, crypto trends, or trading signals</small>
                </div>
            `;
        }
    }
}

// Global error handler to catch script errors
window.addEventListener('error', (event) => {
    console.error('Global script error caught:', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error
    });
});

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('DOM loaded, initializing dashboard...');
        console.log('Available libraries:', {
            React: !!window.React,
            Chart: !!window.Chart,
            d3: !!window.d3,
            axios: !!window.axios
        });
        
        window.dashboard = new DashboardApp();
        console.log('Dashboard app initialized successfully');
    } catch (error) {
        console.error('Error initializing dashboard app:', error);
        
        // Show fallback UI
        const root = document.getElementById('root');
        if (root) {
            root.innerHTML = `
                <div class="container-fluid p-4">
                    <div class="alert alert-danger" role="alert">
                        <h4 class="alert-heading">Dashboard Initialization Error</h4>
                        <p>There was an error loading the dashboard. Please refresh the page.</p>
                        <hr>
                        <p class="mb-0">
                            <button class="btn btn-primary" onclick="window.location.reload()">
                                <i class="bi bi-arrow-clockwise me-2"></i>Refresh Page
                            </button>
                        </p>
                    </div>
                </div>
            `;
        }
    }
});