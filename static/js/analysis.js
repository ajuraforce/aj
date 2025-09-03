// Modern Analytics Platform - Analysis Page
class AnalysisApp {
    constructor() {
        this.state = {
            signals: { signals: [], labels: [], data: [] },
            sentiment: { sentiment: [] },
            flow: { flow: [], labels: [], data: [] },
            portfolio: { allocation: [] },
            regimes: { regime: '', confidence: 0 },
            correlations: { correlations: [] },
            timeframes: { timeframes: [] },
            loading: true,
            expandedCard: null,
            hoveredCard: null
        };
        
        this.charts = {};
        this.init();
    }

    async init() {
        await this.loadAnalysis();
        this.render();
        
        // Refresh data every 30 seconds
        setInterval(() => this.loadAnalysis(), 30000);
    }

    async loadAnalysis() {
        this.state.loading = true;
        this.render();
        
        try {
            const [signalsRes, sentimentRes, flowRes, portfolioRes, regimesRes, correlationsRes, timeframesRes] = await Promise.all([
                this.fetchAPI('/api/analysis/signals'),
                this.fetchAPI('/api/analysis/sentiment'),
                this.fetchAPI('/api/analysis/flow'),
                this.fetchAPI('/api/analysis/portfolio'),
                this.fetchAPI('/api/analysis/regimes'),
                this.fetchAPI('/api/analysis/correlations'),
                this.fetchAPI('/api/analysis/timeframes')
            ]);
            
            this.state.signals = signalsRes;
            this.state.sentiment = sentimentRes;
            this.state.flow = flowRes;
            this.state.portfolio = portfolioRes;
            this.state.regimes = regimesRes;
            this.state.correlations = correlationsRes;
            this.state.timeframes = timeframesRes;
            this.state.loading = false;
            
            this.render();
            this.initializeCharts();
        } catch (e) {
            console.error('Error loading analysis data:', e);
            this.state.loading = false;
            this.render();
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

    toggleCard(cardId) {
        this.state.expandedCard = this.state.expandedCard === cardId ? null : cardId;
        this.render();
        setTimeout(() => this.initializeCharts(), 100);
    }

    setHoveredCard(cardId) {
        this.state.hoveredCard = cardId;
        this.render();
    }

    clearHoveredCard() {
        this.state.hoveredCard = null;
        this.render();
    }

    render() {
        const root = document.getElementById('root');
        if (!root) return;

        root.innerHTML = `
            <!-- Navigation -->
            <nav class="navbar navbar-expand-lg navbar-dark" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;">
                <div class="container-fluid">
                    <a class="navbar-brand fw-bold" href="/">
                        <i class="bi bi-graph-up-arrow me-2"></i>AjxAI Trading Platform
                    </a>
                    
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
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
                                <a class="nav-link active" href="/analysis"><i class="bi bi-bar-chart"></i>Analysis</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/live-alerts"><i class="bi bi-bell"></i>Alerts</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/health"><i class="bi bi-activity"></i>Health</a>
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
                            <li class="nav-item">
                                <a class="nav-link" href="/health"><i class="bi bi-shield-check me-1"></i>Health</a>
                            </li>
                        </ul>
                    </div>
                </div>
            </nav>
            
            <!-- Modern Analytics Platform Layout -->
            <div class="container-fluid" style="background-color: #f0f2f5; min-height: 100vh; padding: 2rem 0;">
                <div class="container-xl">
                    <!-- Page Header -->
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <div>
                            <h2 class="mb-1" style="color: #495057; font-weight: 600;">Advanced Analysis Platform</h2>
                            <p class="text-muted mb-0">Real-time trading intelligence and market insights</p>
                        </div>
                        <button class="btn btn-primary" onclick="analysisApp.loadAnalysis()" ${this.state.loading ? 'disabled' : ''}>
                            ${this.state.loading ? '<span class="spinner-border spinner-border-sm me-2"></span>Loading...' : '<i class="bi bi-arrow-clockwise me-2"></i>Refresh Data'}
                        </button>
                    </div>

                    ${this.state.loading ? this.renderLoading() : this.renderAnalysisGrid()}
                </div>
            </div>
        `;

        this.attachEventHandlers();
    }

    async processAnalysis(event) {
        event.preventDefault();
        
        const text = document.getElementById('analysisText').value.trim();
        if (!text) {
            this.showError('Please enter analysis text');
            return;
        }
        
        const paperMode = document.getElementById('paperMode').checked;
        const autoExecute = document.getElementById('autoExecute').checked;
        
        this.setLoading(true);
        this.hideError();
        
        try {
            const response = await fetch('/api/analyze/paste', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    mode: paperMode ? 'paper' : 'live',
                    auto_execute: autoExecute
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to process analysis');
            }
            
            const data = await response.json();
            this.displayResults(data);
            document.getElementById('pipeline-status').style.display = 'inline-block';
            
        } catch (error) {
            console.error('Analysis processing failed:', error);
            this.showError(error.message || 'Failed to process analysis');
        } finally {
            this.setLoading(false);
        }
    }

    displayResults(data) {
        const resultsDiv = document.getElementById('analysisResults');
        
        let html = `
            <div class="mb-3">
                <small class="text-muted">
                    <i class="bi bi-clock me-1"></i>Processing Time: ${data.processing_time} | 
                    <i class="bi bi-broadcast me-1"></i>Active Monitoring: ${data.active_monitoring} signals
                </small>
            </div>
        `;
        
        if (data.parsed.trades.length > 0) {
            html += `
                <div class="mb-4">
                    <h6 class="fw-semibold mb-3">
                        <i class="bi bi-graph-up me-2"></i>Extracted Trades (${data.parsed.trades.length})
                    </h6>
                    <div style="max-height: 300px; overflow-y: auto;">
            `;
            
            data.parsed.trades.forEach((trade, idx) => {
                const directionClass = trade.direction === 'LONG' ? 'bg-success' : 'bg-danger';
                const statusClass = trade.status === 'triggered' ? 'bg-warning text-dark' : 
                                  trade.status === 'pending' ? 'bg-secondary' : 'bg-info';
                
                html += `
                    <div class="card mb-2 border-start border-primary border-3">
                        <div class="card-body py-2 px-3">
                            <div class="d-flex justify-content-between align-items-start mb-1">
                                <div>
                                    <strong class="text-dark">${trade.asset}</strong>
                                    <span class="badge ms-2 ${directionClass}">${trade.direction}</span>
                                    <span class="badge ms-1 ${statusClass}">${trade.status}</span>
                                </div>
                                <small class="text-muted fw-semibold">R:R ${trade.calculated_rr || trade.rr}</small>
                            </div>
                            <small class="text-muted">
                                <i class="bi bi-target me-1"></i>Entry: ${trade.entry_zone} | 
                                <i class="bi bi-shield-x me-1"></i>Stop: ${trade.stop} | 
                                <i class="bi bi-bullseye me-1"></i>Targets: ${trade.targets.join('/')}
                                ${trade.current_price ? ` | <i class="bi bi-currency-dollar me-1"></i>Current: $${trade.current_price.toLocaleString()}` : ''}
                            </small>
                        </div>
                    </div>
                `;
            });
            
            html += '</div></div>';
        }
        
        if (data.executed_trades.length > 0) {
            html += `
                <div class="mb-4">
                    <h6 class="text-success fw-semibold mb-3">
                        <i class="bi bi-check-circle me-2"></i>Executed Trades (${data.executed_trades.length})
                    </h6>
            `;
            
            data.executed_trades.forEach((trade, idx) => {
                html += `
                    <div class="alert alert-success py-2 px-3 mb-2">
                        <small>
                            <i class="bi bi-check-circle-fill me-2"></i>
                            <strong>${trade.symbol}</strong> executed at $${trade.price} 
                            (Qty: ${trade.quantity})
                        </small>
                    </div>
                `;
            });
            
            html += '</div>';
        }
        
        if (data.parsed.invalidations.length > 0) {
            html += `
                <div class="mb-3">
                    <h6 class="text-warning fw-semibold mb-3">
                        <i class="bi bi-exclamation-triangle me-2"></i>Invalidation Conditions
                    </h6>
            `;
            
            data.parsed.invalidations.forEach((inv, idx) => {
                html += `
                    <div class="alert alert-warning py-2 px-3 mb-2">
                        <small><i class="bi bi-exclamation-triangle me-2"></i>${inv}</small>
                    </div>
                `;
            });
            
            html += '</div>';
        }
        
        resultsDiv.innerHTML = html;
    }

    setLoading(loading) {
        const btn = document.getElementById('processBtn');
        const spinner = document.getElementById('processSpinner');
        
        if (btn && spinner) {
            if (loading) {
                btn.disabled = true;
                spinner.style.display = 'inline-block';
                btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>Processing...`;
            } else {
                btn.disabled = false;
                spinner.style.display = 'none';
                btn.innerHTML = `<i class="bi bi-cpu me-2"></i>Process Analysis`;
            }
        }
    }

    showError(message) {
        const errorDiv = document.getElementById('analysisError');
        if (errorDiv) {
            errorDiv.innerHTML = `<i class="bi bi-exclamation-triangle me-2"></i>${message}`;
            errorDiv.style.display = 'block';
        }
    }

    hideError() {
        const errorDiv = document.getElementById('analysisError');
        if (errorDiv) {
            errorDiv.style.display = 'none';
        }
    }

    clearAnalysis() {
        const textArea = document.getElementById('analysisText');
        const resultsDiv = document.getElementById('analysisResults');
        const statusBadge = document.getElementById('pipeline-status');
        
        if (textArea) textArea.value = '';
        if (statusBadge) statusBadge.style.display = 'none';
        if (resultsDiv) {
            resultsDiv.innerHTML = `
                <div class="text-center text-muted py-5">
                    <i class="bi bi-upload" style="font-size: 3rem; opacity: 0.5;"></i>
                    <p class="mt-3 mb-0">Paste your trading analysis to see parsed trades and automated signals</p>
                    <small class="text-muted">The system will extract trades, map to live data, and create signals</small>
                </div>
            `;
        }
        this.hideError();
    }

    renderLoading() {
        return `
            <div class="text-center p-5">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3">Loading analysis data...</p>
            </div>
        `;
    }

    renderAnalysisGrid() {
        return `
            <!-- Key Metrics Overview Cards -->
            <div class="row mb-4">
                <div class="col-lg-3 col-md-6 mb-3">
                    ${this.renderMetricCard('regime', 'Market Regime', 'bi-thermometer-half', this.getRegimeDisplay(), 'bg-primary')}
                </div>
                <div class="col-lg-3 col-md-6 mb-3">
                    ${this.renderMetricCard('signals', 'Active Signals', 'bi-broadcast', `${(this.state.signals.signals || []).length}`, 'bg-success')}
                </div>
                <div class="col-lg-3 col-md-6 mb-3">
                    ${this.renderMetricCard('flow', 'Institutional Flow', 'bi-building', this.getFlowDirection(), 'bg-info')}
                </div>
                <div class="col-lg-3 col-md-6 mb-3">
                    ${this.renderMetricCard('sentiment', 'Market Sentiment', 'bi-heart', this.getSentimentOverall(), 'bg-warning')}
                </div>
            </div>

            <!-- Analysis-to-Action Pipeline -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card" style="border: none; border-radius: 0.75rem; box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); border-left: 4px solid #667eea;">
                        <div class="card-header bg-gradient-primary text-white" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                            <div class="d-flex align-items-center justify-content-between">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-magic me-2" style="font-size: 1.25rem;"></i>
                                    <h5 class="mb-0 fw-bold">Analysis-to-Action Pipeline</h5>
                                </div>
                                <span id="pipeline-status" class="badge bg-light text-dark" style="display: none;">Active</span>
                            </div>
                        </div>
                        <div class="card-body p-4">
                            <div class="row">
                                <!-- Input Panel -->
                                <div class="col-md-6">
                                    <div class="card border-0 bg-light h-100">
                                        <div class="card-header bg-transparent border-0 pb-2">
                                            <h6 class="mb-0 fw-semibold text-dark">
                                                <i class="bi bi-clipboard me-2"></i>Paste Trading Analysis
                                            </h6>
                                        </div>
                                        <div class="card-body pt-2">
                                            <form onsubmit="analysisApp.processAnalysis(event)">
                                                <div class="mb-3">
                                                    <textarea id="analysisText" class="form-control" rows="8" 
                                                        placeholder="Paste your trading analysis here...

Example:
* NIFTY (Trade 1)
Direction: Long
Entry Zone: 24,600–24,620
Stop: 24,540
Targets: 24,750/24,800
R:R: 1:2.5

Invalidation: Daily close below 24,500" 
                                                        style="font-family: 'Monaco', 'Consolas', monospace; font-size: 0.875rem;"></textarea>
                                                </div>
                                                
                                                <div class="row mb-3">
                                                    <div class="col-md-6">
                                                        <div class="form-check form-switch">
                                                            <input class="form-check-input" type="checkbox" id="paperMode" checked>
                                                            <label class="form-check-label fw-semibold" for="paperMode">Paper Trading Mode</label>
                                                        </div>
                                                    </div>
                                                    <div class="col-md-6">
                                                        <div class="form-check form-switch">
                                                            <input class="form-check-input" type="checkbox" id="autoExecute">
                                                            <label class="form-check-label fw-semibold" for="autoExecute">Auto-Execute Triggered Trades</label>
                                                        </div>
                                                    </div>
                                                </div>
                                                
                                                <div id="analysisError" class="alert alert-danger" style="display: none;"></div>
                                                
                                                <div class="d-grid gap-2 d-md-flex">
                                                    <button type="submit" class="btn btn-primary" id="processBtn">
                                                        <span id="processSpinner" class="spinner-border spinner-border-sm me-2" style="display: none;"></span>
                                                        <i class="bi bi-cpu me-2"></i>Process Analysis
                                                    </button>
                                                    <button type="button" class="btn btn-outline-secondary" onclick="analysisApp.clearAnalysis()">
                                                        <i class="bi bi-x-circle me-2"></i>Clear
                                                    </button>
                                                </div>
                                            </form>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Results Panel -->
                                <div class="col-md-6">
                                    <div class="card border-0 bg-light h-100">
                                        <div class="card-header bg-transparent border-0 pb-2">
                                            <h6 class="mb-0 fw-semibold text-dark">
                                                <i class="bi bi-graph-up me-2"></i>Parsed Trades & Signals
                                            </h6>
                                        </div>
                                        <div class="card-body pt-2">
                                            <div id="analysisResults">
                                                <div class="text-center text-muted py-5">
                                                    <i class="bi bi-upload" style="font-size: 3rem; opacity: 0.5;"></i>
                                                    <p class="mt-3 mb-0">Paste your trading analysis to see parsed trades and automated signals</p>
                                                    <small class="text-muted">The system will extract trades, map to live data, and create signals</small>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Main Analysis Cards Grid -->
            <div class="row">
                <div class="col-lg-6 mb-4">
                    ${this.renderAnalysisCard('signals', 'Trading Signals', 'bi-graph-up-arrow', this.renderSignalsContent())}
                </div>
                <div class="col-lg-6 mb-4">
                    ${this.renderAnalysisCard('portfolio', 'Portfolio Optimization', 'bi-pie-chart', this.renderPortfolioContent())}
                </div>
                <div class="col-lg-6 mb-4">
                    ${this.renderAnalysisCard('flow', 'Institutional Flow', 'bi-building', this.renderFlowContent())}
                </div>
                <div class="col-lg-6 mb-4">
                    ${this.renderAnalysisCard('sentiment', 'Market Sentiment', 'bi-heart-pulse', this.renderSentimentContent())}
                </div>
                <div class="col-lg-6 mb-4">
                    ${this.renderAnalysisCard('correlations', 'Asset Correlations', 'bi-diagram-3', this.renderCorrelationContent())}
                </div>
                <div class="col-lg-6 mb-4">
                    ${this.renderAnalysisCard('timeframes', 'Multi-Timeframe Analysis', 'bi-clock-history', this.renderTimeframeContent())}
                </div>
            </div>
        `;
    }

    renderMetricCard(id, title, icon, value, colorClass) {
        return `
            <div class="card h-100 metric-card" style="border: none; border-radius: 0.75rem; box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); transition: all 0.3s ease;">
                <div class="card-body text-center p-3">
                    <div class="d-flex align-items-center justify-content-center mb-2">
                        <div class="rounded-circle p-3 ${colorClass} text-white">
                            <i class="bi ${icon}" style="font-size: 1.5rem;"></i>
                        </div>
                    </div>
                    <h6 class="card-title text-muted mb-1" style="font-size: 0.875rem;">${title}</h6>
                    <div class="h5 mb-0 fw-bold" style="color: #495057;">${value}</div>
                </div>
            </div>
        `;
    }

    renderAnalysisCard(id, title, icon, content) {
        const isExpanded = this.state.expandedCard === id;
        return `
            <div class="card h-100 analysis-card" style="border: none; border-radius: 0.75rem; box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); transition: all 0.3s ease; cursor: pointer;"
                 onclick="analysisApp.toggleCard('${id}')" 
                 onmouseenter="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 0.5rem 1rem rgba(0, 0, 0, 0.15)';" 
                 onmouseleave="this.style.transform='translateY(0)'; this.style.boxShadow='0 0.125rem 0.25rem rgba(0, 0, 0, 0.075)';">
                <div class="card-header bg-transparent border-0 pb-0">
                    <div class="d-flex align-items-center justify-content-between">
                        <div class="d-flex align-items-center">
                            <i class="bi ${icon} me-2" style="color: #667eea; font-size: 1.25rem;"></i>
                            <h6 class="mb-0 fw-bold" style="color: #495057;">${title}</h6>
                        </div>
                        <i class="bi ${isExpanded ? 'bi-chevron-up' : 'bi-chevron-down'}" style="color: #6c757d;"></i>
                    </div>
                </div>
                <div class="card-body pt-2">
                    ${isExpanded ? content : this.renderCardPreview(id)}
                </div>
            </div>
        `;
    }

    renderCardPreview(cardId) {
        switch(cardId) {
            case 'signals':
                return `<p class="text-muted small mb-0">Click to view ${(this.state.signals.signals || []).length} active trading signals and analysis</p>`;
            case 'portfolio':
                return `<p class="text-muted small mb-0">Click to view optimal portfolio allocation recommendations</p>`;
            case 'flow':
                return `<p class="text-muted small mb-0">Click to view institutional money flow analysis</p>`;
            case 'sentiment':
                return `<p class="text-muted small mb-0">Click to view market sentiment heatmap and trends</p>`;
            case 'correlations':
                return `<p class="text-muted small mb-0">Click to view asset correlation matrix and relationships</p>`;
            case 'timeframes':
                return `<p class="text-muted small mb-0">Click to view multi-timeframe trend analysis</p>`;
            default:
                return `<p class="text-muted small mb-0">Click to expand for detailed analysis</p>`;
        }
    }

    getRegimeDisplay() {
        return this.state.regimes.regime || 'Neutral';
    }

    getFlowDirection() {
        const flows = this.state.flow.flow || [];
        if (flows.length === 0) return 'Neutral';
        const inflowCount = flows.filter(f => f.direction === 'inflow').length;
        return inflowCount > flows.length / 2 ? 'Inflow' : 'Outflow';
    }

    getSentimentOverall() {
        return 'Mixed'; // Placeholder - would calculate from sentiment data
    }

    renderSignalsContent() {
        const signals = this.state.signals.signals || [];
        if (signals.length === 0) {
            return `
                <div class="text-center p-4">
                    <i class="bi bi-info-circle text-muted" style="font-size: 2rem;"></i>
                    <p class="text-muted mt-2 mb-0">No trading signals available</p>
                </div>
            `;
        }
        
        return `
            <div class="table-responsive">
                <table class="table table-sm table-hover">
                    <thead>
                        <tr>
                            <th>Asset</th>
                            <th>Type</th>
                            <th>Confidence</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${signals.slice(0, 5).map(signal => `
                            <tr>
                                <td><strong>${signal.asset}</strong></td>
                                <td><span class="badge ${this.getSignalBadgeClass(signal.type)}">${signal.type}</span></td>
                                <td>${signal.confidence.toFixed(1)}%</td>
                                <td class="small text-muted">${new Date(signal.timestamp).toLocaleTimeString()}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    renderPortfolioContent() {
        const allocation = this.state.portfolio.allocation || [];
        if (allocation.length === 0) {
            return `
                <div class="text-center p-4">
                    <i class="bi bi-info-circle text-muted" style="font-size: 2rem;"></i>
                    <p class="text-muted mt-2 mb-0">No portfolio data available</p>
                </div>
            `;
        }
        
        return `
            <div class="row">
                <div class="col-md-6">
                    <canvas id="portfolioChart" height="200"></canvas>
                </div>
                <div class="col-md-6">
                    <div class="table-responsive">
                        <table class="table table-sm">
                            <thead>
                                <tr><th>Asset</th><th>Allocation</th></tr>
                            </thead>
                            <tbody>
                                ${allocation.slice(0, 5).map(alloc => `
                                    <tr>
                                        <td><strong>${alloc.asset}</strong></td>
                                        <td>${alloc.percentage.toFixed(1)}%</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    renderFlowContent() {
        const flows = this.state.flow.flow || [];
        if (flows.length === 0) {
            return `
                <div class="text-center p-4">
                    <i class="bi bi-info-circle text-muted" style="font-size: 2rem;"></i>
                    <p class="text-muted mt-2 mb-0">No flow data available</p>
                </div>
            `;
        }
        
        return `
            <div class="table-responsive">
                <table class="table table-sm table-hover">
                    <thead>
                        <tr><th>Asset</th><th>Volume</th><th>Direction</th></tr>
                    </thead>
                    <tbody>
                        ${flows.slice(0, 5).map(flow => `
                            <tr>
                                <td><strong>${flow.asset}</strong></td>
                                <td>${flow.volume.toLocaleString()}</td>
                                <td><span class="badge ${flow.direction === 'inflow' ? 'bg-success' : 'bg-danger'}">${flow.direction}</span></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    renderSentimentContent() {
        return `
            <div class="text-center p-4">
                <div class="row">
                    <div class="col-4 text-center">
                        <div class="h4 text-success mb-1">65%</div>
                        <small class="text-muted">Positive</small>
                    </div>
                    <div class="col-4 text-center">
                        <div class="h4 text-warning mb-1">20%</div>
                        <small class="text-muted">Neutral</small>
                    </div>
                    <div class="col-4 text-center">
                        <div class="h4 text-danger mb-1">15%</div>
                        <small class="text-muted">Negative</small>
                    </div>
                </div>
            </div>
        `;
    }

    renderCorrelationContent() {
        return `
            <div class="text-center p-4">
                <div class="correlation-grid">
                    <div class="row small">
                        <div class="col">BTC/ETH: <strong class="text-success">0.85</strong></div>
                        <div class="col">ETH/SOL: <strong class="text-warning">0.72</strong></div>
                    </div>
                    <div class="row small mt-2">
                        <div class="col">BTC/SPY: <strong class="text-danger">-0.12</strong></div>
                        <div class="col">ETH/DXY: <strong class="text-danger">-0.34</strong></div>
                    </div>
                </div>
            </div>
        `;
    }

    renderTimeframeContent() {
        const timeframes = this.state.timeframes.timeframes || [];
        if (timeframes.length === 0) {
            return `
                <div class="text-center p-4">
                    <i class="bi bi-info-circle text-muted" style="font-size: 2rem;"></i>
                    <p class="text-muted mt-2 mb-0">No timeframe data available</p>
                </div>
            `;
        }
        
        return `
            <div class="table-responsive">
                <table class="table table-sm">
                    <thead>
                        <tr><th>Timeframe</th><th>Trend</th><th>Strength</th></tr>
                    </thead>
                    <tbody>
                        ${timeframes.slice(0, 4).map(tf => `
                            <tr>
                                <td><strong>${tf.timeframe}</strong></td>
                                <td><span class="badge ${this.getTrendBadgeClass(tf.trend)}">${tf.trend}</span></td>
                                <td>${tf.strength}%</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    renderSignalsSection() {
        const isExpanded = this.state.expandedSection === 'signals';
        const signals = this.state.signals.signals || [];
        
        return `
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button ${isExpanded ? '' : 'collapsed'}" type="button" 
                            onclick="analysisApp.toggleSection('signals')">
                        <i class="fas fa-chart-line me-2"></i>
                        Trading Signals Summary (${signals.length})
                    </button>
                </h2>
                <div class="accordion-collapse collapse ${isExpanded ? 'show' : ''}">
                    <div class="accordion-body">
                        <div class="row">
                            <div class="col-md-6">
                                <table class="table table-striped table-hover table-sm">
                                    <thead>
                                        <tr>
                                            <th>Asset</th>
                                            <th>Type</th>
                                            <th>Confidence</th>
                                            <th>Time</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${signals.slice(0, 5).map(signal => `
                                            <tr>
                                                <td><strong>${signal.asset}</strong></td>
                                                <td>
                                                    <span class="badge ${this.getSignalBadgeClass(signal.type)}">
                                                        ${signal.type}
                                                    </span>
                                                </td>
                                                <td>${signal.confidence.toFixed(1)}%</td>
                                                <td>${new Date(signal.timestamp).toLocaleTimeString()}</td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <div style="height: 250px;">
                                    <canvas id="signalsChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderSentimentSection() {
        const isExpanded = this.state.expandedSection === 'sentiment';
        
        return `
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button ${isExpanded ? '' : 'collapsed'}" type="button" 
                            onclick="analysisApp.toggleSection('sentiment')">
                        <i class="fas fa-heart me-2"></i>
                        Market Sentiment Heatmap
                    </button>
                </h2>
                <div class="accordion-collapse collapse ${isExpanded ? 'show' : ''}">
                    <div class="accordion-body">
                        <div class="text-center">
                            <div id="sentimentHeatmap">
                                ${this.renderSentimentHeatmap()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderFlowSection() {
        const isExpanded = this.state.expandedSection === 'flow';
        const flows = this.state.flow.flow || [];
        
        return `
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button ${isExpanded ? '' : 'collapsed'}" type="button" 
                            onclick="analysisApp.toggleSection('flow')">
                        <i class="fas fa-building me-2"></i>
                        Institutional Flow Analysis
                    </button>
                </h2>
                <div class="accordion-collapse collapse ${isExpanded ? 'show' : ''}">
                    <div class="accordion-body">
                        <div class="row">
                            <div class="col-md-6">
                                <table class="table table-striped table-hover table-sm">
                                    <thead>
                                        <tr>
                                            <th>Asset</th>
                                            <th>Volume</th>
                                            <th>Direction</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${flows.slice(0, 5).map(flow => `
                                            <tr>
                                                <td><strong>${flow.asset}</strong></td>
                                                <td>${flow.volume.toLocaleString()}</td>
                                                <td>
                                                    <span class="badge ${flow.direction === 'inflow' ? 'bg-success' : 'bg-danger'}">
                                                        ${flow.direction}
                                                    </span>
                                                </td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <div style="height: 250px;">
                                    <canvas id="flowChart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderPortfolioSection() {
        const isExpanded = this.state.expandedSection === 'portfolio';
        const allocation = this.state.portfolio.allocation || [];
        
        return `
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button ${isExpanded ? '' : 'collapsed'}" type="button" 
                            onclick="analysisApp.toggleSection('portfolio')">
                        <i class="fas fa-pie-chart me-2"></i>
                        Portfolio Optimization
                    </button>
                </h2>
                <div class="accordion-collapse collapse ${isExpanded ? 'show' : ''}">
                    <div class="accordion-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div style="height: 300px;">
                                    <canvas id="portfolioChart"></canvas>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <table class="table table-striped table-hover">
                                    <thead>
                                        <tr>
                                            <th>Asset</th>
                                            <th>Allocation</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${allocation.map(alloc => `
                                            <tr>
                                                <td><strong>${alloc.asset}</strong></td>
                                                <td>${alloc.percentage.toFixed(1)}%</td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderRegimeSection() {
        const isExpanded = this.state.expandedSection === 'regime';
        
        return `
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button ${isExpanded ? '' : 'collapsed'}" type="button" 
                            onclick="analysisApp.toggleSection('regime')">
                        <i class="fas fa-thermometer-half me-2"></i>
                        Market Regime Detection
                    </button>
                </h2>
                <div class="accordion-collapse collapse ${isExpanded ? 'show' : ''}">
                    <div class="accordion-body">
                        <div class="alert alert-info text-center">
                            <h5 class="mb-1">Current Market Regime: <strong>${this.state.regimes.regime}</strong></h5>
                            <p class="mb-0">Confidence: <strong>${this.state.regimes.confidence}%</strong></p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderCorrelationSection() {
        const isExpanded = this.state.expandedSection === 'correlation';
        
        return `
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button ${isExpanded ? '' : 'collapsed'}" type="button" 
                            onclick="analysisApp.toggleSection('correlation')">
                        <i class="fas fa-project-diagram me-2"></i>
                        Asset Correlation Matrix
                    </button>
                </h2>
                <div class="accordion-collapse collapse ${isExpanded ? 'show' : ''}">
                    <div class="accordion-body">
                        <div class="text-center">
                            <div id="correlationHeatmap">
                                ${this.renderCorrelationHeatmap()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    renderTimeframeSection() {
        const isExpanded = this.state.expandedSection === 'timeframe';
        const timeframes = this.state.timeframes.timeframes || [];
        
        return `
            <div class="accordion-item">
                <h2 class="accordion-header">
                    <button class="accordion-button ${isExpanded ? '' : 'collapsed'}" type="button" 
                            onclick="analysisApp.toggleSection('timeframe')">
                        <i class="fas fa-clock me-2"></i>
                        Multi-Timeframe Analysis
                    </button>
                </h2>
                <div class="accordion-collapse collapse ${isExpanded ? 'show' : ''}">
                    <div class="accordion-body">
                        <table class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>Timeframe</th>
                                    <th>Trend</th>
                                    <th>Strength</th>
                                    <th>Signals</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${timeframes.map(tf => `
                                    <tr>
                                        <td><strong>${tf.timeframe}</strong></td>
                                        <td>
                                            <span class="badge ${this.getTrendBadgeClass(tf.trend)}">
                                                ${tf.trend}
                                            </span>
                                        </td>
                                        <td>${tf.strength}%</td>
                                        <td>${tf.signals}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    renderSentimentHeatmap() {
        const sentiment = this.state.sentiment.sentiment || [];
        const assets = ['BTC', 'ETH', 'ADA', 'SOL', 'LINK'];
        const sentimentTypes = ['Positive', 'Negative', 'Neutral'];
        
        let html = '<div class="sentiment-heatmap">';
        html += '<div class="heatmap-grid">';
        
        // Header row
        html += '<div class="heatmap-row">';
        html += '<div class="heatmap-cell header"></div>';
        assets.forEach(asset => {
            html += `<div class="heatmap-cell header">${asset}</div>`;
        });
        html += '</div>';
        
        // Data rows
        sentimentTypes.forEach(type => {
            html += '<div class="heatmap-row">';
            html += `<div class="heatmap-cell header">${type}</div>`;
            assets.forEach(asset => {
                const dataPoint = sentiment.find(s => s.x === asset && s.y === type);
                const value = dataPoint ? dataPoint.value : 50;
                const opacity = value / 100;
                html += `<div class="heatmap-cell" style="background-color: rgba(13, 110, 253, ${opacity}); color: white;">${value}%</div>`;
            });
            html += '</div>';
        });
        
        html += '</div></div>';
        return html;
    }

    renderCorrelationHeatmap() {
        const correlations = this.state.correlations.correlations || [];
        const assets = ['BTC', 'ETH', 'ADA', 'SOL', 'LINK'];
        
        let html = '<div class="correlation-heatmap">';
        html += '<div class="heatmap-grid">';
        
        // Header row
        html += '<div class="heatmap-row">';
        html += '<div class="heatmap-cell header"></div>';
        assets.forEach(asset => {
            html += `<div class="heatmap-cell header">${asset}</div>`;
        });
        html += '</div>';
        
        // Data rows
        assets.forEach(asset1 => {
            html += '<div class="heatmap-row">';
            html += `<div class="heatmap-cell header">${asset1}</div>`;
            assets.forEach(asset2 => {
                const dataPoint = correlations.find(c => c.x === asset1 && c.y === asset2);
                const value = dataPoint ? dataPoint.value : (asset1 === asset2 ? 1 : 0);
                const color = value > 0 ? 'rgba(25, 135, 84, ' + Math.abs(value) + ')' : 'rgba(220, 53, 69, ' + Math.abs(value) + ')';
                html += `<div class="heatmap-cell" style="background-color: ${color}; color: white;">${value.toFixed(2)}</div>`;
            });
            html += '</div>';
        });
        
        html += '</div></div>';
        return html;
    }

    getSignalBadgeClass(type) {
        switch(type) {
            case 'BUY': return 'bg-success';
            case 'SELL': return 'bg-danger';
            default: return 'bg-secondary';
        }
    }

    getTrendBadgeClass(trend) {
        switch(trend) {
            case 'bullish': return 'bg-success';
            case 'bearish': return 'bg-danger';
            default: return 'bg-secondary';
        }
    }

    attachEventHandlers() {
        // Add CSS for card animations if not already present
        if (!document.getElementById('analysis-custom-styles')) {
            const style = document.createElement('style');
            style.id = 'analysis-custom-styles';
            style.textContent = `
                .metric-card:hover {
                    transform: translateY(-3px);
                    box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15) !important;
                }
                .analysis-card {
                    border: 1px solid #e9ecef;
                }
                .analysis-card:hover {
                    border-color: #667eea !important;
                }
                .correlation-grid .row {
                    margin: 0;
                }
                .correlation-grid .col {
                    padding: 0.25rem;
                    border-radius: 0.25rem;
                    background: #f8f9fa;
                    margin: 0.125rem;
                }
            `;
            document.head.appendChild(style);
        }
    }

    initializeCharts() {
        // Initialize Chart.js charts for expanded cards
        setTimeout(() => {
            if (this.state.expandedCard === 'signals') {
                this.initSignalsChart();
            }
            if (this.state.expandedCard === 'flow') {
                this.initFlowChart();
            }
            if (this.state.expandedCard === 'portfolio') {
                this.initPortfolioChart();
            }
        }, 100);
    }

    initSignalsChart() {
        const canvas = document.getElementById('signalsChart');
        if (!canvas) return;

        if (this.charts.signals) {
            this.charts.signals.destroy();
        }

        const ctx = canvas.getContext('2d');
        this.charts.signals = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.state.signals.labels || [],
                datasets: [{
                    label: 'Signal Confidence',
                    data: this.state.signals.data || [],
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    initFlowChart() {
        const canvas = document.getElementById('flowChart');
        if (!canvas) return;

        if (this.charts.flow) {
            this.charts.flow.destroy();
        }

        const ctx = canvas.getContext('2d');
        this.charts.flow = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.state.flow.labels || [],
                datasets: [{
                    label: 'Institutional Flow Volume',
                    data: this.state.flow.data || [],
                    borderColor: '#198754',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initPortfolioChart() {
        const canvas = document.getElementById('portfolioChart');
        if (!canvas) return;

        if (this.charts.portfolio) {
            this.charts.portfolio.destroy();
        }

        const allocation = this.state.portfolio.allocation || [];
        const ctx = canvas.getContext('2d');
        this.charts.portfolio = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: allocation.map(a => a.asset),
                datasets: [{
                    data: allocation.map(a => a.percentage),
                    backgroundColor: ['#0d6efd', '#198754', '#ffc107', '#dc3545', '#6f42c1']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
    }
}

// Initialize the app when the page loads
let analysisApp;
document.addEventListener('DOMContentLoaded', () => {
    analysisApp = new AnalysisApp();
});

// Add some custom CSS for heatmaps
const style = document.createElement('style');
style.textContent = `
    .heatmap-grid {
        display: inline-block;
        border: 1px solid #dee2e6;
    }
    .heatmap-row {
        display: flex;
    }
    .heatmap-cell {
        width: 60px;
        height: 40px;
        border: 1px solid #dee2e6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: bold;
    }
    .heatmap-cell.header {
        background-color: #f8f9fa;
        font-weight: bold;
        color: #333;
    }
`;
document.head.appendChild(style);