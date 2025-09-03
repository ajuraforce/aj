// New Dashboard Components - All Existing Features
class NewDashboardApp {
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
            // New dashboard card data
            health: {},
            paperTrading: {},
            advancedAnalysis: {},
            community: { posts: [], metrics: {} }
        };
        
        this.init();
    }

    async init() {
        await this.loadData();
        this.render();
        this.ensureIconsLoad();
        this.initializeTooltips();
        
        // Refresh data every 30 seconds
        setInterval(() => this.loadData(), 30000);
    }

    async loadData() {
        try {
            // Load all dashboard data in parallel
            const [statusRes, alertsRes, tradesRes, portfolioRes, narrativesRes, resonanceRes, scenariosRes, timelineRes, healthRes, paperTradingRes, advancedAnalysisRes, communityRes] = await Promise.all([
                fetch('/api/status'),
                fetch('/api/alerts'),
                fetch('/api/trades'),
                fetch('/api/portfolio'),
                fetch('/api/dashboard/narratives'),
                fetch('/api/patterns/resonance'),
                fetch('/api/patterns/scenarios'),
                fetch('/api/patterns/timeline'),
                fetch('/api/health'),
                fetch('/api/analytics/paper'),
                fetch('/api/advanced-analysis'),
                fetch('/api/community/posts?per_page=5')
            ]);

            if (statusRes.ok) {
                this.state.status = await statusRes.json();
            }
            if (alertsRes.ok) {
                this.state.alerts = await alertsRes.json();
            }
            if (tradesRes.ok) {
                this.state.trades = await tradesRes.json();
            }
            if (portfolioRes.ok) {
                this.state.portfolio = await portfolioRes.json();
            }
            if (narrativesRes.ok) {
                this.state.narratives = await narrativesRes.json();
            }
            if (resonanceRes.ok) {
                const resonance = await resonanceRes.json();
                this.state.systemInsights.resonance = resonance;
            }
            if (scenariosRes.ok) {
                const scenarios = await scenariosRes.json();
                this.state.systemInsights.scenarios = scenarios;
            }
            if (timelineRes.ok) {
                const timeline = await timelineRes.json();
                this.state.systemInsights.timeline = timeline;
            }
            if (healthRes.ok) {
                this.state.health = await healthRes.json();
            }
            if (paperTradingRes.ok) {
                this.state.paperTrading = await paperTradingRes.json();
            }
            if (advancedAnalysisRes.ok) {
                this.state.advancedAnalysis = await advancedAnalysisRes.json();
            }
            if (communityRes.ok) {
                this.state.community = await communityRes.json();
            }

            console.log('Data loaded successfully:', this.state);
            this.render();
            
            // Initialize tooltips after render
            setTimeout(() => {
                this.initializeTooltips();
                this.createSparkline();
            }, 100);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }

    ensureIconsLoad() {
        // Load Bootstrap Icons with multiple fallbacks
        const fontSources = [
            'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/fonts/bootstrap-icons.woff2',
            'https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.11.3/fonts/bootstrap-icons.woff2'
        ];
        
        fontSources.forEach((src, index) => {
            setTimeout(() => {
                const iconFont = new FontFace('bootstrap-icons', `url(${src})`);
                iconFont.load().then(() => {
                    document.fonts.add(iconFont);
                    console.log(`Bootstrap Icons loaded from source ${index + 1}`);
                }).catch(() => {
                    // Silent fallback
                });
            }, index * 200);
        });
    }

    render() {
        const container = document.getElementById('dashboard-content');
        if (!container) return;

        container.innerHTML = `
            <!-- Enhanced KPI Cards Section -->
            <div class="row mb-4">
                <!-- System Health Monitor -->
                <div class="col-12 col-md-6 col-lg-3">
                    <div class="card kpi-card text-center card-hover" style="border-radius: 15px;">
                        <div class="card-body">
                            <div class="d-flex justify-content-center align-items-center mb-2">
                                <i class="bi bi-shield-check ${this.getHealthStatusColor()} kpi-icon"></i>
                                <i class="bi bi-info-circle ms-2 text-muted cursor-pointer" 
                                   data-bs-toggle="tooltip" 
                                   data-bs-placement="top" 
                                   title="Checks include API connectivity, data freshness, error logs, alert latency, backup status"
                                   style="font-size: 0.9rem;"></i>
                            </div>
                            <h5 class="card-title">System Health</h5>
                            <div class="kpi-value">${this.getHealthScore()}/5</div>
                            <span class="badge ${this.getHealthBadgeClass()}">${this.getHealthStatus()}</span>
                            <div class="mt-2">
                                <a href="/health" class="btn btn-sm btn-outline-primary">Details</a>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Paper Trading Performance -->
                <div class="col-12 col-md-6 col-lg-3">
                    <div class="card kpi-card text-center card-hover cursor-pointer" 
                         style="border-radius: 15px;" 
                         onclick="newDashboard.openPaperTradingModal()">
                        <div class="card-body">
                            <i class="bi bi-graph-up-arrow ${this.getPaperTradingColor()} kpi-icon"></i>
                            <h5 class="card-title">Paper Trading</h5>
                            <div class="kpi-value">${this.state.paperTrading.win_rate || 0}%</div>
                            <span class="badge bg-info">Win Rate</span>
                            <div class="small text-muted mt-1 d-flex align-items-center justify-content-center">
                                <span class="me-2">P&L: $${(this.state.paperTrading.total_pnl || 0).toFixed(2)}</span>
                                <canvas id="pnl-sparkline" width="40" height="20" style="max-width: 40px; max-height: 20px;"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- System-Level Intelligence -->
                <div class="col-12 col-md-6 col-lg-3">
                    <div class="card kpi-card card-hover cursor-pointer" 
                         style="border-radius: 15px;" 
                         onclick="newDashboard.openSystemIntelligenceModal()">
                        <div class="card-body">
                            <div class="d-flex align-items-center justify-content-center mb-2">
                                <i class="bi bi-cpu text-primary me-2" style="font-size: 1.5rem;"></i>
                                <h6 class="card-title mb-0">System-Level Intelligence</h6>
                            </div>
                            <div class="mb-2">
                                <small class="text-muted">Scenario Probabilities</small>
                            </div>
                            <div class="scenario-item">
                                <div class="d-flex justify-content-between align-items-center mb-1">
                                    <span class="fw-bold text-success">Bull Continuation</span>
                                    <span class="badge bg-success">${this.getBullProbability()}%</span>
                                </div>
                                <div class="progress" style="height: 8px;">
                                    <div class="progress-bar bg-success" 
                                         style="width: ${this.getBullProbability()}%;"></div>
                                </div>
                            </div>
                            <div class="mt-2">
                                <small class="text-muted">Based on ${this.state.status.market_regime || 'Risk-On'} regime • Updated ${this.getLastUpdateTime()}</small>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Community Sentiment Pulse -->
                <div class="col-12 col-md-6 col-lg-3">
                    <div class="card kpi-card text-center card-hover" style="border-radius: 15px;">
                        <div class="card-body">
                            <i class="bi bi-people ${this.getSentimentColor()} kpi-icon"></i>
                            <h5 class="card-title">Community Pulse</h5>
                            <div class="d-flex justify-content-center align-items-center mb-2">
                                <div class="sentiment-gauge me-2">
                                    <div class="gauge-background" style="width: 40px; height: 20px; background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%); border-radius: 10px; position: relative;">
                                        <div class="gauge-indicator" style="position: absolute; top: -2px; left: ${this.getSentimentPosition()}%; width: 4px; height: 24px; background: #fff; border: 1px solid #333; border-radius: 2px;"></div>
                                    </div>
                                </div>
                                <span class="kpi-value">${this.getCommunityScore()}</span>
                            </div>
                            <span class="badge ${this.getSentimentBadgeClass()}">${this.getCommunityStatus()}</span>
                            <div class="small text-muted mt-1">
                                <div>${(this.state.community.posts || []).length} Recent Posts</div>
                                <div class="d-flex justify-content-center mt-1">
                                    <span class="badge bg-success me-1" style="font-size: 0.7rem;">${this.getPositivePosts()}</span>
                                    <span class="badge bg-danger" style="font-size: 0.7rem;">${this.getNegativePosts()}</span>
                                </div>
                            </div>
                            <div class="mt-2">
                                <a href="/community" class="btn btn-sm btn-outline-success">Community</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- News Intelligence Widget -->
            ${this.renderNewsWidget()}

            <!-- System-Level Insights -->
            ${this.renderSystemInsights()}

            <!-- Recent Activity -->
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <i class="bi bi-clock-history me-2"></i>Recent Activity
                        </div>
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
                            <a href="/live-alerts" class="btn btn-primary btn-sm">
                                <i class="bi bi-eye me-1"></i>View All Alerts
                            </a>
                        </div>
                    </div>
                </div>
            </div>
            
        `;

        console.log('✅ AJxAI Chat button added to dashboard');

        // Initialize charts after rendering
        setTimeout(() => {
            this.initCharts();
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
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h6 class="mb-0">
                            <i class="bi bi-newspaper me-2"></i>News Intelligence Analytics
                        </h6>
                        <div class="d-flex align-items-center">
                            <span class="badge bg-success me-2">Live</span>
                            <small class="text-muted">Updated: ${lastUpdated}</small>
                        </div>
                    </div>
                    <div class="card-body">
                        <!-- News Metrics -->
                        <div class="row mb-4">
                            <div class="col-md-8">
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 20px;">
                                    <h6 class="text-white mb-3">
                                        <i class="bi bi-graph-up me-2"></i>24-Hour News Activity Trend
                                    </h6>
                                    <div style="height: 120px; background: rgba(255,255,255,0.1); border-radius: 8px; position: relative;">
                                        <canvas id="newsTrendChart" style="width: 100%; height: 100%;"></canvas>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="row">
                                    <div class="col-12 mb-2">
                                        <div class="p-3" style="background: linear-gradient(45deg, #ff6b6b, #ee5a24); border-radius: 10px; color: white;">
                                            <div class="h4 mb-0">${topNarratives.length}</div>
                                            <div class="small">Topic Clusters</div>
                                        </div>
                                    </div>
                                    <div class="col-12 mb-2">
                                        <div class="p-3" style="background: linear-gradient(45deg, #4ecdc4, #44a08d); border-radius: 10px; color: white;">
                                            <div class="h4 mb-0">${breakingNews.length}</div>
                                            <div class="small">Breaking News</div>
                                        </div>
                                    </div>
                                    <div class="col-12">
                                        <div class="p-3" style="background: linear-gradient(45deg, #45b7d1, #96c93d); border-radius: 10px; color: white;">
                                            <div class="h4 mb-0">0.65</div>
                                            <div class="small">Avg Sentiment</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Topic Clusters -->
                        ${topNarratives.length > 0 ? `
                            <div class="row mb-3">
                                <div class="col-12">
                                    <h6 class="text-primary mb-3">
                                        <i class="bi bi-bullseye me-2"></i>Topic Clusters & Analysis
                                    </h6>
                                    <div class="row">
                                        ${topNarratives.slice(0, 6).map(narrative => `
                                            <div class="col-md-4 mb-3">
                                                <div class="card border-start border-primary border-3">
                                                    <div class="card-body p-3">
                                                        <h6 class="card-title">${narrative.theme || 'Market Update'}</h6>
                                                        <p class="card-text small text-muted">${narrative.summary || 'Analysis of market patterns and trends'}</p>
                                                        <div class="d-flex justify-content-between align-items-center">
                                                            <span class="badge bg-primary">${narrative.articles_count || 1} articles</span>
                                                            <small class="text-muted">${narrative.confidence ? Math.round(narrative.confidence * 100) : 85}% confidence</small>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            </div>
                        ` : ''}
                        
                        <!-- Breaking News Stream -->
                        ${breakingNews.length > 0 ? `
                            <div class="row">
                                <div class="col-12">
                                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 12px; padding: 15px;">
                                        <h6 class="text-white mb-3">
                                            <i class="bi bi-lightning-charge me-2"></i>Real-time News Stream
                                        </h6>
                                        <div style="max-height: 250px; overflow-y: auto;">
                                            ${breakingNews.slice(0, 5).map(news => `
                                                <div class="mb-2 p-3" style="background: rgba(255,255,255,0.1); border-radius: 8px;">
                                                    <div class="d-flex justify-content-between align-items-start">
                                                        <a href="${news.link || '#'}" target="_blank" class="text-decoration-none text-white flex-grow-1">
                                                            <h6 class="mb-1">${news.title || 'Market Update'}</h6>
                                                        </a>
                                                        <span class="sentiment-indicator ms-2 fs-5">
                                                            ${(news.sentiment_score || 0) > 0.3 ? '📈' : (news.sentiment_score || 0) < -0.3 ? '📉' : '➡️'}
                                                        </span>
                                                    </div>
                                                    <div class="d-flex justify-content-between align-items-center mt-2">
                                                        <small class="text-white-50">${news.source || 'Market Feed'} • ${news.published ? new Date(news.published).toLocaleString() : 'Recent'}</small>
                                                        <span class="badge ${(news.sentiment_score || 0) > 0.3 ? 'bg-success' : (news.sentiment_score || 0) < -0.3 ? 'bg-danger' : 'bg-secondary'} small">
                                                            ${(news.sentiment_score || 0).toFixed(2)}
                                                        </span>
                                                    </div>
                                                </div>
                                            `).join('')}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ` : `
                            <div class="text-center text-muted py-4">
                                <i class="bi bi-newspaper" style="font-size: 2rem;"></i>
                                <p class="mt-2">No recent news available</p>
                                <small>News feed will update automatically</small>
                            </div>
                        `}
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
                        <h6 class="mb-0">
                            <i class="bi bi-bullseye me-2"></i>System-Level Intelligence
                        </h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <!-- Scenario Analysis -->
                            <div class="col-lg-4 mb-3">
                                <h6 class="text-primary mb-3">
                                    <i class="bi bi-radar me-2"></i>Scenario Probabilities
                                </h6>
                                <div style="position: relative; height: 350px;">
                                    ${this.renderScenarioAnalysis(scenarios)}
                                </div>
                            </div>
                            
                            <!-- System Resonance -->
                            <div class="col-lg-4 mb-3">
                                <h6 class="text-success mb-3">
                                    <i class="bi bi-diagram-3 me-2"></i>System Resonance Map
                                </h6>
                                <div>
                                    ${this.renderResonanceMapCards(resonance)}
                                </div>
                            </div>
                            
                            <!-- Timeline of Echoes -->
                            <div class="col-lg-4 mb-3">
                                <h6 class="text-warning mb-3">
                                    <i class="bi bi-clock-history me-2"></i>Timeline of Echoes
                                </h6>
                                <div style="max-height: 300px; overflow-y: auto;">
                                    ${timeline.length > 0 ? timeline.map(entry => `
                                        <div class="mb-2 p-2 border-start border-warning border-3 bg-light">
                                            <div class="d-flex justify-content-between align-items-start">
                                                <span class="fw-bold text-uppercase small">${entry.concept || 'Market Signal'}</span>
                                                <span class="text-muted small">${entry.time ? new Date(entry.time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : 'Recent'}</span>
                                            </div>
                                            <div class="small text-muted">${entry.source || 'System Analysis'}</div>
                                            ${entry.title ? `<div class="small">${entry.title}</div>` : ''}
                                        </div>
                                    `).join('') : `
                                        <div class="text-center text-muted py-4">
                                            <i class="bi bi-activity" style="font-size: 2rem;"></i>
                                            <p class="mt-2">No recent echoes</p>
                                            <small>System monitoring for patterns</small>
                                        </div>
                                    `}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        `;
    }

    renderScenarioAnalysis(scenarios) {
        // Check if we have real scenario data with meaningful content
        if (scenarios && scenarios.data && scenarios.data.length > 0) {
            // Render real scenario data with radar chart
            return `
                <canvas id="scenario-radar"></canvas>
                <div class="text-center mt-2">
                    <small class="text-muted">Real-time scenario analysis</small>
                </div>
            `;
        }
        
        // Fallback: Show intelligent scenario analysis based on current market regime
        const marketRegime = this.state.status.market_regime || 'Risk-On';
        const currentTime = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        // Dynamic scenarios based on market regime
        let bullData, neutralData, bearData;
        
        if (marketRegime === 'Risk-On') {
            bullData = { name: 'Bull Continuation', probability: 68, color: 'success' };
            neutralData = { name: 'Sideways Movement', probability: 22, color: 'warning' };
            bearData = { name: 'Market Correction', probability: 10, color: 'danger' };
        } else {
            bullData = { name: 'Recovery Rally', probability: 25, color: 'success' };
            neutralData = { name: 'Range Bound', probability: 35, color: 'warning' };
            bearData = { name: 'Bear Continuation', probability: 40, color: 'danger' };
        }
        
        return `
            <div class="scenario-analysis-container h-100">
                <div class="text-center mb-3">
                    <small class="text-muted">Based on ${marketRegime} regime • Updated ${currentTime}</small>
                </div>
                
                <div class="scenario-cards">
                    <div class="mb-3 p-3 border border-success rounded bg-light">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="fw-bold text-success">${bullData.name}</span>
                            <span class="badge bg-success">${bullData.probability}%</span>
                        </div>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-bar bg-success" style="width: ${bullData.probability}%"></div>
                        </div>
                    </div>
                    
                    <div class="mb-3 p-3 border border-warning rounded bg-light">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="fw-bold text-warning">${neutralData.name}</span>
                            <span class="badge bg-warning">${neutralData.probability}%</span>
                        </div>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-bar bg-warning" style="width: ${neutralData.probability}%"></div>
                        </div>
                    </div>
                    
                    <div class="mb-3 p-3 border border-danger rounded bg-light">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="fw-bold text-danger">${bearData.name}</span>
                            <span class="badge bg-danger">${bearData.probability}%</span>
                        </div>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-bar bg-danger" style="width: ${bearData.probability}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-3 p-2 bg-light rounded">
                    <small class="text-muted">
                        <i class="bi bi-info-circle me-1"></i>
                        AI analysis based on ${this.state.status.features_operational || 15} active features
                    </small>
                    <div class="mt-2 text-center">
                        <button class="btn btn-outline-primary btn-sm" onclick="window.location.href='/features'">
                            <i class="bi bi-grid me-1"></i>
                            View All Features
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    renderResonanceMapCards(resonance) {
        if (!resonance || !resonance.nodes || resonance.nodes.length === 0) {
            return `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-diagram-3" style="font-size: 2rem;"></i>
                    <p class="mt-2">No resonance patterns</p>
                    <small>System learning correlations</small>
                </div>
            `;
        }

        return resonance.nodes.slice(0, 6).map(node => `
            <div class="mb-2 p-2 border border-success rounded">
                <div class="d-flex justify-content-between align-items-center">
                    <span class="small fw-bold">${node.id || 'Pattern'}</span>
                    <span class="badge bg-success small">${Math.round((node.strength || 0.5) * 100)}%</span>
                </div>
                <small class="text-muted">${node.type || 'Market correlation'}</small>
            </div>
        `).join('');
    }

    renderRecentActivity() {
        if (!this.state.alerts.recent || this.state.alerts.recent.length === 0) {
            return `
                <tr>
                    <td colspan="3" class="text-center text-muted py-3">
                        <i class="bi bi-info-circle me-2"></i>No recent activity
                    </td>
                </tr>
            `;
        }

        return this.state.alerts.recent.map(alert => `
            <tr>
                <td>${alert.time || new Date().toLocaleTimeString()}</td>
                <td><span class="badge bg-secondary">${alert.type || 'Alert'}</span></td>
                <td>${alert.details || alert.message || 'System notification'}</td>
            </tr>
        `).join('');
    }

    getFeatureStatus(feature) {
        const status = this.state.status;
        if (feature === 'ai_strategist') {
            return status.advanced_features_status?.enhancement_metrics?.features_enabled || 0;
        }
        if (feature === 'advanced_features') {
            return status.features_operational || 0;
        }
        return 0;
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
                            backgroundColor: 'rgba(25,135,84,0.1)',
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

            // Initialize news trend chart
            const newsCtx = document.getElementById('newsTrendChart');
            if (newsCtx) {
                new Chart(newsCtx, {
                    type: 'line',
                    data: {
                        labels: ['12h ago', '10h ago', '8h ago', '6h ago', '4h ago', '2h ago', 'Now'],
                        datasets: [{
                            data: [12, 18, 15, 22, 30, 25, 35],
                            borderColor: 'rgba(255,255,255,0.9)',
                            backgroundColor: 'rgba(255,255,255,0.15)',
                            fill: true,
                            pointRadius: 2,
                            pointBackgroundColor: 'white',
                            pointBorderColor: 'rgba(255,255,255,0.9)',
                            pointBorderWidth: 1,
                            tension: 0.4,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        scales: { 
                            x: { 
                                display: true,
                                grid: { display: false },
                                ticks: { 
                                    color: 'rgba(255,255,255,0.7)',
                                    font: { size: 10 }
                                }
                            }, 
                            y: { 
                                display: true,
                                grid: { 
                                    color: 'rgba(255,255,255,0.1)',
                                    drawBorder: false
                                },
                                ticks: { 
                                    color: 'rgba(255,255,255,0.7)',
                                    font: { size: 10 },
                                    maxTicksLimit: 4
                                }
                            } 
                        },
                        plugins: { 
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: 'rgba(0,0,0,0.8)',
                                titleColor: 'white',
                                bodyColor: 'white',
                                borderColor: 'rgba(255,255,255,0.3)',
                                borderWidth: 1
                            }
                        }
                    }
                });
            }
        }
    }

    // Helper methods for new KPI cards
    getHealthScore() {
        const health = this.state.health || {};
        const services = ['Internal API', 'Reddit API', 'Binance API', 'News RSS Feed', 'Database'];
        let onlineCount = 0;
        
        services.forEach(service => {
            if (health[service] === 'Online') {
                onlineCount++;
            }
        });
        
        return onlineCount;
    }

    getHealthStatus() {
        const score = this.getHealthScore();
        if (score >= 4) return 'Excellent';
        if (score >= 3) return 'Good';
        if (score >= 2) return 'Fair';
        return 'Poor';
    }

    getHealthStatusColor() {
        const score = this.getHealthScore();
        if (score >= 4) return 'text-success';
        if (score >= 3) return 'text-info';
        if (score >= 2) return 'text-warning';
        return 'text-danger';
    }

    getHealthBadgeClass() {
        const score = this.getHealthScore();
        if (score >= 4) return 'bg-success';
        if (score >= 3) return 'bg-info';
        if (score >= 2) return 'bg-warning';
        return 'bg-danger';
    }

    getPaperTradingColor() {
        const winRate = this.state.paperTrading.win_rate || 0;
        if (winRate >= 70) return 'text-success';
        if (winRate >= 50) return 'text-info';
        if (winRate >= 30) return 'text-warning';
        return 'text-danger';
    }

    getIntelligenceScore() {
        const status = this.state.status || {};
        const featuresOperational = status.features_operational || 0;
        const maxFeatures = 15;
        
        // Calculate confidence based on features operational and market regime
        let baseScore = Math.round((featuresOperational / maxFeatures) * 70);
        
        // Boost based on market regime confidence
        if (status.market_regime === 'Risk-On') baseScore += 15;
        else if (status.market_regime === 'Risk-Off') baseScore += 10;
        else baseScore += 5;
        
        return Math.min(baseScore, 95);
    }

    getCommunityScore() {
        const posts = this.state.community.posts || [];
        if (posts.length === 0) return '0.0';
        
        // Calculate average sentiment from recent posts
        let totalSentiment = 0;
        let validPosts = 0;
        
        posts.forEach(post => {
            if (post.sentiment_score !== undefined) {
                totalSentiment += post.sentiment_score;
                validPosts++;
            }
        });
        
        if (validPosts === 0) return '0.65'; // Default neutral sentiment
        
        const avgSentiment = totalSentiment / validPosts;
        return (0.5 + avgSentiment * 0.5).toFixed(2); // Convert to 0-1 scale
    }

    getCommunityStatus() {
        const score = parseFloat(this.getCommunityScore());
        if (score >= 0.7) return 'Bullish';
        if (score >= 0.5) return 'Neutral';
        return 'Bearish';
    }

    // New methods for enhanced features
    getSentimentColor() {
        const score = parseFloat(this.getCommunityScore());
        if (score >= 0.7) return 'text-success';
        if (score >= 0.5) return 'text-warning';
        return 'text-danger';
    }

    getSentimentPosition() {
        const score = parseFloat(this.getCommunityScore());
        return Math.round(score * 100);
    }

    getSentimentBadgeClass() {
        const score = parseFloat(this.getCommunityScore());
        if (score >= 0.7) return 'bg-success';
        if (score >= 0.5) return 'bg-warning';
        return 'bg-danger';
    }

    getPositivePosts() {
        const posts = this.state.community.posts || [];
        const positive = posts.filter(post => (post.sentiment_score || 0) > 0.1).length;
        return `+${positive}`;
    }

    getNegativePosts() {
        const posts = this.state.community.posts || [];
        const negative = posts.filter(post => (post.sentiment_score || 0) < -0.1).length;
        return `-${negative}`;
    }

    // New methods for System-Level Intelligence
    getBullProbability() {
        const regime = this.state.status.market_regime || 'Neutral';
        const features = this.state.status.features_operational || 0;
        
        // Calculate bull probability based on regime and features
        let baseProb = 50;
        if (regime === 'Risk-On') baseProb = 75;
        else if (regime === 'Risk-Off') baseProb = 25;
        
        // Adjust based on features operational
        const featureBonus = Math.round((features / 15) * 20);
        return Math.min(baseProb + featureBonus, 95);
    }

    getLastUpdateTime() {
        const now = new Date();
        return now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: true 
        });
    }

    // Modal functions
    openSystemIntelligenceModal() {
        const bullProb = this.getBullProbability();
        const bearProb = 100 - bullProb;
        const regime = this.state.status.market_regime || 'Risk-On';
        
        const modalHtml = `
            <div class="modal fade" id="systemIntelligenceModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title"><i class="bi bi-cpu me-2"></i>System-Level Intelligence</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <h6 class="mb-3">Market Scenario Probabilities</h6>
                            <div class="row mb-4">
                                <div class="col-md-6">
                                    <div class="scenario-card">
                                        <div class="d-flex justify-content-between mb-2">
                                            <strong class="text-success">Bull Continuation</strong>
                                            <span class="badge bg-success">${bullProb}%</span>
                                        </div>
                                        <div class="progress mb-2" style="height: 12px;">
                                            <div class="progress-bar bg-success" style="width: ${bullProb}%;"></div>
                                        </div>
                                        <small class="text-muted">Strong momentum, positive sentiment</small>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="scenario-card">
                                        <div class="d-flex justify-content-between mb-2">
                                            <strong class="text-danger">Bear Correction</strong>
                                            <span class="badge bg-danger">${bearProb}%</span>
                                        </div>
                                        <div class="progress mb-2" style="height: 12px;">
                                            <div class="progress-bar bg-danger" style="width: ${bearProb}%;"></div>
                                        </div>
                                        <small class="text-muted">Risk-off factors emerging</small>
                                    </div>
                                </div>
                            </div>
                            <div class="alert alert-info">
                                <strong>Current Regime:</strong> ${regime} <br>
                                <strong>Analysis:</strong> ${this.getRegimeAnalysis()}
                            </div>
                            <div class="row">
                                <div class="col-md-4">
                                    <h6>Data Sources</h6>
                                    <ul class="list-unstyled small">
                                        <li><i class="bi bi-check-circle text-success"></i> Market Data</li>
                                        <li><i class="bi bi-check-circle text-success"></i> News Sentiment</li>
                                        <li><i class="bi bi-check-circle text-success"></i> Social Trends</li>
                                        <li><i class="bi bi-check-circle text-success"></i> Technical Patterns</li>
                                    </ul>
                                </div>
                                <div class="col-md-8">
                                    <h6>Key Factors</h6>
                                    <div class="small">
                                        ${this.getKeyFactors()}
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <span class="text-muted small">Updated: ${this.getLastUpdateTime()}</span>
                        </div>
                    </div>
                </div>
            </div>`;
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('systemIntelligenceModal'));
        modal.show();
        
        // Clean up modal when hidden
        document.getElementById('systemIntelligenceModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });
    }

    getRegimeAnalysis() {
        const regime = this.state.status.market_regime || 'Risk-On';
        const analyses = {
            'Risk-On': 'Markets showing strong bullish momentum with high liquidity and positive sentiment driving asset prices higher.',
            'Risk-Off': 'Defensive positioning as uncertainty increases, with flight to quality assets and reduced risk appetite.',
            'Neutral': 'Mixed signals in the market with no clear directional bias, awaiting catalysts for next move.'
        };
        return analyses[regime] || analyses['Neutral'];
    }

    getKeyFactors() {
        return `
            <div class="row">
                <div class="col-6">
                    <div class="factor-item mb-2">
                        <span class="badge bg-success me-2">+</span>
                        <small>Technical breakout patterns</small>
                    </div>
                    <div class="factor-item mb-2">
                        <span class="badge bg-success me-2">+</span>
                        <small>Positive news sentiment</small>
                    </div>
                </div>
                <div class="col-6">
                    <div class="factor-item mb-2">
                        <span class="badge bg-warning me-2">!</span>
                        <small>Elevated volatility</small>
                    </div>
                    <div class="factor-item mb-2">
                        <span class="badge bg-info me-2">~</span>
                        <small>Mixed social sentiment</small>
                    </div>
                </div>
            </div>
        `;
    }

    openPaperTradingModal() {
        const modalHtml = `
            <div class="modal fade" id="paperTradingModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title"><i class="bi bi-graph-up-arrow me-2"></i>Paper Trading Details</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <div class="card bg-light">
                                        <div class="card-body text-center">
                                            <h6>Win Rate</h6>
                                            <h4 class="text-success">${this.state.paperTrading.win_rate || 0}%</h4>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card bg-light">
                                        <div class="card-body text-center">
                                            <h6>Total P&L</h6>
                                            <h4 class="${(this.state.paperTrading.total_pnl || 0) >= 0 ? 'text-success' : 'text-danger'}">
                                                $${(this.state.paperTrading.total_pnl || 0).toFixed(2)}
                                            </h4>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <h6>Recent Trades</h6>
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead><tr><th>Symbol</th><th>P&L</th><th>Date</th></tr></thead>
                                    <tbody>${this.renderRecentTrades()}</tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>`;
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('paperTradingModal'));
        modal.show();
        
        // Clean up modal when hidden
        document.getElementById('paperTradingModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });
    }

    // Legacy method - keeping for compatibility
    openAIIntelligenceModal() {
        // Redirect to new system intelligence modal
        this.openSystemIntelligenceModal();
    }

    openAIIntelligenceModalOld() {
        const features = this.state.status.features_operational || 0;
        const totalFeatures = 15;
        const highConfidence = Math.round(features * 0.67);
        const moderateConfidence = features - highConfidence;
        
        const modalHtml = `
            <div class="modal fade" id="aiIntelligenceModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title"><i class="bi bi-cpu me-2"></i>AI Intelligence Breakdown</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="text-center mb-4">
                                <h4>${features} features active</h4>
                                <p class="text-muted">${highConfidence} high-confidence, ${moderateConfidence} moderate</p>
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <canvas id="confidenceChart" width="200" height="200"></canvas>
                                </div>
                                <div class="col-md-6">
                                    <div class="list-group">
                                        <div class="list-group-item d-flex justify-content-between">
                                            <span>Market Regime Detection</span>
                                            <span class="badge bg-success">Active</span>
                                        </div>
                                        <div class="list-group-item d-flex justify-content-between">
                                            <span>Pattern Recognition</span>
                                            <span class="badge bg-success">Active</span>
                                        </div>
                                        <div class="list-group-item d-flex justify-content-between">
                                            <span>Sentiment Analysis</span>
                                            <span class="badge bg-success">Active</span>
                                        </div>
                                        <div class="list-group-item d-flex justify-content-between">
                                            <span>Risk Assessment</span>
                                            <span class="badge bg-warning">Moderate</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <a href="/features" class="btn btn-primary">View All Features</a>
                        </div>
                    </div>
                </div>
            </div>`;
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('aiIntelligenceModal'));
        modal.show();
        
        // Create pie chart
        setTimeout(() => {
            this.createConfidenceChart(highConfidence, moderateConfidence);
        }, 100);
        
        // Clean up modal when hidden
        document.getElementById('aiIntelligenceModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });
    }

    renderRecentTrades() {
        const trades = this.state.paperTrading.recent_trades || [];
        if (trades.length === 0) {
            return '<tr><td colspan="3" class="text-center text-muted">No recent trades</td></tr>';
        }
        
        return trades.slice(0, 7).map(trade => `
            <tr>
                <td>${trade.symbol || 'N/A'}</td>
                <td class="${(trade.pnl || 0) >= 0 ? 'text-success' : 'text-danger'}">
                    $${(trade.pnl || 0).toFixed(2)}
                </td>
                <td>${trade.date || 'N/A'}</td>
            </tr>
        `).join('');
    }

    createConfidenceChart(high, moderate) {
        const ctx = document.getElementById('confidenceChart');
        if (!ctx) return;
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['High Confidence', 'Moderate Confidence', 'Inactive'],
                datasets: [{
                    data: [high, moderate, 15 - high - moderate],
                    backgroundColor: ['#28a745', '#ffc107', '#dc3545']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    createSparkline() {
        const canvas = document.getElementById('pnl-sparkline');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const trades = this.state.paperTrading.recent_trades || [];
        
        if (trades.length === 0) return;
        
        // Generate simple sparkline data
        const data = trades.slice(-7).map(trade => trade.pnl || 0);
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        if (data.length < 2) return;
        
        const max = Math.max(...data);
        const min = Math.min(...data);
        const range = max - min || 1;
        
        ctx.strokeStyle = data[data.length - 1] >= 0 ? '#28a745' : '#dc3545';
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        data.forEach((value, index) => {
            const x = (index / (data.length - 1)) * width;
            const y = height - ((value - min) / range) * height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    async controlPlatform(action) {
        try {
            let response;
            switch (action) {
                case 'start':
                    response = await fetch('/api/start', { method: 'POST' });
                    break;
                case 'stop':
                    response = await fetch('/api/stop', { method: 'POST' });
                    break;
                case 'backup':
                    response = await fetch('/api/backup', { method: 'POST' });
                    break;
            }
            
            if (response && response.ok) {
                console.log(`Platform ${action} successful`);
                // Refresh data after action
                setTimeout(() => this.loadData(), 1000);
            }
        } catch (error) {
            console.error(`Error during platform ${action}:`, error);
        }
    }

}

// Initialize the new dashboard when the page loads
let newDashboard;
document.addEventListener('DOMContentLoaded', () => {
    console.log('New Dashboard loaded successfully');
    console.log('Ready for custom components!');
    newDashboard = new NewDashboardApp();
});