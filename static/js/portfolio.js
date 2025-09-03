// Portfolio Page Vanilla JavaScript Implementation
class PortfolioApp {
    constructor() {
        this.state = {
            holdings: [],
            pnlData: { labels: [], data: [] },
            riskData: { risk_level: '', risk_score: 0, suggestions: [] },
            assetClassFilter: '',
            dateFrom: '',
            dateTo: '',
            loading: true,
            totalValue: 0,
            totalPnL: 0
        };
        
        this.init();
    }

    async init() {
        await this.loadPortfolio();
        this.render();
        
        // Refresh data every 30 seconds
        setInterval(() => this.loadPortfolio(), 30000);
    }

    async loadPortfolio() {
        this.state.loading = true;
        this.render();
        
        try {
            const params = {};
            if (this.state.assetClassFilter) params.asset_class = this.state.assetClassFilter;
            if (this.state.dateFrom) params.date_from = this.state.dateFrom;
            if (this.state.dateTo) params.date_to = this.state.dateTo;
            
            const [holdingsRes, pnlRes, riskRes] = await Promise.all([
                this.fetchAPI('/api/portfolio/holdings', { params }),
                this.fetchAPI('/api/portfolio/pnl', { params }),
                this.fetchAPI('/api/portfolio/risk')
            ]);
            
            this.state.holdings = holdingsRes.holdings || [];
            this.state.pnlData = pnlRes;
            this.state.riskData = riskRes;
            
            // Calculate totals
            this.state.totalValue = this.state.holdings.reduce((sum, h) => sum + h.value, 0);
            this.state.totalPnL = this.state.pnlData.data && this.state.pnlData.data.length > 0 
                ? this.state.pnlData.data[this.state.pnlData.data.length - 1] 
                : 0;
            
        } catch (e) {
            console.error('Error loading portfolio:', e);
        } finally {
            this.state.loading = false;
            this.render();
        }
    }

    async fetchAPI(endpoint, options = {}) {
        try {
            const params = options.params || {};
            const queryString = new URLSearchParams(params).toString();
            const url = queryString ? `${endpoint}?${queryString}` : endpoint;
            
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return await response.json();
        } catch (e) {
            console.error(`Error fetching ${endpoint}:`, e);
            return {};
        }
    }

    async exportPortfolio() {
        try {
            const response = await this.fetchAPI('/api/portfolio/export');
            if (response.url) {
                window.location.href = response.url;
            } else {
                alert(response.message || 'Export feature coming soon');
            }
        } catch (e) {
            console.error('Export failed:', e);
            alert('Export failed');
        }
    }

    setAssetClassFilter(assetClass) {
        this.state.assetClassFilter = assetClass;
        this.loadPortfolio();
    }

    setDateFrom(date) {
        this.state.dateFrom = date;
    }

    setDateTo(date) {
        this.state.dateTo = date;
    }

    applyFilters() {
        this.loadPortfolio();
    }

    render() {
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
                        <!-- Center navigation links -->
                        <ul class="navbar-nav mx-auto" role="menubar">
                            <li class="nav-item" role="none">
                                <a class="nav-link" href="/" role="menuitem" aria-label="Dashboard">
                                    <i class="bi bi-speedometer2 me-1" aria-hidden="true"></i>Dashboard
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link" href="/alerts" role="menuitem" aria-label="Alerts">
                                    <i class="bi bi-bell me-1" aria-hidden="true"></i>Alerts
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link" href="/trades" role="menuitem" aria-label="Trades">
                                    <i class="bi bi-arrow-left-right me-1" aria-hidden="true"></i>Trades
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link active" href="/portfolio" role="menuitem" aria-label="Portfolio">
                                    <i class="bi bi-briefcase me-1" aria-hidden="true"></i>Portfolio
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link" href="/analysis" role="menuitem" aria-label="Analysis">
                                    <i class="bi bi-graph-up me-1" aria-hidden="true"></i>Analysis
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link" href="/screening" role="menuitem" aria-label="Screening">
                                    <i class="bi bi-funnel me-1" aria-hidden="true"></i>Screening
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link" href="/backtesting" role="menuitem" aria-label="Backtesting">
                                    <i class="bi bi-clock-history me-1" aria-hidden="true"></i>Backtesting
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link" href="/community" role="menuitem" aria-label="Community">
                                    <i class="bi bi-people me-1" aria-hidden="true"></i>Community
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link" href="/settings" role="menuitem" aria-label="Settings">
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

            <!-- Main Content -->
            <div class="container-fluid p-4">
                <div class="row mb-4">
                    <div class="col">
                        <h2>Portfolio Management</h2>
                        <p class="text-muted">Monitor and optimize your investment portfolio</p>
                    </div>
                </div>

                <!-- Portfolio Summary Cards -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card bg-primary text-white">
                            <div class="card-body">
                                <h5 class="card-title">Total Value</h5>
                                <h3>$${this.state.totalValue.toLocaleString()}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-${this.state.totalPnL >= 0 ? 'success' : 'danger'} text-white">
                            <div class="card-body">
                                <h5 class="card-title">Total P&L</h5>
                                <h3>$${this.state.totalPnL.toLocaleString()}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-warning text-white">
                            <div class="card-body">
                                <h5 class="card-title">Risk Level</h5>
                                <h3>${this.state.riskData.risk_level || 'Medium'}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-info text-white">
                            <div class="card-body">
                                <h5 class="card-title">Holdings</h5>
                                <h3>${this.state.holdings.length}</h3>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">Portfolio Overview</h5>
                    </div>
                    <div class="card-body">
                        <!-- Filters -->
                        <div class="row g-3 mb-3">
                            <div class="col-md-3">
                                <select class="form-select" value="${this.state.assetClassFilter}" 
                                        onchange="portfolioApp.setAssetClassFilter(this.value)">
                                    <option value="">All Asset Classes</option>
                                    <option value="crypto">Crypto</option>
                                    <option value="equity">Equity</option>
                                    <option value="bond">Bonds</option>
                                    <option value="commodity">Commodities</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <input type="date" class="form-control" placeholder="From Date" 
                                       value="${this.state.dateFrom}" 
                                       onchange="portfolioApp.setDateFrom(this.value)">
                            </div>
                            <div class="col-md-3">
                                <input type="date" class="form-control" placeholder="To Date" 
                                       value="${this.state.dateTo}" 
                                       onchange="portfolioApp.setDateTo(this.value)">
                            </div>
                            <div class="col-md-3">
                                <button class="btn btn-primary" onclick="portfolioApp.applyFilters()">Apply Filters</button>
                            </div>
                        </div>

                        <!-- Charts Row -->
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">Asset Allocation</div>
                                    <div class="card-body" style="height: 300px;">
                                        <canvas id="allocationChart"></canvas>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">P&L Over Time</div>
                                    <div class="card-body" style="height: 300px;">
                                        <canvas id="pnlChart"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Holdings Table -->
                        ${this.renderHoldingsTable()}

                        <!-- Risk Assessment -->
                        ${this.renderRiskAssessment()}

                        <!-- Export Button -->
                        <div class="text-end mt-3">
                            <button class="btn btn-outline-primary" onclick="portfolioApp.exportPortfolio()">
                                <i class="bi bi-download me-2"></i>Export Portfolio
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Initialize charts after render
        this.initCharts();
    }

    renderHoldingsTable() {
        if (this.state.loading) {
            return `
                <div class="text-center mb-4">
                    <div class="spinner-border" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
        }

        if (this.state.holdings.length === 0) {
            return '<div class="alert alert-info">No holdings found for the selected filters</div>';
        }

        return `
            <div class="card mb-4">
                <div class="card-header">Portfolio Holdings</div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    <th>Asset</th>
                                    <th>Quantity</th>
                                    <th>Current Price</th>
                                    <th>Market Value</th>
                                    <th>Allocation (%)</th>
                                    <th>Asset Class</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${this.state.holdings.map(holding => `
                                    <tr>
                                        <td><strong>${holding.asset}</strong></td>
                                        <td>${holding.quantity.toLocaleString()}</td>
                                        <td>$${holding.current_price.toLocaleString()}</td>
                                        <td>$${holding.value.toLocaleString()}</td>
                                        <td>
                                            <div class="progress" style="height: 20px;">
                                                <div class="progress-bar" role="progressbar" 
                                                     style="width: ${holding.allocation}%" 
                                                     aria-valuenow="${holding.allocation}" 
                                                     aria-valuemin="0" aria-valuemax="100">
                                                    ${holding.allocation.toFixed(1)}%
                                                </div>
                                            </div>
                                        </td>
                                        <td>
                                            <span class="badge bg-secondary">${holding.class}</span>
                                        </td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    renderRiskAssessment() {
        return `
            <div class="card mb-4">
                <div class="card-header">Risk Assessment</div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4">
                            <h6>Risk Level: 
                                <span class="badge bg-${this.getRiskBadgeColor(this.state.riskData.risk_level)}">
                                    ${this.state.riskData.risk_level || 'Medium'}
                                </span>
                            </h6>
                            <div class="mt-3">
                                <label class="form-label">Risk Score: ${this.state.riskData.risk_score || 6}/10</label>
                                <div class="progress">
                                    <div class="progress-bar bg-${this.getRiskBadgeColor(this.state.riskData.risk_level)}" 
                                         role="progressbar" 
                                         style="width: ${(this.state.riskData.risk_score || 6) * 10}%" 
                                         aria-valuenow="${this.state.riskData.risk_score || 6}" 
                                         aria-valuemin="0" aria-valuemax="10">
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-8">
                            <h6>Risk Management Suggestions:</h6>
                            <ul class="list-group list-group-flush">
                                ${(this.state.riskData.suggestions || []).map(suggestion => `
                                    <li class="list-group-item">
                                        <i class="bi bi-lightbulb text-warning me-2"></i>${suggestion}
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getRiskBadgeColor(riskLevel) {
        switch(riskLevel?.toLowerCase()) {
            case 'low': return 'success';
            case 'high': return 'danger';
            case 'medium':
            default: return 'warning';
        }
    }

    initCharts() {
        this.initAllocationChart();
        this.initPnLChart();
    }

    initAllocationChart() {
        const canvas = document.getElementById('allocationChart');
        if (!canvas || this.state.holdings.length === 0) return;

        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.allocationChart) {
            this.allocationChart.destroy();
        }

        const colors = ['#0d6efd', '#198754', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14', '#20c997'];
        
        this.allocationChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: this.state.holdings.map(h => h.asset),
                datasets: [{
                    data: this.state.holdings.map(h => h.allocation),
                    backgroundColor: colors.slice(0, this.state.holdings.length),
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    initPnLChart() {
        const canvas = document.getElementById('pnlChart');
        if (!canvas || !this.state.pnlData.data || this.state.pnlData.data.length === 0) return;

        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.pnlChart) {
            this.pnlChart.destroy();
        }

        this.pnlChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.state.pnlData.labels,
                datasets: [{
                    data: this.state.pnlData.data,
                    borderColor: '#198754',
                    backgroundColor: 'rgba(25,135,84,0.2)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: true },
                    y: { 
                        display: true,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `P&L: $${context.parsed.y.toLocaleString()}`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Initialize the portfolio application
let portfolioApp;
document.addEventListener('DOMContentLoaded', function() {
    portfolioApp = new PortfolioApp();
});