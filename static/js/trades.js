// Trades Page Vanilla JavaScript Implementation
class TradesApp {
    constructor() {
        this.state = {
            trades: [],
            total: 0,
            page: 1,
            perPage: 20,
            statusFilter: 'open',
            assetFilter: '',
            dateFrom: '',
            dateTo: '',
            loading: true,
            selectedTrade: null,
            pnlData: { labels: [], data: [] }
        };
        
        this.init();
    }

    async init() {
        await this.loadTrades();
        this.render();
        
        // Refresh data every 30 seconds
        setInterval(() => this.loadTrades(), 30000);
    }

    async loadTrades() {
        this.state.loading = true;
        this.render();
        
        try {
            const params = new URLSearchParams({
                status: this.state.statusFilter,
                page: this.state.page,
                per_page: this.state.perPage
            });
            
            if (this.state.assetFilter) params.append('asset', this.state.assetFilter);
            if (this.state.dateFrom) params.append('date_from', this.state.dateFrom);
            if (this.state.dateTo) params.append('date_to', this.state.dateTo);
            
            const [tradesRes, portfolioRes] = await Promise.all([
                this.fetchAPI(`/api/trades?${params}`),
                this.fetchAPI('/api/portfolio')
            ]);
            
            this.state.trades = tradesRes.trades || [];
            this.state.total = tradesRes.total || 0;
            this.state.pnlData = {
                labels: portfolioRes.labels || [],
                data: portfolioRes.pnl_history || portfolioRes.pnl_data || []
            };
            
        } catch (e) {
            console.error('Error loading trades:', e);
        } finally {
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

    async closeTrade(id) {
        try {
            const response = await fetch('/api/trades/close', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ id })
            });
            
            if (response.ok) {
                await this.loadTrades();
            } else {
                alert('Failed to close trade');
            }
        } catch (e) {
            console.error('Error closing trade:', e);
            alert('Error closing trade');
        }
    }

    setStatusFilter(status) {
        this.state.statusFilter = status;
        this.state.page = 1;
        this.loadTrades();
    }

    setAssetFilter(asset) {
        this.state.assetFilter = asset;
        this.state.page = 1;
    }

    setDateFrom(date) {
        this.state.dateFrom = date;
        this.state.page = 1;
    }

    setDateTo(date) {
        this.state.dateTo = date;
        this.state.page = 1;
    }

    setPage(page) {
        this.state.page = page;
        this.loadTrades();
    }

    showTradeDetails(trade) {
        this.state.selectedTrade = trade;
        this.render();
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('tradeDetailsModal'));
        modal.show();
    }

    render() {
        const root = document.getElementById('root');
        if (!root) return;

        const totalPages = Math.ceil(this.state.total / this.state.perPage);
        
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
                                <a class="nav-link active" href="/trades" role="menuitem" aria-label="Trades">
                                    <i class="bi bi-arrow-left-right me-1" aria-hidden="true"></i>Trades
                                </a>
                            </li>
                            <li class="nav-item" role="none">
                                <a class="nav-link" href="/portfolio" role="menuitem" aria-label="Portfolio">
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
                        <h2>Trading Management</h2>
                        <p class="text-muted">Monitor and manage your trading positions</p>
                    </div>
                </div>

                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">Trades</h5>
                    </div>
                    <div class="card-body">
                        <!-- Tabs -->
                        <ul class="nav nav-tabs mb-3">
                            <li class="nav-item">
                                <button class="nav-link ${this.state.statusFilter === 'open' ? 'active' : ''}" 
                                        onclick="tradesApp.setStatusFilter('open')">
                                    Open Trades
                                </button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link ${this.state.statusFilter === 'closed' ? 'active' : ''}" 
                                        onclick="tradesApp.setStatusFilter('closed')">
                                    Closed Trades
                                </button>
                            </li>
                        </ul>

                        <!-- Filters -->
                        <div class="row g-3 mb-3">
                            <div class="col-md-3">
                                <input type="text" class="form-control" placeholder="Filter by Asset (e.g., BTC)" 
                                       value="${this.state.assetFilter}" 
                                       onchange="tradesApp.setAssetFilter(this.value)">
                            </div>
                            <div class="col-md-3">
                                <input type="date" class="form-control" placeholder="From Date" 
                                       value="${this.state.dateFrom}" 
                                       onchange="tradesApp.setDateFrom(this.value)">
                            </div>
                            <div class="col-md-3">
                                <input type="date" class="form-control" placeholder="To Date" 
                                       value="${this.state.dateTo}" 
                                       onchange="tradesApp.setDateTo(this.value)">
                            </div>
                            <div class="col-md-3">
                                <button class="btn btn-primary" onclick="tradesApp.loadTrades()">Apply Filters</button>
                            </div>
                        </div>

                        <!-- P&L Chart -->
                        <div class="card mb-3">
                            <div class="card-header">P&L Trend</div>
                            <div class="card-body" style="height: 200px;">
                                <canvas id="pnlChart"></canvas>
                            </div>
                        </div>

                        <!-- Trades Table -->
                        ${this.renderTradesTable()}

                        <!-- Pagination -->
                        ${totalPages > 1 ? this.renderPagination(totalPages) : ''}
                    </div>
                </div>
            </div>

            <!-- Trade Details Modal -->
            ${this.renderTradeDetailsModal()}
        `;

        // Initialize chart after render
        this.initChart();
    }

    renderTradesTable() {
        if (this.state.loading) {
            return `
                <div class="text-center">
                    <div class="spinner-border" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
        }

        if (this.state.trades.length === 0) {
            return '<div class="text-center text-muted">No trades found</div>';
        }

        return `
            <div class="table-responsive">
                <table class="table table-striped table-hover">
                    <thead>
                        <tr>
                            <th>Asset</th>
                            <th>Entry Price</th>
                            <th>Exit Price</th>
                            <th>Quantity</th>
                            <th>P&L</th>
                            <th>Timestamp</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.state.trades.map(trade => `
                            <tr>
                                <td>${trade.asset}</td>
                                <td>$${trade.entry_price}</td>
                                <td>${trade.exit_price ? '$' + trade.exit_price : 'Open'}</td>
                                <td>${trade.quantity}</td>
                                <td>
                                    <span class="badge ${trade.pnl > 0 ? 'bg-success' : trade.pnl < 0 ? 'bg-danger' : 'bg-secondary'}">
                                        $${trade.pnl}
                                    </span>
                                </td>
                                <td>${new Date(trade.timestamp).toLocaleString()}</td>
                                <td>
                                    <button class="btn btn-outline-primary btn-sm" 
                                            onclick="tradesApp.showTradeDetails(${JSON.stringify(trade).replace(/"/g, '&quot;')})">
                                        Details
                                    </button>
                                    ${trade.status === 'open' ? 
                                        `<button class="btn btn-outline-danger btn-sm ms-2" 
                                                onclick="tradesApp.closeTrade('${trade.id}')">
                                            Close
                                        </button>` : ''
                                    }
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    renderPagination(totalPages) {
        const pages = [];
        for (let i = 1; i <= totalPages; i++) {
            pages.push(`
                <li class="page-item ${i === this.state.page ? 'active' : ''}">
                    <button class="page-link" onclick="tradesApp.setPage(${i})">${i}</button>
                </li>
            `);
        }

        return `
            <nav aria-label="Trades pagination">
                <ul class="pagination justify-content-center mt-3">
                    ${pages.join('')}
                </ul>
            </nav>
        `;
    }

    renderTradeDetailsModal() {
        const trade = this.state.selectedTrade;
        if (!trade) return '';

        return `
            <div class="modal fade" id="tradeDetailsModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Trade Details</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <p><strong>Asset:</strong> ${trade.asset}</p>
                            <p><strong>Entry Price:</strong> $${trade.entry_price}</p>
                            <p><strong>Exit Price:</strong> ${trade.exit_price || 'N/A'}</p>
                            <p><strong>Quantity:</strong> ${trade.quantity}</p>
                            <p><strong>P&L:</strong> $${trade.pnl}</p>
                            <p><strong>Status:</strong> ${trade.status}</p>
                            <p><strong>Side:</strong> ${trade.side || 'N/A'}</p>
                            <p><strong>Confidence:</strong> ${trade.confidence || 0}%</p>
                            <p><strong>Timestamp:</strong> ${new Date(trade.timestamp).toLocaleString()}</p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    initChart() {
        const canvas = document.getElementById('pnlChart');
        if (!canvas || !this.state.pnlData.data || this.state.pnlData.data.length === 0) return;

        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }

        this.chart = new Chart(ctx, {
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
                    y: { display: true }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `P&L: $${context.parsed.y.toFixed(2)}`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Initialize the trades application
let tradesApp;
document.addEventListener('DOMContentLoaded', function() {
    tradesApp = new TradesApp();
});