// Backtesting Page as Vanilla JavaScript
class BacktestingApp {
    constructor() {
        this.state = {
            strategy: '',
            asset: '',
            dateFrom: '',
            dateTo: '',
            presets: [],
            selectedPreset: '',
            backtestId: null,
            progress: 0,
            status: '',
            results: null,
            showDetailsModal: false,
            loading: false,
            statusCheckInterval: null
        };
        
        this.charts = {};
        this.init();
    }

    async init() {
        await this.loadPresets();
        this.render();
        this.initializeDatePickers();
    }

    async loadPresets() {
        try {
            const response = await fetch('/api/backtest/presets');
            const data = await response.json();
            this.state.presets = data.presets || [];
        } catch (e) {
            console.error('Error loading presets:', e);
        }
    }

    async runBacktest() {
        if (!this.state.strategy || !this.state.asset) {
            alert('Please select a strategy and asset');
            return;
        }
        
        this.state.loading = true;
        this.state.progress = 0;
        this.state.status = 'running';
        this.state.results = null;
        this.state.backtestId = null;
        this.render();
        
        try {
            const params = {
                strategy: this.state.strategy,
                asset: this.state.asset,
                date_from: this.state.dateFrom,
                date_to: this.state.dateTo
            };
            
            const response = await fetch('/api/backtest/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });
            
            const data = await response.json();
            if (response.ok) {
                this.state.backtestId = data.id;
                this.startStatusPolling();
            } else {
                this.state.status = 'error';
                alert('Error starting backtest: ' + (data.error || 'Unknown error'));
            }
        } catch (e) {
            console.error('Error starting backtest:', e);
            this.state.status = 'error';
            alert('Error starting backtest');
        } finally {
            this.state.loading = false;
            this.render();
        }
    }

    startStatusPolling() {
        this.state.statusCheckInterval = setInterval(async () => {
            if (!this.state.backtestId) return;
            
            try {
                const response = await fetch(`/api/backtest/status/${this.state.backtestId}`);
                const data = await response.json();
                
                if (response.ok) {
                    this.state.progress = data.progress;
                    this.state.status = data.status;
                    
                    if (data.status === 'complete') {
                        this.stopStatusPolling();
                        await this.loadResults(this.state.backtestId);
                    } else if (data.status === 'error') {
                        this.stopStatusPolling();
                        alert('Backtest failed: ' + (data.error || 'Unknown error'));
                    }
                    
                    this.render();
                }
            } catch (e) {
                console.error('Error checking backtest status:', e);
            }
        }, 1000); // Check every second
    }

    stopStatusPolling() {
        if (this.state.statusCheckInterval) {
            clearInterval(this.state.statusCheckInterval);
            this.state.statusCheckInterval = null;
        }
    }

    async loadResults(backtestId) {
        try {
            const response = await fetch(`/api/backtest/results/${backtestId}`);
            const data = await response.json();
            
            if (response.ok) {
                this.state.results = data;
                this.render();
                
                // Initialize charts after DOM is updated
                setTimeout(() => this.initializeCharts(), 100);
            }
        } catch (e) {
            console.error('Error loading results:', e);
        }
    }

    async savePreset() {
        const name = prompt('Enter preset name:');
        if (name && name.trim()) {
            try {
                const preset = {
                    name: name.trim(),
                    params: {
                        strategy: this.state.strategy,
                        asset: this.state.asset,
                        date_from: this.state.dateFrom,
                        date_to: this.state.dateTo
                    }
                };
                
                const response = await fetch('/api/backtest/presets', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(preset)
                });
                
                if (response.ok) {
                    await this.loadPresets();
                    this.render();
                    alert('Preset saved successfully!');
                }
            } catch (e) {
                console.error('Error saving preset:', e);
                alert('Error saving preset');
            }
        }
    }

    async loadPreset(presetId) {
        if (!presetId) return;
        
        try {
            const response = await fetch(`/api/backtest/presets/${presetId}`);
            const data = await response.json();
            
            if (response.ok) {
                const params = data.params;
                this.state.strategy = params.strategy || '';
                this.state.asset = params.asset || '';
                this.state.dateFrom = params.date_from || '';
                this.state.dateTo = params.date_to || '';
                this.state.selectedPreset = presetId;
                this.render();
                this.initializeDatePickers();
            }
        } catch (e) {
            console.error('Error loading preset:', e);
        }
    }

    async exportResults() {
        if (!this.state.backtestId) return;
        
        try {
            const response = await fetch(`/api/backtest/export/${this.state.backtestId}`);
            if (response.ok) {
                // Create a temporary link to download the file
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `backtest_${this.state.backtestId}_results.xlsx`;
                a.click();
                window.URL.revokeObjectURL(url);
            }
        } catch (e) {
            console.error('Export failed:', e);
            alert('Export failed');
        }
    }

    openDetailsModal() {
        this.state.showDetailsModal = true;
        this.render();
    }

    closeDetailsModal() {
        this.state.showDetailsModal = false;
        this.render();
    }

    updateField(field, value) {
        this.state[field] = value;
        this.render();
    }

    initializeDatePickers() {
        // Initialize flatpickr for date inputs
        const fromDateInput = document.getElementById('date_from');
        const toDateInput = document.getElementById('date_to');
        
        if (fromDateInput) {
            flatpickr(fromDateInput, {
                dateFormat: 'Y-m-d',
                defaultDate: this.state.dateFrom || null,
                onChange: (selectedDates, dateStr) => {
                    this.state.dateFrom = dateStr;
                }
            });
        }
        
        if (toDateInput) {
            flatpickr(toDateInput, {
                dateFormat: 'Y-m-d',
                defaultDate: this.state.dateTo || null,
                onChange: (selectedDates, dateStr) => {
                    this.state.dateTo = dateStr;
                }
            });
        }
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
                                <a class="nav-link active" href="/backtesting" role="menuitem" aria-label="Backtesting">
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
            
            <div class="container-fluid p-4">
                <div class="card mb-4">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">
                            <i class="bi bi-graph-up me-2"></i>Strategy Backtesting
                        </h5>
                        <span class="badge ${this.getStatusBadgeClass()}">${this.getStatusText()}</span>
                    </div>
                    <div class="card-body">
                        ${this.renderStrategyForm()}
                        ${this.renderProgress()}
                        ${this.renderResults()}
                    </div>
                </div>
            </div>
            
            ${this.renderDetailsModal()}
        `;

        // Attach event handlers
        this.attachEventHandlers();
    }

    renderStrategyForm() {
        return `
            <form class="row g-3 mb-4" onsubmit="event.preventDefault();">
                <div class="col-md-3">
                    <label class="form-label">Strategy</label>
                    <select class="form-select" id="strategy" onchange="backtestingApp.updateField('strategy', this.value)">
                        <option value="">Select Strategy</option>
                        <option value="moving_average" ${this.state.strategy === 'moving_average' ? 'selected' : ''}>Moving Average Crossover</option>
                        <option value="rsi" ${this.state.strategy === 'rsi' ? 'selected' : ''}>RSI Strategy</option>
                        <option value="bollinger_bands" ${this.state.strategy === 'bollinger_bands' ? 'selected' : ''}>Bollinger Bands</option>
                        <option value="macd" ${this.state.strategy === 'macd' ? 'selected' : ''}>MACD Strategy</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <label class="form-label">Asset</label>
                    <input type="text" class="form-control" placeholder="e.g., BTC" value="${this.state.asset}" 
                           onchange="backtestingApp.updateField('asset', this.value)">
                </div>
                <div class="col-md-2">
                    <label class="form-label">From Date</label>
                    <input type="text" class="form-control" id="date_from" placeholder="Select date" value="${this.state.dateFrom}">
                </div>
                <div class="col-md-2">
                    <label class="form-label">To Date</label>
                    <input type="text" class="form-control" id="date_to" placeholder="Select date" value="${this.state.dateTo}">
                </div>
                <div class="col-md-3">
                    <label class="form-label">&nbsp;</label>
                    <div>
                        <button type="button" class="btn btn-primary" onclick="backtestingApp.runBacktest()" 
                                ${this.state.loading || this.state.status === 'running' ? 'disabled' : ''}>
                            ${this.state.loading ? '<span class="spinner-border spinner-border-sm me-1"></span>' : '<i class="bi bi-play me-1"></i>'}
                            Run Backtest
                        </button>
                    </div>
                </div>
                
                <div class="col-12">
                    <div class="row">
                        <div class="col-md-6">
                            <button type="button" class="btn btn-outline-secondary me-2" onclick="backtestingApp.savePreset()">
                                <i class="bi bi-bookmark me-1"></i>Save Preset
                            </button>
                            <select class="form-select d-inline" style="width: auto;" value="${this.state.selectedPreset}" 
                                    onchange="backtestingApp.loadPreset(this.value)">
                                <option value="">Load Preset...</option>
                                ${this.state.presets.map(p => `<option value="${p.id}">${p.name}</option>`).join('')}
                            </select>
                        </div>
                    </div>
                </div>
            </form>
        `;
    }

    renderProgress() {
        if (this.state.status !== 'running') return '';
        
        return `
            <div class="mb-4">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <h6 class="mb-0">Backtest Progress</h6>
                    <span class="text-muted">${this.state.progress}%</span>
                </div>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" style="width: ${this.state.progress}%;" 
                         aria-valuenow="${this.state.progress}" aria-valuemin="0" aria-valuemax="100">
                        ${this.state.progress}%
                    </div>
                </div>
            </div>
        `;
    }

    renderResults() {
        if (!this.state.results) {
            if (this.state.status === 'complete') {
                return `
                    <div class="alert alert-success text-center">
                        <i class="bi bi-check-circle me-2"></i>
                        Backtest complete - loading results...
                    </div>
                `;
            } else if (this.state.status === 'error') {
                return `
                    <div class="alert alert-danger text-center">
                        <i class="bi bi-exclamation-triangle me-2"></i>
                        Backtest failed - please try again
                    </div>
                `;
            }
            return '';
        }

        const results = this.state.results;
        
        return `
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card h-100">
                        <div class="card-header">
                            <h6 class="mb-0"><i class="bi bi-graph-up me-2"></i>P&L Over Time</h6>
                        </div>
                        <div class="card-body">
                            <div style="height: 300px;">
                                <canvas id="pnlChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card h-100">
                        <div class="card-header">
                            <h6 class="mb-0"><i class="bi bi-graph-down me-2"></i>Drawdown Analysis</h6>
                        </div>
                        <div class="card-body">
                            <div style="height: 300px;">
                                <canvas id="drawdownChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card mb-3">
                <div class="card-header">
                    <h6 class="mb-0"><i class="bi bi-clipboard-data me-2"></i>Performance Metrics</h6>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <table class="table table-bordered table-hover">
                                <tbody>
                                    <tr>
                                        <td><strong>Strategy</strong></td>
                                        <td>${results.strategy}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Asset</strong></td>
                                        <td>${results.asset}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Total P&L</strong></td>
                                        <td class="${results.total_pnl >= 0 ? 'text-success' : 'text-danger'}">
                                            $${this.formatNumber(results.total_pnl)}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><strong>Win Rate</strong></td>
                                        <td>${results.win_rate}%</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        <div class="col-md-6">
                            <table class="table table-bordered table-hover">
                                <tbody>
                                    <tr>
                                        <td><strong>Max Drawdown</strong></td>
                                        <td class="text-danger">${results.max_drawdown}%</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Total Trades</strong></td>
                                        <td>${results.total_trades}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Winning Trades</strong></td>
                                        <td class="text-success">${results.winning_trades}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Sharpe Ratio</strong></td>
                                        <td>${results.sharpe_ratio}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <div class="text-end">
                <button class="btn btn-outline-primary me-2" onclick="backtestingApp.openDetailsModal()">
                    <i class="bi bi-list-ul me-1"></i>View Details
                </button>
                <button class="btn btn-outline-success" onclick="backtestingApp.exportResults()">
                    <i class="bi bi-download me-1"></i>Export Results
                </button>
            </div>
        `;
    }

    renderDetailsModal() {
        if (!this.state.showDetailsModal) return '';

        return `
            <div class="modal show d-block" tabindex="-1" style="background-color: rgba(0,0,0,0.5);">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="bi bi-list-ul me-2"></i>Backtest Details
                            </h5>
                            <button type="button" class="btn-close" onclick="backtestingApp.closeDetailsModal()"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Strategy Configuration</h6>
                                    <ul class="list-unstyled">
                                        <li><strong>Strategy:</strong> ${this.state.results?.strategy || 'N/A'}</li>
                                        <li><strong>Asset:</strong> ${this.state.results?.asset || 'N/A'}</li>
                                        <li><strong>Period:</strong> ${this.state.dateFrom || 'N/A'} to ${this.state.dateTo || 'N/A'}</li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6>Performance Summary</h6>
                                    <ul class="list-unstyled">
                                        <li><strong>ROI:</strong> ${this.state.results ? ((this.state.results.total_pnl / 10000) * 100).toFixed(2) : 'N/A'}%</li>
                                        <li><strong>Best Month:</strong> ${this.state.results ? Math.max(...this.state.results.pnl_data).toFixed(2) : 'N/A'}</li>
                                        <li><strong>Worst Month:</strong> ${this.state.results ? Math.min(...this.state.results.pnl_data).toFixed(2) : 'N/A'}</li>
                                    </ul>
                                </div>
                            </div>
                            <div class="mt-3">
                                <h6>Monthly Performance</h6>
                                <div class="table-responsive">
                                    <table class="table table-sm table-striped">
                                        <thead>
                                            <tr>
                                                <th>Month</th>
                                                <th>P&L</th>
                                                <th>Drawdown</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            ${this.state.results ? this.state.results.labels.map((label, i) => `
                                                <tr>
                                                    <td>${label}</td>
                                                    <td class="${this.state.results.pnl_data[i] >= 0 ? 'text-success' : 'text-danger'}">
                                                        $${this.state.results.pnl_data[i].toFixed(2)}
                                                    </td>
                                                    <td class="text-danger">${this.state.results.drawdown_data[i].toFixed(2)}%</td>
                                                </tr>
                                            `).join('') : ''}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" onclick="backtestingApp.closeDetailsModal()">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getStatusBadgeClass() {
        switch(this.state.status) {
            case 'running': return 'bg-warning text-dark';
            case 'complete': return 'bg-success';
            case 'error': return 'bg-danger';
            default: return 'bg-secondary';
        }
    }

    getStatusText() {
        switch(this.state.status) {
            case 'running': return 'Running...';
            case 'complete': return 'Complete';
            case 'error': return 'Error';
            default: return 'Ready';
        }
    }

    formatNumber(num) {
        if (Math.abs(num) >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (Math.abs(num) >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        } else {
            return num.toFixed(2);
        }
    }

    initializeCharts() {
        if (!this.state.results) return;
        
        this.initPnLChart();
        this.initDrawdownChart();
    }

    initPnLChart() {
        const canvas = document.getElementById('pnlChart');
        if (!canvas) return;

        if (this.charts.pnl) {
            this.charts.pnl.destroy();
        }

        const ctx = canvas.getContext('2d');
        this.charts.pnl = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.state.results.labels,
                datasets: [{
                    label: 'Cumulative P&L',
                    data: this.state.results.pnl_data,
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
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(0);
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    initDrawdownChart() {
        const canvas = document.getElementById('drawdownChart');
        if (!canvas) return;

        if (this.charts.drawdown) {
            this.charts.drawdown.destroy();
        }

        const ctx = canvas.getContext('2d');
        this.charts.drawdown = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.state.results.labels,
                datasets: [{
                    label: 'Drawdown %',
                    data: this.state.results.drawdown_data,
                    backgroundColor: '#dc3545',
                    borderColor: '#dc3545',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    attachEventHandlers() {
        // Event handlers are attached via onclick attributes in HTML
        // Re-initialize date pickers after render
        setTimeout(() => this.initializeDatePickers(), 100);
    }

    // Cleanup when page is unloaded
    destroy() {
        this.stopStatusPolling();
        if (this.charts.pnl) this.charts.pnl.destroy();
        if (this.charts.drawdown) this.charts.drawdown.destroy();
    }
}

// Initialize the app when the page loads
let backtestingApp;
document.addEventListener('DOMContentLoaded', () => {
    backtestingApp = new BacktestingApp();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (backtestingApp) {
        backtestingApp.destroy();
    }
});