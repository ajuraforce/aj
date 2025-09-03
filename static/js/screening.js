// Screening Page as Vanilla JavaScript
class ScreeningApp {
    constructor() {
        this.state = {
            filters: {
                asset_type: '',
                price_min: '',
                price_max: '',
                volume_min: '',
                date_from: '',
                date_to: ''
            },
            results: [],
            presets: [],
            selectedPreset: '',
            selectedResult: null,
            showAlertModal: false,
            loading: false,
            alertForm: {
                alert_type: 'buy',
                confidence_threshold: 70
            }
        };
        
        this.init();
    }

    async init() {
        await this.loadPresets();
        this.render();
        this.initializeDatePickers();
    }

    async loadPresets() {
        try {
            const response = await fetch('/api/screening/presets');
            const data = await response.json();
            this.state.presets = data.presets || [];
        } catch (e) {
            console.error('Error loading presets:', e);
        }
    }

    async runScreening() {
        this.state.loading = true;
        this.render();
        
        try {
            const response = await fetch('/api/screening/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.state.filters)
            });
            
            const data = await response.json();
            if (response.ok) {
                this.state.results = data.results || [];
            } else {
                console.error('Screening error:', data.error);
                this.state.results = [];
            }
        } catch (e) {
            console.error('Error running screening:', e);
            this.state.results = [];
        } finally {
            this.state.loading = false;
            this.render();
        }
    }

    async savePreset() {
        const name = prompt('Enter preset name:');
        if (name && name.trim()) {
            try {
                const response = await fetch('/api/screening/presets', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name: name.trim(),
                        filters: this.state.filters
                    })
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
            const response = await fetch(`/api/screening/presets/${presetId}`);
            const data = await response.json();
            
            if (response.ok) {
                this.state.filters = { ...data.filters };
                this.state.selectedPreset = presetId;
                this.render();
                this.initializeDatePickers();
            }
        } catch (e) {
            console.error('Error loading preset:', e);
        }
    }

    openAlertModal(result) {
        this.state.selectedResult = result;
        this.state.showAlertModal = true;
        this.state.alertForm.confidence_threshold = result.confidence;
        this.render();
    }

    closeAlertModal() {
        this.state.showAlertModal = false;
        this.state.selectedResult = null;
        this.render();
    }

    async createAlert() {
        if (!this.state.selectedResult) return;
        
        try {
            const alertData = {
                ...this.state.selectedResult,
                alert_type: this.state.alertForm.alert_type,
                confidence_threshold: this.state.alertForm.confidence_threshold
            };
            
            const response = await fetch('/api/screening/alert', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(alertData)
            });
            
            if (response.ok) {
                this.closeAlertModal();
                alert('Alert created successfully!');
            } else {
                alert('Error creating alert');
            }
        } catch (e) {
            console.error('Error creating alert:', e);
            alert('Error creating alert');
        }
    }

    updateFilter(key, value) {
        this.state.filters[key] = value;
        this.render();
    }

    updateAlertForm(key, value) {
        this.state.alertForm[key] = value;
        this.render();
    }

    exportResults() {
        if (this.state.results.length === 0) {
            alert('No results to export');
            return;
        }
        
        // Create CSV content
        const headers = ['Asset', 'Price', 'Volume', 'Signal', 'Confidence', 'Type'];
        const csvContent = [
            headers.join(','),
            ...this.state.results.map(r => [
                r.asset,
                r.price,
                r.volume,
                r.signal,
                r.confidence,
                r.type
            ].join(','))
        ].join('\n');
        
        // Download CSV
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `screening_results_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    initializeDatePickers() {
        // Initialize flatpickr for date inputs
        const fromDateInput = document.getElementById('date_from');
        const toDateInput = document.getElementById('date_to');
        
        if (fromDateInput) {
            flatpickr(fromDateInput, {
                dateFormat: 'Y-m-d',
                onChange: (selectedDates, dateStr) => {
                    this.state.filters.date_from = dateStr;
                }
            });
        }
        
        if (toDateInput) {
            flatpickr(toDateInput, {
                dateFormat: 'Y-m-d',
                onChange: (selectedDates, dateStr) => {
                    this.state.filters.date_to = dateStr;
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
                                <a class="nav-link active" href="/screening" role="menuitem" aria-label="Screening">
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
            
            <div class="container-fluid p-4">
                <div class="card mb-4">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">
                            <i class="bi bi-funnel me-2"></i>Asset Screening
                        </h5>
                        <span class="badge bg-primary">${this.state.results.length} Results</span>
                    </div>
                    <div class="card-body">
                        ${this.renderFilterForm()}
                        ${this.renderResults()}
                    </div>
                </div>
            </div>
            
            ${this.renderAlertModal()}
        `;

        // Attach event handlers
        this.attachEventHandlers();
    }

    renderFilterForm() {
        return `
            <form class="row g-3 mb-4" onsubmit="event.preventDefault();">
                <div class="col-md-2">
                    <label class="form-label">Asset Type</label>
                    <select class="form-select" id="asset_type" onchange="screeningApp.updateFilter('asset_type', this.value)">
                        <option value="">All Assets</option>
                        <option value="crypto" ${this.state.filters.asset_type === 'crypto' ? 'selected' : ''}>Crypto</option>
                        <option value="equity" ${this.state.filters.asset_type === 'equity' ? 'selected' : ''}>Equity</option>
                    </select>
                </div>
                <div class="col-md-2">
                    <label class="form-label">Min Price</label>
                    <input type="number" class="form-control" placeholder="0" value="${this.state.filters.price_min}" 
                           onchange="screeningApp.updateFilter('price_min', this.value)">
                </div>
                <div class="col-md-2">
                    <label class="form-label">Max Price</label>
                    <input type="number" class="form-control" placeholder="No limit" value="${this.state.filters.price_max}" 
                           onchange="screeningApp.updateFilter('price_max', this.value)">
                </div>
                <div class="col-md-2">
                    <label class="form-label">Min Volume</label>
                    <input type="number" class="form-control" placeholder="0" value="${this.state.filters.volume_min}" 
                           onchange="screeningApp.updateFilter('volume_min', this.value)">
                </div>
                <div class="col-md-2">
                    <label class="form-label">From Date</label>
                    <input type="text" class="form-control" id="date_from" placeholder="Select date" value="${this.state.filters.date_from}">
                </div>
                <div class="col-md-2">
                    <label class="form-label">To Date</label>
                    <input type="text" class="form-control" id="date_to" placeholder="Select date" value="${this.state.filters.date_to}">
                </div>
                
                <div class="col-12">
                    <div class="row align-items-end">
                        <div class="col-md-4">
                            <button type="button" class="btn btn-primary me-2" onclick="screeningApp.runScreening()" ${this.state.loading ? 'disabled' : ''}>
                                ${this.state.loading ? '<span class="spinner-border spinner-border-sm me-1"></span>' : '<i class="bi bi-search me-1"></i>'}
                                Run Screening
                            </button>
                            <button type="button" class="btn btn-outline-secondary me-2" onclick="screeningApp.savePreset()">
                                <i class="bi bi-bookmark me-1"></i>Save Preset
                            </button>
                        </div>
                        <div class="col-md-3">
                            <select class="form-select" value="${this.state.selectedPreset}" onchange="screeningApp.loadPreset(this.value)">
                                <option value="">Load Preset...</option>
                                ${this.state.presets.map(p => `<option value="${p.id}">${p.name}</option>`).join('')}
                            </select>
                        </div>
                        <div class="col-md-2 ms-auto">
                            <button type="button" class="btn btn-outline-success" onclick="screeningApp.exportResults()" 
                                    ${this.state.results.length === 0 ? 'disabled' : ''}>
                                <i class="bi bi-download me-1"></i>Export
                            </button>
                        </div>
                    </div>
                </div>
            </form>
        `;
    }

    renderResults() {
        if (this.state.loading) {
            return `
                <div class="text-center p-5">
                    <div class="spinner-border" role="status">
                        <span class="visually-hidden">Screening in progress...</span>
                    </div>
                    <p class="mt-3">Running screening analysis...</p>
                </div>
            `;
        }

        if (this.state.results.length === 0) {
            return `
                <div class="alert alert-info text-center">
                    <i class="bi bi-info-circle me-2"></i>
                    No results found. Adjust your filters and run the screening again.
                </div>
            `;
        }

        return `
            <div class="table-responsive">
                <table class="table table-striped table-hover">
                    <thead class="table-dark">
                        <tr>
                            <th><i class="bi bi-currency-exchange me-1"></i>Asset</th>
                            <th><i class="bi bi-cash me-1"></i>Price</th>
                            <th><i class="bi bi-bar-chart me-1"></i>Volume</th>
                            <th><i class="bi bi-arrow-up-down me-1"></i>Signal</th>
                            <th><i class="bi bi-percent me-1"></i>Confidence</th>
                            <th><i class="bi bi-tag me-1"></i>Type</th>
                            <th><i class="bi bi-gear me-1"></i>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.state.results.map(result => `
                            <tr>
                                <td><strong>${result.asset}</strong></td>
                                <td>$${this.formatNumber(result.price)}</td>
                                <td>${this.formatNumber(result.volume)}</td>
                                <td>
                                    <span class="badge ${this.getSignalBadgeClass(result.signal)}">
                                        ${result.signal}
                                    </span>
                                </td>
                                <td>
                                    <div class="progress" style="height: 20px;">
                                        <div class="progress-bar" role="progressbar" 
                                             style="width: ${result.confidence}%;" 
                                             aria-valuenow="${result.confidence}" aria-valuemin="0" aria-valuemax="100">
                                            ${result.confidence}%
                                        </div>
                                    </div>
                                </td>
                                <td>
                                    <span class="badge ${result.type === 'crypto' ? 'bg-warning' : 'bg-info'}">
                                        ${result.type}
                                    </span>
                                </td>
                                <td>
                                    <button class="btn btn-outline-primary btn-sm" 
                                            onclick="screeningApp.openAlertModal(${JSON.stringify(result).replace(/"/g, '&quot;')})">
                                        <i class="bi bi-bell me-1"></i>Alert
                                    </button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    renderAlertModal() {
        if (!this.state.showAlertModal || !this.state.selectedResult) {
            return '';
        }

        return `
            <div class="modal show d-block" tabindex="-1" style="background-color: rgba(0,0,0,0.5);">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="bi bi-bell me-2"></i>Create Alert from Screening
                            </h5>
                            <button type="button" class="btn-close" onclick="screeningApp.closeAlertModal()"></button>
                        </div>
                        <div class="modal-body">
                            <form>
                                <div class="mb-3">
                                    <label class="form-label">Asset</label>
                                    <input type="text" class="form-control" value="${this.state.selectedResult.asset}" readonly>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Current Price</label>
                                    <input type="text" class="form-control" value="$${this.formatNumber(this.state.selectedResult.price)}" readonly>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Alert Type</label>
                                    <select class="form-select" value="${this.state.alertForm.alert_type}" 
                                            onchange="screeningApp.updateAlertForm('alert_type', this.value)">
                                        <option value="buy">Buy Signal</option>
                                        <option value="sell">Sell Signal</option>
                                        <option value="watch">Watch List</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Confidence Threshold (%)</label>
                                    <input type="number" class="form-control" min="1" max="100" 
                                           value="${this.state.alertForm.confidence_threshold}"
                                           onchange="screeningApp.updateAlertForm('confidence_threshold', this.value)">
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" onclick="screeningApp.closeAlertModal()">Cancel</button>
                            <button type="button" class="btn btn-primary" onclick="screeningApp.createAlert()">
                                <i class="bi bi-bell me-1"></i>Create Alert
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        } else {
            return num.toFixed(2);
        }
    }

    getSignalBadgeClass(signal) {
        switch(signal.toUpperCase()) {
            case 'BUY': return 'bg-success';
            case 'SELL': return 'bg-danger';
            case 'HOLD': return 'bg-warning';
            default: return 'bg-secondary';
        }
    }

    attachEventHandlers() {
        // Event handlers are attached via onclick attributes in HTML
        // Re-initialize date pickers after render
        setTimeout(() => this.initializeDatePickers(), 100);
    }
}

// Initialize the app when the page loads
let screeningApp;
document.addEventListener('DOMContentLoaded', () => {
    screeningApp = new ScreeningApp();
});