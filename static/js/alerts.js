// Alerts Page Implementation
class AlertsApp {
    constructor() {
        this.state = {
            alerts: [],
            total: 0,
            page: 1,
            perPage: 20,
            priorityFilter: '',
            assetFilter: '',
            typeFilter: '',
            loading: false
        };
        
        this.init();
    }

    async init() {
        await this.loadAlerts();
        this.render();
        
        // Auto-refresh every 30 seconds
        setInterval(() => this.loadAlerts(), 30000);
    }

    async loadAlerts() {
        this.state.loading = true;
        this.render();
        
        try {
            const params = new URLSearchParams({
                page: this.state.page,
                per_page: this.state.perPage
            });
            
            if (this.state.priorityFilter) params.append('priority', this.state.priorityFilter);
            if (this.state.assetFilter) params.append('asset', this.state.assetFilter);
            if (this.state.typeFilter) params.append('type', this.state.typeFilter);
            
            const response = await fetch(`/api/alerts?${params}`);
            const data = await response.json();
            
            this.state.alerts = data.alerts || [];
            this.state.total = data.total || 0;
            this.state.loading = false;
            
            this.render();
        } catch (e) {
            console.error('Error loading alerts:', e);
            this.state.loading = false;
            this.render();
        }
    }

    handleFilterChange(filterType, value) {
        this.state[filterType] = value;
        this.state.page = 1; // Reset to first page when filtering
        this.loadAlerts();
    }

    handlePageChange(newPage) {
        this.state.page = newPage;
        this.loadAlerts();
    }

    getPriorityBadge(priority) {
        const priorityMap = {
            3: { label: 'High', class: 'bg-danger' },
            2: { label: 'Medium', class: 'bg-warning' },
            1: { label: 'Low', class: 'bg-secondary' }
        };
        const p = priorityMap[priority] || { label: 'Unknown', class: 'bg-secondary' };
        return `<span class="badge ${p.class}">${p.label}</span>`;
    }

    formatTimestamp(timestamp) {
        try {
            const date = new Date(timestamp * 1000); // Convert Unix timestamp
            return date.toLocaleString();
        } catch (e) {
            return 'Invalid Date';
        }
    }

    render() {
        const root = document.getElementById('alerts-root');
        if (!root) return;

        const totalPages = Math.ceil(this.state.total / this.state.perPage);
        
        root.innerHTML = `
            <div class="container-fluid p-4">
                <!-- Header -->
                <div class="row mb-4">
                    <div class="col">
                        <h2>Alerts Management</h2>
                        <p class="text-muted">View and filter platform alerts</p>
                        <nav aria-label="breadcrumb">
                            <ol class="breadcrumb">
                                <li class="breadcrumb-item"><a href="/" class="text-decoration-none">Dashboard</a></li>
                                <li class="breadcrumb-item active">Alerts</li>
                            </ol>
                        </nav>
                    </div>
                </div>

                <!-- Filters -->
                <div class="card mb-4">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Filter Alerts</h5>
                        <span class="badge bg-info">${this.state.total} Total Alerts</span>
                    </div>
                    <div class="card-body">
                        <div class="row g-3">
                            <div class="col-md-3">
                                <label class="form-label">Priority</label>
                                <select class="form-select" onchange="alertsApp.handleFilterChange('priorityFilter', this.value)">
                                    <option value="">All Priorities</option>
                                    <option value="3" ${this.state.priorityFilter === '3' ? 'selected' : ''}>High</option>
                                    <option value="2" ${this.state.priorityFilter === '2' ? 'selected' : ''}>Medium</option>
                                    <option value="1" ${this.state.priorityFilter === '1' ? 'selected' : ''}>Low</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Asset</label>
                                <input type="text" class="form-control" placeholder="e.g., BTC, ETH" 
                                       value="${this.state.assetFilter}" 
                                       onchange="alertsApp.handleFilterChange('assetFilter', this.value)">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Alert Type</label>
                                <select class="form-select" onchange="alertsApp.handleFilterChange('typeFilter', this.value)">
                                    <option value="">All Types</option>
                                    <option value="TRADING_SIGNAL" ${this.state.typeFilter === 'TRADING_SIGNAL' ? 'selected' : ''}>Trading Signal</option>
                                    <option value="CORRELATION_BREAK" ${this.state.typeFilter === 'CORRELATION_BREAK' ? 'selected' : ''}>Correlation Break</option>
                                    <option value="INSTITUTIONAL_FLOW" ${this.state.typeFilter === 'INSTITUTIONAL_FLOW' ? 'selected' : ''}>Institutional Flow</option>
                                    <option value="SENTIMENT_FLOW" ${this.state.typeFilter === 'SENTIMENT_FLOW' ? 'selected' : ''}>Sentiment Flow</option>
                                    <option value="ML_ANOMALY" ${this.state.typeFilter === 'ML_ANOMALY' ? 'selected' : ''}>ML Anomaly</option>
                                </select>
                            </div>
                            <div class="col-md-3 d-flex align-items-end">
                                <button class="btn btn-primary w-100" onclick="alertsApp.loadAlerts()">
                                    <i class="bi bi-arrow-clockwise me-2"></i>Refresh
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Alerts Table -->
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Recent Alerts</h5>
                    </div>
                    <div class="card-body p-0">
                        ${this.state.loading ? this.renderLoading() : this.renderAlertsTable()}
                    </div>
                    ${totalPages > 1 ? this.renderPagination(totalPages) : ''}
                </div>
            </div>
        `;
    }

    renderLoading() {
        return `
            <div class="text-center py-5">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-muted mt-2">Loading alerts...</p>
            </div>
        `;
    }

    renderAlertsTable() {
        if (this.state.alerts.length === 0) {
            return `
                <div class="text-center py-5">
                    <i class="bi bi-exclamation-circle text-muted" style="font-size: 3rem;"></i>
                    <p class="text-muted mt-2">No alerts found matching your criteria</p>
                </div>
            `;
        }

        return `
            <div class="table-responsive">
                <table class="table table-hover mb-0">
                    <thead class="table-light">
                        <tr>
                            <th>Time</th>
                            <th>Asset</th>
                            <th>Type</th>
                            <th>Priority</th>
                            <th>Confidence</th>
                            <th>Message</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${this.state.alerts.map(alert => `
                            <tr>
                                <td class="small">${this.formatTimestamp(alert.timestamp)}</td>
                                <td><strong>${alert.symbol || alert.asset || 'N/A'}</strong></td>
                                <td><span class="badge bg-info">${alert.alert_type || alert.type || 'Unknown'}</span></td>
                                <td>${this.getPriorityBadge(alert.priority || 1)}</td>
                                <td>
                                    <div class="progress" style="height: 20px;">
                                        <div class="progress-bar" role="progressbar" 
                                             style="width: ${alert.confidence || 0}%"
                                             aria-valuenow="${alert.confidence || 0}" aria-valuemin="0" aria-valuemax="100">
                                            ${Math.round(alert.confidence || 0)}%
                                        </div>
                                    </div>
                                </td>
                                <td class="alert-message">${alert.message || 'No message'}</td>
                                <td>
                                    <div class="btn-group btn-group-sm" role="group">
                                        <button type="button" class="btn btn-outline-secondary" title="View Details">
                                            <i class="bi bi-eye"></i>
                                        </button>
                                        <button type="button" class="btn btn-outline-danger" title="Dismiss">
                                            <i class="bi bi-x"></i>
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    renderPagination(totalPages) {
        const currentPage = this.state.page;
        let pages = [];
        
        // Always show first page
        pages.push(1);
        
        // Add pages around current page
        for (let i = Math.max(2, currentPage - 1); i <= Math.min(totalPages - 1, currentPage + 1); i++) {
            if (!pages.includes(i)) pages.push(i);
        }
        
        // Always show last page
        if (totalPages > 1 && !pages.includes(totalPages)) {
            pages.push(totalPages);
        }

        return `
            <div class="card-footer">
                <nav aria-label="Alerts pagination">
                    <ul class="pagination justify-content-center mb-0">
                        <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
                            <button class="page-link" onclick="alertsApp.handlePageChange(${currentPage - 1})" 
                                    ${currentPage === 1 ? 'disabled' : ''}>Previous</button>
                        </li>
                        ${pages.map(p => `
                            <li class="page-item ${p === currentPage ? 'active' : ''}">
                                <button class="page-link" onclick="alertsApp.handlePageChange(${p})">${p}</button>
                            </li>
                        `).join('')}
                        <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
                            <button class="page-link" onclick="alertsApp.handlePageChange(${currentPage + 1})"
                                    ${currentPage === totalPages ? 'disabled' : ''}>Next</button>
                        </li>
                    </ul>
                </nav>
            </div>
        `;
    }
}

// Initialize alerts app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('alerts-root')) {
        window.alertsApp = new AlertsApp();
    }
});