// Settings Page Vanilla JavaScript Implementation
class SettingsApp {
    constructor() {
        this.state = {
            settings: {},
            activeTab: 'profile',
            loading: true,
            message: { text: '', type: 'success' },
            submitting: false,
            backupInProgress: false
        };
        
        this.socket = null;
        this.init();
    }

    async init() {
        this.setupSocketIO();
        await this.loadSettings();
        this.render();
    }

    setupSocketIO() {
        try {
            this.socket = io();
            
            this.socket.on('settings_updated', (data) => {
                console.log('Settings updated:', data);
                this.loadSettings();
            });
            
            this.socket.on('backup_status', (data) => {
                if (data.status === 'completed') {
                    this.showMessage('Backup completed successfully', 'success');
                    this.state.backupInProgress = false;
                    this.loadSettings();
                }
            });
            
            console.log('Socket.IO connected for real-time settings updates');
        } catch (error) {
            console.error('Socket.IO connection failed:', error);
        }
    }

    async loadSettings() {
        this.state.loading = true;
        this.render();
        
        try {
            const response = await fetch('/api/settings');
            if (response.ok) {
                this.state.settings = await response.json();
            }
        } catch (error) {
            console.error('Error loading settings:', error);
            this.showMessage('Failed to load settings', 'danger');
        } finally {
            this.state.loading = false;
            this.render();
        }
    }

    async updateSettings(section, data) {
        this.state.submitting = true;
        this.render();
        
        try {
            const response = await fetch(`/api/settings/${section}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            if (response.ok) {
                this.showMessage(result.message, 'success');
            } else {
                this.showMessage(result.error || 'Update failed', 'danger');
            }
        } catch (error) {
            console.error('Error updating settings:', error);
            this.showMessage('Update failed', 'danger');
        } finally {
            this.state.submitting = false;
            this.render();
        }
    }

    async addApiKey() {
        const name = document.getElementById('api-key-name').value;
        const key = document.getElementById('api-key-value').value;
        
        if (!name || !key) {
            this.showMessage('Please fill in both name and key', 'warning');
            return;
        }
        
        await this.updateSettings('api-keys', { name, key });
        
        // Clear form
        document.getElementById('api-key-name').value = '';
        document.getElementById('api-key-value').value = '';
    }

    async deleteApiKey(keyId) {
        if (!confirm('Are you sure you want to delete this API key?')) return;
        
        try {
            const response = await fetch(`/api/settings/api-keys/${keyId}`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            if (response.ok) {
                this.showMessage(result.message, 'success');
            } else {
                this.showMessage(result.error || 'Delete failed', 'danger');
            }
        } catch (error) {
            console.error('Error deleting API key:', error);
            this.showMessage('Delete failed', 'danger');
        }
    }

    async enable2FA() {
        const code = document.getElementById('2fa-code').value;
        
        if (!code) {
            this.showMessage('Please enter verification code', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/settings/security/2fa', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code })
            });
            
            const result = await response.json();
            if (response.ok) {
                this.showMessage(result.message, 'success');
                document.getElementById('2fa-code').value = '';
            } else {
                this.showMessage(result.error || '2FA setup failed', 'danger');
            }
        } catch (error) {
            console.error('Error setting up 2FA:', error);
            this.showMessage('2FA setup failed', 'danger');
        }
    }

    async changePassword() {
        const currentPassword = document.getElementById('current-password').value;
        const newPassword = document.getElementById('new-password').value;
        const confirmPassword = document.getElementById('confirm-password').value;
        
        if (!currentPassword || !newPassword || !confirmPassword) {
            this.showMessage('Please fill in all password fields', 'warning');
            return;
        }
        
        if (newPassword !== confirmPassword) {
            this.showMessage('New passwords do not match', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/settings/security/password', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    current_password: currentPassword, 
                    new_password: newPassword 
                })
            });
            
            const result = await response.json();
            if (response.ok) {
                this.showMessage(result.message, 'success');
                // Clear form
                document.getElementById('current-password').value = '';
                document.getElementById('new-password').value = '';
                document.getElementById('confirm-password').value = '';
            } else {
                this.showMessage(result.error || 'Password change failed', 'danger');
            }
        } catch (error) {
            console.error('Error changing password:', error);
            this.showMessage('Password change failed', 'danger');
        }
    }

    async runBackup() {
        this.state.backupInProgress = true;
        this.render();
        
        try {
            const response = await fetch('/api/settings/backup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'manual' })
            });
            
            const result = await response.json();
            if (response.ok) {
                this.showMessage(result.message, 'success');
            } else {
                this.showMessage(result.error || 'Backup failed', 'danger');
                this.state.backupInProgress = false;
            }
        } catch (error) {
            console.error('Error running backup:', error);
            this.showMessage('Backup failed', 'danger');
            this.state.backupInProgress = false;
        }
        
        this.render();
    }

    showMessage(text, type = 'success') {
        this.state.message = { text, type };
        this.render();
        
        // Auto-hide message after 5 seconds
        setTimeout(() => {
            this.state.message = { text: '', type: 'success' };
            this.render();
        }, 5000);
    }

    setActiveTab(tabName) {
        this.state.activeTab = tabName;
        this.render();
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
                                <a class="nav-link active" href="/settings" role="menuitem" aria-label="Settings">
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
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">
                            <i class="fas fa-cog me-2"></i>Settings
                        </h5>
                    </div>
                    <div class="card-body">
                        ${this.state.message.text ? this.renderMessage() : ''}
                        
                        ${this.state.loading ? this.renderLoading() : this.renderTabs()}
                    </div>
                </div>
            </div>
        `;

        this.bindEvents();
    }

    renderMessage() {
        return `
            <div class="alert alert-${this.state.message.type} alert-dismissible fade show" role="alert">
                ${this.state.message.text}
                <button type="button" class="btn-close" onclick="app.state.message = {text: '', type: 'success'}; app.render();"></button>
            </div>
        `;
    }

    renderLoading() {
        return `
            <div class="text-center py-4">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        `;
    }

    renderTabs() {
        const tabs = [
            { id: 'profile', name: 'Profile', icon: 'fas fa-user' },
            { id: 'api-keys', name: 'API Keys', icon: 'fas fa-key' },
            { id: 'integrations', name: 'Integrations', icon: 'fas fa-plug' },
            { id: 'notifications', name: 'Notifications', icon: 'fas fa-bell' },
            { id: 'security', name: 'Security', icon: 'fas fa-shield-alt' },
            { id: 'backup', name: 'Backup', icon: 'fas fa-save' }
        ];

        return `
            <!-- Tab Navigation -->
            <ul class="nav nav-tabs mb-4">
                ${tabs.map(tab => `
                    <li class="nav-item">
                        <a class="nav-link ${this.state.activeTab === tab.id ? 'active' : ''}" 
                           href="#" onclick="app.setActiveTab('${tab.id}')">
                            <i class="${tab.icon} me-2"></i>${tab.name}
                        </a>
                    </li>
                `).join('')}
            </ul>

            <!-- Tab Content -->
            <div class="tab-content">
                ${this.renderTabContent()}
            </div>
        `;
    }

    renderTabContent() {
        switch (this.state.activeTab) {
            case 'profile': return this.renderProfileTab();
            case 'api-keys': return this.renderApiKeysTab();
            case 'integrations': return this.renderIntegrationsTab();
            case 'notifications': return this.renderNotificationsTab();
            case 'security': return this.renderSecurityTab();
            case 'backup': return this.renderBackupTab();
            default: return this.renderProfileTab();
        }
    }

    renderProfileTab() {
        const profile = this.state.settings.profile || {};
        return `
            <div class="row">
                <div class="col-md-6">
                    <form onsubmit="app.handleProfileSubmit(event)">
                        <div class="mb-3">
                            <label class="form-label">Name</label>
                            <input type="text" class="form-control" id="profile-name" 
                                   value="${profile.name || ''}" ${this.state.submitting ? 'disabled' : ''}>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Email</label>
                            <input type="email" class="form-control" id="profile-email" 
                                   value="${profile.email || ''}" ${this.state.submitting ? 'disabled' : ''}>
                        </div>
                        <button type="submit" class="btn btn-primary" ${this.state.submitting ? 'disabled' : ''}>
                            ${this.state.submitting ? '<span class="spinner-border spinner-border-sm me-2"></span>' : ''}
                            Save Profile
                        </button>
                    </form>
                </div>
            </div>
        `;
    }

    renderApiKeysTab() {
        const apiKeys = this.state.settings.api_keys || [];
        return `
            <div class="row">
                <div class="col-md-8">
                    <h6>Existing API Keys</h6>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Key</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${apiKeys.length > 0 ? apiKeys.map(key => `
                                    <tr>
                                        <td>${key.name}</td>
                                        <td><code>${key.key}</code></td>
                                        <td>
                                            <button class="btn btn-outline-danger btn-sm" 
                                                    onclick="app.deleteApiKey(${key.id})">
                                                <i class="fas fa-trash"></i> Delete
                                            </button>
                                        </td>
                                    </tr>
                                `).join('') : '<tr><td colspan="3" class="text-muted">No API keys configured</td></tr>'}
                            </tbody>
                        </table>
                    </div>
                    
                    <hr class="my-4">
                    
                    <h6>Add New API Key</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="form-label">Key Name</label>
                                <input type="text" class="form-control" id="api-key-name" 
                                       placeholder="e.g., OpenAI">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="form-label">Key Value</label>
                                <input type="password" class="form-control" id="api-key-value" 
                                       placeholder="Enter API key">
                            </div>
                        </div>
                    </div>
                    <button class="btn btn-primary" onclick="app.addApiKey()">
                        <i class="fas fa-plus me-2"></i>Add API Key
                    </button>
                </div>
            </div>
        `;
    }

    renderIntegrationsTab() {
        const integrations = this.state.settings.integrations || {};
        return `
            <div class="row">
                <div class="col-md-6">
                    <h6>Platform Integrations</h6>
                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" id="telegram-switch" 
                               ${integrations.telegram ? 'checked' : ''}
                               onchange="app.updateIntegration('telegram', this.checked)">
                        <label class="form-check-label" for="telegram-switch">
                            <i class="fab fa-telegram me-2"></i>Telegram Bot
                        </label>
                    </div>
                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" id="reddit-switch" 
                               ${integrations.reddit ? 'checked' : ''}
                               onchange="app.updateIntegration('reddit', this.checked)">
                        <label class="form-check-label" for="reddit-switch">
                            <i class="fab fa-reddit me-2"></i>Reddit Posting
                        </label>
                    </div>
                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" id="discord-switch" 
                               ${integrations.discord ? 'checked' : ''}
                               onchange="app.updateIntegration('discord', this.checked)">
                        <label class="form-check-label" for="discord-switch">
                            <i class="fab fa-discord me-2"></i>Discord Webhooks
                        </label>
                    </div>
                </div>
            </div>
        `;
    }

    renderNotificationsTab() {
        const notifications = this.state.settings.notifications || {};
        return `
            <div class="row">
                <div class="col-md-6">
                    <h6>Notification Preferences</h6>
                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" id="email-notifications" 
                               ${notifications.email ? 'checked' : ''}
                               onchange="app.updateNotification('email', this.checked)">
                        <label class="form-check-label" for="email-notifications">
                            <i class="fas fa-envelope me-2"></i>Email Notifications
                        </label>
                    </div>
                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" id="sms-notifications" 
                               ${notifications.sms ? 'checked' : ''}
                               onchange="app.updateNotification('sms', this.checked)">
                        <label class="form-check-label" for="sms-notifications">
                            <i class="fas fa-sms me-2"></i>SMS Notifications
                        </label>
                    </div>
                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" id="push-notifications" 
                               ${notifications.push ? 'checked' : ''}
                               onchange="app.updateNotification('push', this.checked)">
                        <label class="form-check-label" for="push-notifications">
                            <i class="fas fa-bell me-2"></i>Push Notifications
                        </label>
                    </div>
                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" id="alert-notifications" 
                               ${notifications.alerts ? 'checked' : ''}
                               onchange="app.updateNotification('alerts', this.checked)">
                        <label class="form-check-label" for="alert-notifications">
                            <i class="fas fa-exclamation-triangle me-2"></i>Trading Alerts
                        </label>
                    </div>
                </div>
            </div>
        `;
    }

    renderSecurityTab() {
        const security = this.state.settings.security || {};
        return `
            <div class="row">
                <div class="col-md-6">
                    <h6>Two-Factor Authentication</h6>
                    ${security['2fa_enabled'] ? 
                        '<div class="alert alert-success"><i class="fas fa-check me-2"></i>2FA is enabled</div>' :
                        `<div class="mb-3">
                            <label class="form-label">Verification Code</label>
                            <input type="text" class="form-control" id="2fa-code" 
                                   placeholder="Enter code (use 'verify' or '123456')">
                            <div class="form-text">Enter verification code to enable 2FA</div>
                        </div>
                        <button class="btn btn-primary mb-4" onclick="app.enable2FA()">
                            <i class="fas fa-shield-alt me-2"></i>Enable 2FA
                        </button>`
                    }
                    
                    <hr>
                    
                    <h6>Change Password</h6>
                    <div class="mb-3">
                        <label class="form-label">Current Password</label>
                        <input type="password" class="form-control" id="current-password">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">New Password</label>
                        <input type="password" class="form-control" id="new-password">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Confirm New Password</label>
                        <input type="password" class="form-control" id="confirm-password">
                    </div>
                    <button class="btn btn-warning" onclick="app.changePassword()">
                        <i class="fas fa-key me-2"></i>Change Password
                    </button>
                </div>
            </div>
        `;
    }

    renderBackupTab() {
        const backup = this.state.settings.backup || {};
        return `
            <div class="row">
                <div class="col-md-6">
                    <h6>Manual Backup</h6>
                    <p class="text-muted">Last backup: ${backup.last_backup || 'Never'}</p>
                    <button class="btn btn-primary mb-4" onclick="app.runBackup()" 
                            ${this.state.backupInProgress ? 'disabled' : ''}>
                        ${this.state.backupInProgress ? 
                            '<span class="spinner-border spinner-border-sm me-2"></span>Running...' :
                            '<i class="fas fa-save me-2"></i>Run Backup Now'
                        }
                    </button>
                    
                    <hr>
                    
                    <h6>Backup Schedule</h6>
                    <div class="mb-3">
                        <label class="form-label">Frequency</label>
                        <select class="form-select" onchange="app.updateBackupSchedule(this.value)">
                            <option value="daily" ${backup.frequency === 'daily' ? 'selected' : ''}>Daily</option>
                            <option value="weekly" ${backup.frequency === 'weekly' ? 'selected' : ''}>Weekly</option>
                            <option value="monthly" ${backup.frequency === 'monthly' ? 'selected' : ''}>Monthly</option>
                        </select>
                    </div>
                    <div class="form-check form-switch">
                        <input class="form-check-input" type="checkbox" id="auto-backup" 
                               ${backup.auto_backup ? 'checked' : ''}
                               onchange="app.updateBackupSetting('auto_backup', this.checked)">
                        <label class="form-check-label" for="auto-backup">
                            Enable Automatic Backups
                        </label>
                    </div>
                </div>
            </div>
        `;
    }

    bindEvents() {
        // Event binding is handled via inline event handlers for simplicity
    }

    // Helper methods for form handling
    handleProfileSubmit(event) {
        event.preventDefault();
        const name = document.getElementById('profile-name').value;
        const email = document.getElementById('profile-email').value;
        this.updateSettings('profile', { name, email });
    }

    updateIntegration(service, enabled) {
        this.updateSettings('integrations', { [service]: enabled });
    }

    updateNotification(type, enabled) {
        this.updateSettings('notifications', { [type]: enabled });
    }

    async updateBackupSchedule(frequency) {
        try {
            const response = await fetch('/api/settings/backup/schedule', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frequency })
            });
            
            if (response.ok) {
                this.showMessage('Backup schedule updated', 'success');
            }
        } catch (error) {
            console.error('Error updating backup schedule:', error);
        }
    }

    async updateBackupSetting(setting, value) {
        try {
            const response = await fetch('/api/settings/backup/schedule', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ [setting]: value })
            });
            
            if (response.ok) {
                this.showMessage('Backup setting updated', 'success');
            }
        } catch (error) {
            console.error('Error updating backup setting:', error);
        }
    }
}

// Initialize the app when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new SettingsApp();
});