/**
 * Profile Page JavaScript
 * Handles profile management, trade history, and user settings
 */

// Global state
let profileData = {};
let tradeHistory = [];

// Initialize page when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    loadProfileData();
    loadTradeHistory();
    setupEventListeners();
});

/**
 * Initialize page components
 */
function initializePage() {
    console.log('Initializing profile page...');
    
    // Setup avatar display logic
    const avatarImage = document.getElementById('avatarImage');
    const defaultAvatar = document.getElementById('defaultAvatar');
    
    if (avatarImage) {
        avatarImage.onerror = function() {
            // Hide image and show default avatar
            avatarImage.style.display = 'none';
            defaultAvatar.style.display = 'flex';
        };
        
        avatarImage.onload = function() {
            // Show image and hide default avatar
            if (avatarImage.src && avatarImage.src !== '') {
                avatarImage.style.display = 'block';
                defaultAvatar.style.display = 'none';
            }
        };
    }
}

/**
 * Load user profile data from API
 */
async function loadProfileData() {
    try {
        const response = await fetch('/api/profile/get');
        if (!response.ok) throw new Error('Failed to load profile');
        
        profileData = await response.json();
        displayProfileData(profileData);
        
    } catch (error) {
        console.error('Error loading profile:', error);
        showAlert('Failed to load profile data', 'danger');
        
        // Set default data if API fails
        profileData = {
            name: 'AJxAI User',
            email: 'user@ajxai.com',
            avatar: '/static/images/default-avatar.png',
            bio: 'Welcome to AJxAI Trading Platform',
            privacy: 'public',
            notifications: { email: true, push: false }
        };
        displayProfileData(profileData);
    }
}

/**
 * Display profile data in the UI
 */
function displayProfileData(data) {
    // Update profile display
    document.getElementById('profileName').textContent = data.name || 'AJxAI User';
    document.getElementById('profileEmail').textContent = data.email || 'user@ajxai.com';
    document.getElementById('profileBio').textContent = data.bio || 'Welcome to AJxAI Trading Platform';
    
    // Handle avatar display
    const avatarImage = document.getElementById('avatarImage');
    const defaultAvatar = document.getElementById('defaultAvatar');
    
    if (data.avatar && data.avatar !== 'default.png' && data.avatar !== '') {
        avatarImage.src = data.avatar;
        avatarImage.style.display = 'block';
        defaultAvatar.style.display = 'none';
    } else {
        avatarImage.style.display = 'none';
        defaultAvatar.style.display = 'flex';
    }
    
    // Populate form fields
    document.getElementById('nameInput').value = data.name || '';
    document.getElementById('emailInput').value = data.email || '';
    document.getElementById('avatarInput').value = data.avatar || '';
    document.getElementById('bioInput').value = data.bio || '';
    document.getElementById('privacySelect').value = data.privacy || 'public';
    
    // Set notification preferences
    if (data.notifications) {
        document.getElementById('emailNotifications').checked = data.notifications.email || false;
        document.getElementById('pushNotifications').checked = data.notifications.push || false;
    }
}

/**
 * Load trade history and performance metrics
 */
async function loadTradeHistory() {
    try {
        const response = await fetch('/api/profile/history');
        if (!response.ok) throw new Error('Failed to load trade history');
        
        const historyData = await response.json();
        tradeHistory = historyData.history || [];
        
        // Update performance metrics
        document.getElementById('totalTrades').textContent = historyData.total_trades || 0;
        document.getElementById('totalPnl').textContent = `$${(historyData.total_pnl || 0).toFixed(2)}`;
        document.getElementById('winRate').textContent = `${historyData.win_rate || 0}%`;
        document.getElementById('bestTrade').textContent = historyData.best_trade ? 
            `$${historyData.best_trade.pnl.toFixed(2)}` : '-';
        
        // Update P&L color
        const pnlElement = document.getElementById('totalPnl');
        if (historyData.total_pnl > 0) {
            pnlElement.className = 'text-success';
        } else if (historyData.total_pnl < 0) {
            pnlElement.className = 'text-danger';
        } else {
            pnlElement.className = 'text-muted';
        }
        
        displayTradeHistory(tradeHistory);
        
    } catch (error) {
        console.error('Error loading trade history:', error);
        showAlert('Failed to load trade history', 'warning');
        
        // Show empty state
        document.getElementById('historyTableBody').innerHTML = 
            '<tr><td colspan="5" class="text-center text-muted">No trades found</td></tr>';
    }
}

/**
 * Display trade history in table
 */
function displayTradeHistory(history) {
    const tbody = document.getElementById('historyTableBody');
    
    if (!history || history.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No trades found</td></tr>';
        return;
    }
    
    tbody.innerHTML = history.map(trade => {
        const pnlClass = trade.pnl > 0 ? 'text-success' : trade.pnl < 0 ? 'text-danger' : 'text-muted';
        return `
            <tr>
                <td>${formatDate(trade.date)}</td>
                <td>${trade.asset}</td>
                <td><span class="badge bg-${trade.type === 'buy' ? 'success' : 'danger'}">${trade.type || 'trade'}</span></td>
                <td>${trade.quantity || '-'}</td>
                <td class="${pnlClass}">$${(trade.pnl || 0).toFixed(2)}</td>
            </tr>
        `;
    }).join('');
}

/**
 * Setup event listeners for forms and interactions
 */
function setupEventListeners() {
    // Profile form submission
    document.getElementById('profileForm').addEventListener('submit', handleProfileUpdate);
    
    // Password form submission
    document.getElementById('passwordForm').addEventListener('submit', handlePasswordChange);
    
    // Preferences form submission
    document.getElementById('preferencesForm').addEventListener('submit', handlePreferencesUpdate);
    
    // Avatar URL input - update preview on change
    document.getElementById('avatarInput').addEventListener('input', function() {
        const avatarImage = document.getElementById('avatarImage');
        if (this.value) {
            avatarImage.src = this.value;
        }
    });
}

/**
 * Handle profile form submission
 */
async function handleProfileUpdate(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const updates = {
        name: formData.get('name') || document.getElementById('nameInput').value,
        email: formData.get('email') || document.getElementById('emailInput').value,
        avatar: formData.get('avatar') || document.getElementById('avatarInput').value,
        bio: formData.get('bio') || document.getElementById('bioInput').value
    };
    
    try {
        const response = await fetch('/api/profile/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updates)
        });
        
        if (!response.ok) throw new Error('Failed to update profile');
        
        const result = await response.json();
        showAlert(result.message || 'Profile updated successfully', 'success');
        
        // Reload profile data
        await loadProfileData();
        
    } catch (error) {
        console.error('Error updating profile:', error);
        showAlert('Failed to update profile', 'danger');
    }
}

/**
 * Handle password change form submission
 */
async function handlePasswordChange(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const oldPassword = formData.get('oldPassword') || document.getElementById('oldPassword').value;
    const newPassword = formData.get('newPassword') || document.getElementById('newPassword').value;
    const confirmPassword = formData.get('confirmPassword') || document.getElementById('confirmPassword').value;
    
    // Validate passwords match
    if (newPassword !== confirmPassword) {
        showAlert('New passwords do not match', 'danger');
        return;
    }
    
    if (newPassword.length < 8) {
        showAlert('Password must be at least 8 characters long', 'danger');
        return;
    }
    
    try {
        const response = await fetch('/api/profile/password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                old_password: oldPassword,
                new_password: newPassword
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to change password');
        }
        
        const result = await response.json();
        showAlert(result.message || 'Password changed successfully', 'success');
        
        // Clear form
        event.target.reset();
        
    } catch (error) {
        console.error('Error changing password:', error);
        showAlert(error.message || 'Failed to change password', 'danger');
    }
}

/**
 * Handle preferences form submission
 */
async function handlePreferencesUpdate(event) {
    event.preventDefault();
    
    const privacy = document.getElementById('privacySelect').value;
    const emailNotifications = document.getElementById('emailNotifications').checked;
    const pushNotifications = document.getElementById('pushNotifications').checked;
    
    const updates = {
        privacy: privacy,
        notifications: {
            email: emailNotifications,
            push: pushNotifications
        }
    };
    
    try {
        const response = await fetch('/api/profile/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updates)
        });
        
        if (!response.ok) throw new Error('Failed to update preferences');
        
        const result = await response.json();
        showAlert(result.message || 'Preferences updated successfully', 'success');
        
    } catch (error) {
        console.error('Error updating preferences:', error);
        showAlert('Failed to update preferences', 'danger');
    }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alertContainer');
    
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alertElement);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertElement.parentNode) {
            alertElement.parentNode.removeChild(alertElement);
        }
    }, 5000);
}

/**
 * Format date for display
 */
function formatDate(dateString) {
    if (!dateString) return '-';
    
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    } catch (error) {
        return dateString;
    }
}

// Export functions for potential external use
window.ProfilePage = {
    loadProfileData,
    loadTradeHistory,
    showAlert
};