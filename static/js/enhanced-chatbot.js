/* Enhanced Chatbot Widget - Modern implementation with advanced features */
class EnhancedChatbot {
    constructor() {
        this.isOpen = false;
        this.isLoading = false;
        this.messages = [];
        this.messageHistory = [];
        this.container = null;
        this.widget = null;
        this.button = null;
        this.messagesContainer = null;
        this.messageInput = null;
        this.sendButton = null;
        this.quickActions = [
            'Market trends today',
            'Portfolio analysis',
            'Trading signals',
            'Risk assessment',
            'Help'
        ];
        
        this.init();
    }

    init() {
        try {
            this.createWidget();
            this.setupEventListeners();
            this.loadCSS();
            console.log('✅ Enhanced Chatbot initialized successfully');
        } catch (error) {
            console.error('❌ Enhanced Chatbot initialization failed:', error);
        }
    }

    loadCSS() {
        // Load enhanced chatbot CSS
        const cssLink = document.createElement('link');
        cssLink.rel = 'stylesheet';
        cssLink.href = '/static/css/enhanced-chatbot.css';
        document.head.appendChild(cssLink);
    }

    createWidget() {
        // Remove any existing chatbot widgets
        const existingWidget = document.querySelector('.enhanced-chatbot-container');
        if (existingWidget) {
            existingWidget.remove();
        }

        // Create main container
        this.container = document.createElement('div');
        this.container.className = 'enhanced-chatbot-container';
        
        this.container.innerHTML = `
            <!-- Floating Chat Button -->
            <button class="enhanced-chat-button" id="enhanced-chat-toggle" aria-label="Open AI Assistant">
                🤖
                <div class="enhanced-notification-badge" id="notification-badge" style="display: none;">1</div>
            </button>

            <!-- Chat Widget -->
            <div class="enhanced-chat-widget" id="enhanced-chat-widget">
                <!-- Header -->
                <div class="enhanced-chat-header">
                    <div class="enhanced-status-indicator"></div>
                    <div class="enhanced-chat-title">
                        <h3>AJxAI Trading Assistant</h3>
                        <div class="enhanced-chat-subtitle">Multi-Domain Market Intelligence</div>
                    </div>
                    <div class="enhanced-chat-actions">
                        <button class="enhanced-chat-action-btn" id="clear-chat" title="Clear conversation">
                            🔄
                        </button>
                        <button class="enhanced-chat-action-btn" id="minimize-chat" title="Minimize">
                            ✕
                        </button>
                    </div>
                </div>

                <!-- Messages Container -->
                <div class="enhanced-messages-container" id="enhanced-messages">
                    <div class="enhanced-empty-state">
                        <div class="enhanced-empty-icon">🤖</div>
                        <div class="enhanced-empty-title">Welcome to AJxAI Assistant</div>
                        <div class="enhanced-empty-subtitle">
                            I can help you with market analysis, trading strategies,<br>
                            geopolitical insights, and crypto trends.
                        </div>
                    </div>
                </div>

                <!-- Quick Actions -->
                <div class="enhanced-quick-actions" id="enhanced-quick-actions">
                    ${this.quickActions.map(action => 
                        `<button class="enhanced-quick-action" data-action="${action}">${action}</button>`
                    ).join('')}
                </div>

                <!-- Input Area -->
                <div class="enhanced-input-area">
                    <div class="enhanced-input-container">
                        <textarea 
                            class="enhanced-message-input" 
                            id="enhanced-message-input"
                            placeholder="Ask AJxAI about market patterns, correlations, or trading signals..."
                            rows="1"
                        ></textarea>
                        <button class="enhanced-send-button" id="enhanced-send-btn">
                            <span class="material-icons">send</span>
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Append to body
        document.body.appendChild(this.container);

        // Store references to important elements
        this.button = document.getElementById('enhanced-chat-toggle');
        this.widget = document.getElementById('enhanced-chat-widget');
        this.messagesContainer = document.getElementById('enhanced-messages');
        this.messageInput = document.getElementById('enhanced-message-input');
        this.sendButton = document.getElementById('enhanced-send-btn');
        this.quickActionsContainer = document.getElementById('enhanced-quick-actions');
        this.notificationBadge = document.getElementById('notification-badge');

        console.log('✅ Enhanced Chatbot widget created');
    }

    setupEventListeners() {
        // Toggle chat
        this.button.addEventListener('click', () => this.toggleChat());

        // Send message
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Enter key to send (Shift+Enter for new line)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.autoResizeTextarea());

        // Clear chat
        document.getElementById('clear-chat').addEventListener('click', () => this.clearChat());

        // Minimize chat
        document.getElementById('minimize-chat').addEventListener('click', () => this.toggleChat());

        // Quick actions
        this.quickActionsContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('enhanced-quick-action')) {
                const action = e.target.getAttribute('data-action');
                this.sendQuickAction(action);
            }
        });

        // Close chat when clicking outside (optional)
        document.addEventListener('click', (e) => {
            if (this.isOpen && !this.container.contains(e.target)) {
                // Optionally close chat when clicking outside
                // this.toggleChat();
            }
        });

        console.log('✅ Event listeners setup complete');
    }

    toggleChat() {
        this.isOpen = !this.isOpen;
        
        if (this.isOpen) {
            this.openChat();
        } else {
            this.closeChat();
        }
    }

    openChat() {
        this.widget.classList.add('open', 'animate-in');
        this.button.classList.add('open');
        this.button.querySelector('.chat-icon').textContent = 'close';
        
        // Hide notification badge
        this.notificationBadge.style.display = 'none';
        
        // Focus on input after animation
        setTimeout(() => {
            this.messageInput.focus();
        }, 300);

        // Hide quick actions if there are messages
        if (this.messages.length > 0) {
            this.quickActionsContainer.style.display = 'none';
        }
    }

    closeChat() {
        this.widget.classList.remove('open', 'animate-in');
        this.button.classList.remove('open');
        this.button.querySelector('.chat-icon').textContent = 'chat';
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;

        // Add user message to UI
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.autoResizeTextarea();
        this.setLoading(true);

        // Hide quick actions and empty state
        this.hideEmptyState();

        try {
            const response = await fetch('/api/ajxai/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    history: this.messageHistory.slice(-10) // Last 10 messages for context
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.addMessage('assistant', data.response, {
                    timestamp: data.timestamp,
                    tokens: data.tokens_used
                });
                
                // Store in history for context
                this.messageHistory.push(
                    { role: 'user', content: message },
                    { role: 'assistant', content: data.response }
                );
            } else {
                let errorMessage = 'I apologize, but I encountered an error processing your request. Please try again.';
                if (data.error && data.error.includes('OpenAI API key')) {
                    errorMessage = '🔑 The AI assistant requires an OpenAI API key to be configured. Please contact your administrator to set up the OPENAI_API_KEY environment variable.';
                }
                this.addMessage('assistant', errorMessage, {
                    isError: true
                });
                console.error('Chat API error:', data.error);
            }
        } catch (error) {
            this.addMessage('assistant', 'I\'m having trouble connecting right now. Please check your internet connection and try again.', {
                isError: true
            });
            console.error('Chat request failed:', error);
        } finally {
            this.setLoading(false);
        }
    }

    sendQuickAction(action) {
        // Set the input value and send
        this.messageInput.value = action;
        this.sendMessage();
    }

    addMessage(role, content, options = {}) {
        const timestamp = options.timestamp || new Date().toISOString();
        const message = { 
            role, 
            content, 
            timestamp, 
            isError: options.isError || false,
            tokens: options.tokens
        };
        
        this.messages.push(message);
        this.renderMessage(message);
        this.scrollToBottom();
    }

    renderMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = `enhanced-message ${message.role}${message.isError ? ' error' : ''}`;
        
        const timeStr = new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });

        messageElement.innerHTML = `
            <div class="enhanced-message-content">${this.formatMessageContent(message.content)}</div>
            <div class="enhanced-message-time">${timeStr}</div>
        `;

        this.messagesContainer.appendChild(messageElement);
    }

    formatMessageContent(content) {
        // Enhanced markdown-like formatting
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/`(.*?)`/g, '<code style="background: #f3f4f6; padding: 2px 4px; border-radius: 4px; font-family: monospace;">$1</code>') // Inline code
            .replace(/\n/g, '<br>') // Line breaks
            .replace(/(\b(?:https?|ftp):\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener" style="color: #667eea; text-decoration: underline;">$1</a>'); // Links
    }

    setLoading(loading) {
        this.isLoading = loading;
        
        if (loading) {
            this.sendButton.disabled = true;
            this.sendButton.innerHTML = '<div class="enhanced-loading-spinner"></div>';
            this.messageInput.disabled = true;
            this.showTypingIndicator();
        } else {
            this.sendButton.disabled = false;
            this.sendButton.innerHTML = '<span class="material-icons">send</span>';
            this.messageInput.disabled = false;
            this.hideTypingIndicator();
        }
    }

    showTypingIndicator() {
        if (document.querySelector('.enhanced-typing-indicator')) return;

        const typingElement = document.createElement('div');
        typingElement.className = 'enhanced-typing-indicator';
        typingElement.innerHTML = `
            <div class="enhanced-typing-dots">
                <div class="enhanced-typing-dot"></div>
                <div class="enhanced-typing-dot"></div>
                <div class="enhanced-typing-dot"></div>
            </div>
            <div class="enhanced-typing-text">AJxAI is analyzing...</div>
        `;
        
        this.messagesContainer.appendChild(typingElement);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingElement = document.querySelector('.enhanced-typing-indicator');
        if (typingElement) {
            typingElement.remove();
        }
    }

    hideEmptyState() {
        const emptyState = document.querySelector('.enhanced-empty-state');
        if (emptyState) {
            emptyState.style.display = 'none';
        }
        this.quickActionsContainer.style.display = 'none';
    }

    autoResizeTextarea() {
        const textarea = this.messageInput;
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 120); // Max height of 120px
        textarea.style.height = newHeight + 'px';
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    clearChat() {
        this.messages = [];
        this.messageHistory = [];
        this.messagesContainer.innerHTML = `
            <div class="enhanced-empty-state">
                <div class="enhanced-empty-icon">🤖</div>
                <div class="enhanced-empty-title">Welcome to AJxAI Assistant</div>
                <div class="enhanced-empty-subtitle">
                    I can help you with market analysis, trading strategies,<br>
                    geopolitical insights, and crypto trends.
                </div>
            </div>
        `;
        this.quickActionsContainer.style.display = 'flex';
        console.log('Chat cleared');
    }

    showNotification() {
        this.notificationBadge.style.display = 'flex';
    }

    // Method to be called externally to send a message
    sendExternalMessage(message) {
        if (!this.isOpen) {
            this.toggleChat();
        }
        this.messageInput.value = message;
        setTimeout(() => {
            this.sendMessage();
        }, 500);
    }

    // Method to update status
    updateStatus(status) {
        const statusIndicator = document.querySelector('.enhanced-status-indicator');
        if (statusIndicator) {
            statusIndicator.style.background = status === 'online' ? '#4ade80' : '#ef4444';
        }
    }
}

// Initialize Enhanced Chatbot
let enhancedChatbot = null;

function initializeEnhancedChatbot() {
    try {
        if (!enhancedChatbot) {
            enhancedChatbot = new EnhancedChatbot();
            window.enhancedChatbot = enhancedChatbot; // Make it globally accessible
            console.log('✅ Enhanced Chatbot Manager initialized successfully');
        }
    } catch (error) {
        console.error('❌ Enhanced Chatbot initialization failed:', error);
        // Retry after 2 seconds if failed
        setTimeout(initializeEnhancedChatbot, 2000);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeEnhancedChatbot);
} else {
    // DOM is already loaded
    initializeEnhancedChatbot();
}

// Additional fallback with delay
window.addEventListener('load', () => {
    setTimeout(() => {
        if (!enhancedChatbot) {
            console.log('🔄 Retrying Enhanced Chatbot initialization...');
            initializeEnhancedChatbot();
        }
    }, 1000);
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedChatbot;
}