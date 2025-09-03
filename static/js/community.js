// Community Page Vanilla JavaScript Implementation
class CommunityApp {
    constructor() {
        this.state = {
            posts: [],
            selectedPost: null,
            newPost: '',
            newComment: '',
            page: 1,
            total: 0,
            loading: false,
            submittingPost: false,
            submittingComment: false
        };
        
        this.socket = null;
        this.init();
    }

    async init() {
        this.setupSocketIO();
        await this.loadPosts();
        this.render();
    }

    setupSocketIO() {
        try {
            this.socket = io();
            
            this.socket.on('new_post', (post) => {
                this.state.posts.unshift(post);
                this.render();
            });
            
            this.socket.on('new_comment', (data) => {
                const postIndex = this.state.posts.findIndex(p => p.id === data.post_id);
                if (postIndex !== -1) {
                    if (!this.state.posts[postIndex].comments) {
                        this.state.posts[postIndex].comments = [];
                    }
                    this.state.posts[postIndex].comments.push(data.comment);
                    this.render();
                }
            });
            
            this.socket.on('like_update', (data) => {
                const postIndex = this.state.posts.findIndex(p => p.id === data.post_id);
                if (postIndex !== -1) {
                    this.state.posts[postIndex].likes = data.likes;
                    this.render();
                }
            });
            
            console.log('Socket.IO connected for real-time community updates');
        } catch (error) {
            console.error('Socket.IO connection failed:', error);
        }
    }

    async loadPosts() {
        this.state.loading = true;
        this.render();
        
        try {
            const response = await fetch(`/api/community/posts?page=${this.state.page}&per_page=10`);
            if (response.ok) {
                const data = await response.json();
                this.state.posts = data.posts || [];
                this.state.total = data.total || 0;
            }
        } catch (error) {
            console.error('Error loading posts:', error);
        } finally {
            this.state.loading = false;
            this.render();
        }
    }

    async createPost() {
        if (!this.state.newPost.trim()) return;
        
        this.state.submittingPost = true;
        this.render();
        
        try {
            const response = await fetch('/api/community/post', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    content: this.state.newPost,
                    premium: false
                })
            });
            
            if (response.ok) {
                this.state.newPost = '';
                // Post will be added via Socket.IO event
            }
        } catch (error) {
            console.error('Error creating post:', error);
        } finally {
            this.state.submittingPost = false;
            this.render();
        }
    }

    async addComment(postId) {
        if (!this.state.newComment.trim()) return;
        
        this.state.submittingComment = true;
        this.render();
        
        try {
            const response = await fetch('/api/community/comment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    post_id: postId,
                    content: this.state.newComment
                })
            });
            
            if (response.ok) {
                this.state.newComment = '';
                // Comment will be added via Socket.IO event
            }
        } catch (error) {
            console.error('Error adding comment:', error);
        } finally {
            this.state.submittingComment = false;
            this.render();
        }
    }

    async likePost(postId) {
        try {
            const response = await fetch('/api/community/like', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    post_id: postId
                })
            });
            
            if (!response.ok) {
                console.error('Error liking post');
            }
            // Like count will be updated via Socket.IO event
        } catch (error) {
            console.error('Error liking post:', error);
        }
    }

    toggleComments(postId) {
        this.state.selectedPost = this.state.selectedPost === postId ? null : postId;
        this.state.newComment = '';
        this.render();
    }

    formatDate(timestamp) {
        return new Date(timestamp).toLocaleString();
    }

    render() {
        const root = document.getElementById('root');
        if (!root) return;

        const totalPages = Math.ceil(this.state.total / 10);

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
                                <a class="nav-link" href="/analysis"><i class="bi bi-bar-chart"></i>Analysis</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/live-alerts"><i class="bi bi-bell"></i>Alerts</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" href="/health"><i class="bi bi-activity"></i>Health</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link active" href="/community"><i class="bi bi-people"></i>Community</a>
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
                                <a class="nav-link active" href="/community" role="menuitem" aria-label="Community">
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
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">
                            <i class="fas fa-users me-2"></i>Community
                        </h5>
                    </div>
                    <div class="card-body">
                        <!-- New Post Form -->
                        <div class="mb-4">
                            <h6>Share your thoughts</h6>
                            <textarea 
                                class="form-control mb-3" 
                                rows="3" 
                                placeholder="Share a trading idea, market insight, or start a discussion..."
                                ${this.state.submittingPost ? 'disabled' : ''}
                                onchange="app.updateNewPost(this.value)"
                            >${this.state.newPost}</textarea>
                            <button 
                                class="btn btn-primary" 
                                onclick="app.createPost()"
                                ${!this.state.newPost.trim() || this.state.submittingPost ? 'disabled' : ''}
                            >
                                ${this.state.submittingPost ? 
                                    '<span class="spinner-border spinner-border-sm me-2"></span>Posting...' : 
                                    'Post'
                                }
                            </button>
                        </div>

                        <!-- Posts Feed -->
                        ${this.state.loading ? 
                            '<div class="text-center py-4"><div class="spinner-border" role="status"></div></div>' :
                            this.state.posts.length > 0 ? this.renderPosts() : this.renderNoPosts()
                        }

                        <!-- Pagination -->
                        ${totalPages > 1 ? this.renderPagination(totalPages) : ''}
                    </div>
                </div>
            </div>
        `;

        this.bindEvents();
    }

    renderPosts() {
        return this.state.posts.map(post => `
            <div class="list-group-item mb-3 border rounded">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <div class="mb-2">
                            ${post.content}
                            ${post.premium ? 
                                '<span class="badge bg-warning ms-2"><i class="fas fa-crown me-1"></i>Premium</span>' : 
                                ''
                            }
                        </div>
                        <small class="text-muted">
                            <i class="fas fa-user me-1"></i>
                            ${post.author || 'Anonymous'} • ${this.formatDate(post.timestamp)}
                        </small>
                    </div>
                </div>
                
                <div class="d-flex align-items-center mt-3">
                    <button 
                        class="btn btn-outline-primary btn-sm me-3" 
                        onclick="app.likePost(${post.id})"
                    >
                        <i class="fas fa-thumbs-up me-1"></i>
                        Like <span class="badge bg-secondary ms-1">${post.likes}</span>
                    </button>
                    <button 
                        class="btn btn-outline-secondary btn-sm" 
                        onclick="app.toggleComments(${post.id})"
                    >
                        <i class="fas fa-comment me-1"></i>
                        Comments (${post.comments ? post.comments.length : 0})
                    </button>
                </div>

                <!-- Comments Section -->
                ${this.state.selectedPost === post.id ? this.renderComments(post) : ''}
            </div>
        `).join('');
    }

    renderComments(post) {
        return `
            <div class="mt-3 border-top pt-3">
                <!-- Add Comment Form -->
                <div class="mb-3">
                    <input 
                        type="text" 
                        class="form-control mb-2" 
                        placeholder="Add a comment..."
                        value="${this.state.newComment}"
                        ${this.state.submittingComment ? 'disabled' : ''}
                        onchange="app.updateNewComment(this.value)"
                    />
                    <button 
                        class="btn btn-outline-secondary btn-sm" 
                        onclick="app.addComment(${post.id})"
                        ${!this.state.newComment.trim() || this.state.submittingComment ? 'disabled' : ''}
                    >
                        ${this.state.submittingComment ? 
                            '<span class="spinner-border spinner-border-sm me-1"></span>Adding...' : 
                            'Comment'
                        }
                    </button>
                </div>
                
                <!-- Comments List -->
                ${post.comments && post.comments.length > 0 ? 
                    post.comments.map(comment => `
                        <div class="border-bottom py-2">
                            <div>${comment.content}</div>
                            <small class="text-muted">
                                <i class="fas fa-user me-1"></i>
                                ${comment.author || 'Anonymous'} • ${this.formatDate(comment.timestamp)}
                            </small>
                        </div>
                    `).join('') :
                    '<p class="text-muted">No comments yet. Be the first to comment!</p>'
                }
            </div>
        `;
    }

    renderNoPosts() {
        return `
            <div class="alert alert-info">
                <i class="fas fa-info-circle me-2"></i>
                No posts yet. Start the conversation by sharing your thoughts!
            </div>
        `;
    }

    renderPagination(totalPages) {
        return `
            <div class="d-flex justify-content-center mt-4">
                <nav>
                    <ul class="pagination">
                        <li class="page-item ${this.state.page === 1 ? 'disabled' : ''}">
                            <a class="page-link" href="#" onclick="app.changePage(${this.state.page - 1})">Previous</a>
                        </li>
                        <li class="page-item active">
                            <span class="page-link">${this.state.page}</span>
                        </li>
                        <li class="page-item ${this.state.page === totalPages ? 'disabled' : ''}">
                            <a class="page-link" href="#" onclick="app.changePage(${this.state.page + 1})">Next</a>
                        </li>
                    </ul>
                </nav>
            </div>
        `;
    }

    bindEvents() {
        // Event listeners are handled via inline onclick attributes for simplicity
    }

    updateNewPost(value) {
        this.state.newPost = value;
    }

    updateNewComment(value) {
        this.state.newComment = value;
    }

    async changePage(page) {
        if (page < 1 || page > Math.ceil(this.state.total / 10)) return;
        this.state.page = page;
        await this.loadPosts();
    }
}

// Initialize the app when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new CommunityApp();
});