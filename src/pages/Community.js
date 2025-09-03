import React, { useState, useEffect } from 'react';
import { Container, Card, Form, Button, ListGroup, Badge, Alert, Pagination, Spinner } from 'react-bootstrap';
import api from '../services/api';
import socket from '../services/socket';

function Community() {
  const [posts, setPosts] = useState([]);
  const [newPost, setNewPost] = useState('');
  const [selectedPost, setSelectedPost] = useState(null);
  const [newComment, setNewComment] = useState('');
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [submittingPost, setSubmittingPost] = useState(false);
  const [submittingComment, setSubmittingComment] = useState(false);

  useEffect(() => {
    loadPosts();
    
    // Set up Socket.IO event listeners
    socket.on('new_post', (post) => {
      setPosts(prev => [post, ...prev]);
    });
    
    socket.on('new_comment', (data) => {
      setPosts(prev => prev.map(p => 
        p.id === data.post_id 
          ? { ...p, comments: [...p.comments, data.comment] } 
          : p
      ));
    });
    
    socket.on('like_update', (data) => {
      setPosts(prev => prev.map(p => 
        p.id === data.post_id 
          ? { ...p, likes: data.likes } 
          : p
      ));
    });

    // Cleanup listeners on unmount
    return () => {
      socket.off('new_post');
      socket.off('new_comment');
      socket.off('like_update');
    };
  }, [page]);

  async function loadPosts() {
    setLoading(true);
    try {
      const response = await api.get('/api/community/posts', { 
        params: { page, per_page: 10 } 
      });
      setPosts(response.data.posts || []);
      setTotal(response.data.total || 0);
    } catch (error) {
      console.error('Error loading posts:', error);
    } finally {
      setLoading(false);
    }
  }

  async function createPost() {
    if (!newPost.trim()) return;
    
    setSubmittingPost(true);
    try {
      await api.post('/api/community/post', { 
        content: newPost, 
        premium: false 
      });
      setNewPost('');
    } catch (error) {
      console.error('Error creating post:', error);
    } finally {
      setSubmittingPost(false);
    }
  }

  async function addComment(postId) {
    if (!newComment.trim()) return;
    
    setSubmittingComment(true);
    try {
      await api.post('/api/community/comment', { 
        post_id: postId, 
        content: newComment 
      });
      setNewComment('');
    } catch (error) {
      console.error('Error adding comment:', error);
    } finally {
      setSubmittingComment(false);
    }
  }

  async function likePost(postId) {
    try {
      await api.post('/api/community/like', { 
        post_id: postId 
      });
    } catch (error) {
      console.error('Error liking post:', error);
    }
  }

  function toggleComments(postId) {
    setSelectedPost(selectedPost === postId ? null : postId);
    setNewComment(''); // Clear comment input when switching posts
  }

  const totalPages = Math.ceil(total / 10);

  return (
    <Container fluid className="p-4">
      <Card className="mb-4">
        <Card.Header as="h5">
          <i className="fas fa-users me-2"></i>
          Community
        </Card.Header>
        <Card.Body>
          {/* New Post Form */}
          <div className="mb-4">
            <h6>Share your thoughts</h6>
            <Form.Control
              as="textarea"
              rows={3}
              value={newPost}
              onChange={(e) => setNewPost(e.target.value)}
              placeholder="Share a trading idea, market insight, or start a discussion..."
              className="mb-3"
              disabled={submittingPost}
            />
            <Button 
              variant="primary" 
              onClick={createPost}
              disabled={!newPost.trim() || submittingPost}
            >
              {submittingPost ? (
                <>
                  <Spinner size="sm" className="me-2" />
                  Posting...
                </>
              ) : (
                'Post'
              )}
            </Button>
          </div>

          {/* Posts Feed */}
          {loading ? (
            <div className="text-center py-4">
              <Spinner animation="border" role="status">
                <span className="visually-hidden">Loading...</span>
              </Spinner>
            </div>
          ) : posts.length > 0 ? (
            <ListGroup>
              {posts.map(post => (
                <ListGroup.Item key={post.id} className="mb-3">
                  <div className="d-flex justify-content-between align-items-start">
                    <div className="flex-grow-1">
                      <div className="mb-2">
                        {post.content}
                        {post.premium && (
                          <Badge bg="warning" className="ms-2">
                            <i className="fas fa-crown me-1"></i>
                            Premium
                          </Badge>
                        )}
                      </div>
                      <small className="text-muted">
                        <i className="fas fa-user me-1"></i>
                        {post.author || 'Anonymous'} • {new Date(post.timestamp).toLocaleString()}
                      </small>
                    </div>
                  </div>
                  
                  <div className="d-flex align-items-center mt-3">
                    <Button 
                      variant="outline-primary" 
                      size="sm" 
                      onClick={() => likePost(post.id)} 
                      className="me-3"
                    >
                      <i className="fas fa-thumbs-up me-1"></i>
                      Like <Badge bg="secondary" className="ms-1">{post.likes}</Badge>
                    </Button>
                    <Button 
                      variant="outline-secondary" 
                      size="sm" 
                      onClick={() => toggleComments(post.id)}
                    >
                      <i className="fas fa-comment me-1"></i>
                      Comments ({post.comments?.length || 0})
                    </Button>
                  </div>

                  {/* Comments Section */}
                  {selectedPost === post.id && (
                    <div className="mt-3 border-top pt-3">
                      {/* Add Comment Form */}
                      <div className="mb-3">
                        <Form.Control
                          placeholder="Add a comment..."
                          value={newComment}
                          onChange={(e) => setNewComment(e.target.value)}
                          className="mb-2"
                          disabled={submittingComment}
                        />
                        <Button 
                          variant="outline-secondary" 
                          size="sm" 
                          onClick={() => addComment(post.id)}
                          disabled={!newComment.trim() || submittingComment}
                        >
                          {submittingComment ? (
                            <>
                              <Spinner size="sm" className="me-1" />
                              Adding...
                            </>
                          ) : (
                            'Comment'
                          )}
                        </Button>
                      </div>
                      
                      {/* Comments List */}
                      {post.comments && post.comments.length > 0 ? (
                        <ListGroup variant="flush">
                          {post.comments.map(comment => (
                            <ListGroup.Item key={comment.id} className="px-0 py-2">
                              <div>{comment.content}</div>
                              <small className="text-muted">
                                <i className="fas fa-user me-1"></i>
                                {comment.author || 'Anonymous'} • {new Date(comment.timestamp).toLocaleString()}
                              </small>
                            </ListGroup.Item>
                          ))}
                        </ListGroup>
                      ) : (
                        <p className="text-muted">No comments yet. Be the first to comment!</p>
                      )}
                    </div>
                  )}
                </ListGroup.Item>
              ))}
            </ListGroup>
          ) : (
            <Alert variant="info">
              <i className="fas fa-info-circle me-2"></i>
              No posts yet. Start the conversation by sharing your thoughts!
            </Alert>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="d-flex justify-content-center mt-4">
              <Pagination>
                <Pagination.Prev 
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                />
                <Pagination.Item active>{page}</Pagination.Item>
                <Pagination.Next 
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                />
              </Pagination>
            </div>
          )}
        </Card.Body>
      </Card>
    </Container>
  );
}

export default Community;