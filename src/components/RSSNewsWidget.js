import React, { useState, useEffect } from 'react';
import { Card, Badge, Row, Col, ProgressBar, Button, Dropdown } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import './RSSNewsWidget.css';

function RSSNewsWidget() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [systemStatus, setSystemStatus] = useState('loading');
  const [analyzedCount, setAnalyzedCount] = useState(0);

  useEffect(() => {
    loadRSSArticles();
    loadSystemStatus();
    const interval = setInterval(() => {
      loadRSSArticles();
      loadSystemStatus();
    }, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [selectedCategory]);

  async function loadRSSArticles() {
    try {
      const params = new URLSearchParams({
        limit: '20'
      });
      if (selectedCategory !== 'all') {
        params.append('category', selectedCategory);
      }

      const response = await fetch(`/api/rss/articles?${params}`);
      const data = await response.json();
      
      setArticles(data.articles || []);
      setLastUpdated(new Date().toLocaleTimeString());
      setLoading(false);
    } catch (error) {
      console.error('Error loading RSS articles:', error);
      setLoading(false);
    }
  }

  async function loadSystemStatus() {
    try {
      const response = await fetch('/api/rss/health');
      const data = await response.json();
      setSystemStatus(data.status);
    } catch (error) {
      console.error('Error loading RSS system status:', error);
      setSystemStatus('error');
    }
  }

  async function triggerAnalysis() {
    try {
      setLoading(true);
      const response = await fetch('/api/rss/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ batch_size: 25 })
      });
      const data = await response.json();
      setAnalyzedCount(data.analyzed_count || 0);
      
      // Refresh articles after analysis
      setTimeout(loadRSSArticles, 2000);
    } catch (error) {
      console.error('Error triggering RSS analysis:', error);
      setLoading(false);
    }
  }

  function getSentimentColor(score) {
    if (score > 0.3) return 'success';
    if (score < -0.3) return 'danger';
    return 'warning';
  }

  function getSentimentIcon(score) {
    if (score > 0.3) return '📈';
    if (score < -0.3) return '📉';
    return '➡️';
  }

  function getRelevanceColor(score) {
    if (score > 0.8) return 'danger';
    if (score > 0.6) return 'warning';
    if (score > 0.4) return 'info';
    return 'secondary';
  }

  function getStatusIcon(status) {
    switch (status) {
      case 'online': return '🟢';
      case 'degraded': return '🟡';
      case 'offline': return '🔴';
      default: return '⚪';
    }
  }

  function formatTimestamp(timestamp) {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return 'Unknown';
    }
  }

  function truncateText(text, maxLength = 150) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  }

  if (loading && articles.length === 0) {
    return (
      <div className="rss-news-widget">
        <Card className="mb-4">
          <Card.Header>
            <h6 className="mb-0">📰 RSS News Analysis</h6>
          </Card.Header>
          <Card.Body className="text-center">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </Card.Body>
        </Card>
      </div>
    );
  }

  return (
    <div className="rss-news-widget">
      {/* RSS News Analysis Header */}
      <Card className="mb-4 rss-main-card">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <div className="d-flex align-items-center gap-2">
            <h6 className="mb-0">📰 RSS News Analysis</h6>
            <Badge bg={systemStatus === 'online' ? 'success' : systemStatus === 'degraded' ? 'warning' : 'danger'}>
              {getStatusIcon(systemStatus)} {systemStatus}
            </Badge>
          </div>
          <div className="d-flex align-items-center gap-2">
            <small className="text-muted">Updated: {lastUpdated}</small>
            <Button 
              variant="outline-primary" 
              size="sm" 
              onClick={triggerAnalysis}
              disabled={loading}
            >
              {loading ? (
                <span className="spinner-border spinner-border-sm me-1" />
              ) : (
                '🔄'
              )}
              Analyze
            </Button>
          </div>
        </Card.Header>

        {/* Category Filter */}
        <Card.Body className="pb-2">
          <div className="d-flex justify-content-between align-items-center mb-3">
            <Dropdown>
              <Dropdown.Toggle variant="outline-secondary" size="sm">
                📂 {selectedCategory === 'all' ? 'All Categories' : selectedCategory}
              </Dropdown.Toggle>
              <Dropdown.Menu>
                <Dropdown.Item onClick={() => setSelectedCategory('all')}>All Categories</Dropdown.Item>
                <Dropdown.Item onClick={() => setSelectedCategory('crypto')}>Crypto</Dropdown.Item>
                <Dropdown.Item onClick={() => setSelectedCategory('finance')}>Finance</Dropdown.Item>
                <Dropdown.Item onClick={() => setSelectedCategory('geopolitics')}>Geopolitics</Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown>
            
            <small className="text-muted">
              {articles.length} articles {analyzedCount > 0 && `(${analyzedCount} analyzed)`}
            </small>
          </div>

          {/* Articles List */}
          <div className="rss-articles-container">
            {articles.length === 0 ? (
              <div className="text-center text-muted py-4">
                <div>📭</div>
                <small>No articles available for this category</small>
              </div>
            ) : (
              articles.slice(0, 10).map((article, index) => (
                <div key={article.id || index} className="rss-article-item">
                  <div className="d-flex justify-content-between align-items-start mb-2">
                    <div className="flex-grow-1">
                      <div className="d-flex align-items-center gap-2 mb-1">
                        <Badge bg="light" text="dark" className="source-badge">
                          {article.source || 'Unknown'}
                        </Badge>
                        {article.category && (
                          <Badge bg="outline-secondary" className="category-badge">
                            {article.category}
                          </Badge>
                        )}
                        {article.sentiment_score !== undefined && (
                          <Badge bg={getSentimentColor(article.sentiment_score)}>
                            {getSentimentIcon(article.sentiment_score)} 
                            {(article.sentiment_score * 100).toFixed(0)}%
                          </Badge>
                        )}
                      </div>
                      
                      <a 
                        href={article.link} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="article-title-link"
                      >
                        <strong>{article.title}</strong>
                      </a>
                      
                      {article.summary && (
                        <p className="article-summary">
                          {truncateText(article.summary)}
                        </p>
                      )}
                      
                      {article.detected_assets && article.detected_assets.length > 0 && (
                        <div className="detected-assets mb-2">
                          <small className="text-muted">Assets: </small>
                          {article.detected_assets.slice(0, 5).map((asset, i) => (
                            <Badge key={i} bg="info" className="me-1 asset-tag">
                              {asset}
                            </Badge>
                          ))}
                          {article.detected_assets.length > 5 && (
                            <Badge bg="light" text="dark">
                              +{article.detected_assets.length - 5} more
                            </Badge>
                          )}
                        </div>
                      )}
                      
                      <div className="article-meta">
                        <small className="text-muted">
                          {formatTimestamp(article.published_date)}
                          {article.relevance_score && (
                            <>
                              {' • '}
                              <Badge bg={getRelevanceColor(article.relevance_score)} className="relevance-badge">
                                {(article.relevance_score * 100).toFixed(0)}% relevant
                              </Badge>
                            </>
                          )}
                        </small>
                      </div>
                    </div>
                  </div>
                  
                  {/* Article Analysis Progress */}
                  {article.relevance_score && (
                    <ProgressBar 
                      variant={getRelevanceColor(article.relevance_score)} 
                      now={article.relevance_score * 100} 
                      className="relevance-progress"
                      style={{ height: '3px' }}
                    />
                  )}
                </div>
              ))
            )}
          </div>
          
          {articles.length > 10 && (
            <div className="text-center mt-3">
              <small className="text-muted">
                Showing top 10 of {articles.length} articles
              </small>
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
}

export default RSSNewsWidget;