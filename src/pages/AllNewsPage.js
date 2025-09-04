import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Badge, Button, Alert, Spinner, Form } from 'react-bootstrap';

function AllNewsPage() {
  const [allNews, setAllNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [lastUpdated, setLastUpdated] = useState('');

  useEffect(() => {
    loadAllNews();
    const interval = setInterval(loadAllNews, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []);

  async function loadAllNews() {
    try {
      const response = await fetch('/api/news/all');
      const data = await response.json();
      
      setAllNews(data.articles || []);
      setLastUpdated(new Date().toLocaleTimeString());
      setLoading(false);
    } catch (error) {
      console.error('Error loading all news:', error);
      setLoading(false);
    }
  }

  function getSentimentIcon(score) {
    if (score > 0.3) return '📈';
    if (score < -0.3) return '📉';
    return '➡️';
  }

  function getSentimentBadge(score) {
    if (score > 0.3) return 'success';
    if (score < -0.3) return 'danger';
    return 'secondary';
  }

  function formatDate(dateString) {
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  }

  const filteredNews = allNews.filter(article => {
    const matchesFilter = filter === 'all' || 
                         (filter === 'high' && article.urgency === 'high') ||
                         (filter === 'positive' && article.sentiment_score > 0.3) ||
                         (filter === 'negative' && article.sentiment_score < -0.3);
    
    const matchesSearch = searchTerm === '' || 
                         article.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         article.source.toLowerCase().includes(searchTerm.toLowerCase());
    
    return matchesFilter && matchesSearch;
  });

  if (loading) {
    return (
      <Container className="mt-4">
        <div className="text-center">
          <Spinner animation="border" variant="primary" />
          <p className="mt-2">Loading all news articles...</p>
        </div>
      </Container>
    );
  }

  return (
    <Container className="mt-4">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h1>📰 All News Articles</h1>
          <p className="text-muted">Last updated: {lastUpdated}</p>
        </div>
        <Button variant="outline-primary" href="/">
          ← Back to Dashboard
        </Button>
      </div>

      {/* Filters */}
      <Card className="mb-4">
        <Card.Body>
          <Row>
            <Col md={6}>
              <Form.Group>
                <Form.Label>Filter by Type</Form.Label>
                <Form.Select value={filter} onChange={(e) => setFilter(e.target.value)}>
                  <option value="all">All Articles</option>
                  <option value="high">High Priority Only</option>
                  <option value="positive">Positive Sentiment</option>
                  <option value="negative">Negative Sentiment</option>
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={6}>
              <Form.Group>
                <Form.Label>Search Articles</Form.Label>
                <Form.Control
                  type="text"
                  placeholder="Search by title or source..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </Form.Group>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      {/* Stats */}
      <Alert variant="info" className="mb-4">
        Showing <strong>{filteredNews.length}</strong> of <strong>{allNews.length}</strong> total articles
      </Alert>

      {/* News Articles */}
      <Row>
        {filteredNews.length > 0 ? (
          filteredNews.map((article, index) => (
            <Col md={6} lg={4} key={index} className="mb-4">
              <Card className="h-100 news-article-card">
                <Card.Header className="d-flex justify-content-between align-items-start">
                  <div>
                    <Badge bg="primary" className="me-2">{article.source}</Badge>
                    {article.urgency === 'high' && (
                      <Badge bg="danger">HIGH PRIORITY</Badge>
                    )}
                  </div>
                  <div className="text-end">
                    <Badge bg={getSentimentBadge(article.sentiment_score)}>
                      {getSentimentIcon(article.sentiment_score)} 
                      {(article.sentiment_score * 100).toFixed(0)}%
                    </Badge>
                  </div>
                </Card.Header>
                
                <Card.Body>
                  <Card.Title className="fs-6">
                    <a 
                      href={article.url || article.link} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-decoration-none"
                    >
                      {article.title}
                    </a>
                  </Card.Title>
                  
                  <div className="small text-muted">
                    📅 {formatDate(article.published || article.timestamp)}
                  </div>
                  
                  {article.relevance_score && (
                    <div className="mt-2">
                      <div className="small">Relevance</div>
                      <div className="progress" style={{height: '4px'}}>
                        <div 
                          className="progress-bar" 
                          style={{width: `${article.relevance_score * 100}%`}}
                        ></div>
                      </div>
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          ))
        ) : (
          <Col>
            <Alert variant="warning">
              No articles found matching your filter criteria.
            </Alert>
          </Col>
        )}
      </Row>
    </Container>
  );
}

export default AllNewsPage;