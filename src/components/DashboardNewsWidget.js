import React, { useState, useEffect } from 'react';
import { Card, Badge, Row, Col, ProgressBar } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import './DashboardNewsWidget.css'; // Custom CSS

function DashboardNewsWidget() {
  const [narratives, setNarratives] = useState([]);
  const [breakingNews, setBreakingNews] = useState([]);
  const [lastUpdated, setLastUpdated] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardNews();
    const interval = setInterval(loadDashboardNews, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []);

  async function loadDashboardNews() {
    try {
      const response = await fetch('/api/dashboard/narratives');
      const data = await response.json();
      
      setNarratives(data.top_narratives || []);
      setBreakingNews(data.breaking_news || []);
      setLastUpdated(new Date(data.last_updated).toLocaleTimeString());
      setLoading(false);
    } catch (error) {
      console.error('Error loading dashboard news:', error);
      setLoading(false);
    }
  }

  function getHeatColor(score) {
    if (score > 8) return 'danger';
    if (score > 5) return 'warning';
    if (score > 2) return 'info';
    return 'secondary';
  }

  function getImpactIcon(level) {
    switch (level) {
      case 'high': return '🔥';
      case 'medium': return '⚡';
      case 'low': return '📊';
      default: return '📄';
    }
  }

  function getSentimentIcon(score) {
    if (score > 0.3) return '📈';
    if (score < -0.3) return '📉';
    return '➡️';
  }

  if (loading) {
    return (
      <div className="dashboard-news-widget">
        <Card className="mb-4">
          <Card.Header>
            <h6 className="mb-0">🔥 Top News Narratives</h6>
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
    <div className="dashboard-news-widget">
      {/* Top Narratives Section */}
      <Card className="mb-4 news-narratives-card">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <h6 className="mb-0">🔥 Top News Narratives</h6>
          <small className="text-muted">Updated: {lastUpdated}</small>
        </Card.Header>
        <Card.Body className="p-2">
          {narratives.length > 0 ? (
            narratives.map((narrative, index) => (
              <div key={narrative.id} className={`narrative-item ${index === 0 ? 'featured' : ''}`}>
                <Row className="align-items-center">
                  <Col xs={1} className="text-center">
                    <div className="priority-indicator">
                      {getImpactIcon(narrative.impact_level)}
                    </div>
                  </Col>
                  <Col xs={7}>
                    <div className="narrative-content">
                      <div className="narrative-title">
                        <span className="text-decoration-none">
                          {narrative.title}
                        </span>
                      </div>
                      <div className="narrative-description">
                        {narrative.description}
                      </div>
                      <div className="narrative-meta">
                        <Badge bg={getHeatColor(narrative.heat_score)} size="sm" className="me-1">
                          Heat: {narrative.heat_score.toFixed(1)}
                        </Badge>
                        <Badge bg="outline-secondary" size="sm">
                          {narrative.cross_source_count} sources
                        </Badge>
                      </div>
                    </div>
                  </Col>
                  <Col xs={4}>
                    <div className="affected-assets">
                      {narrative.asset_impacts && narrative.asset_impacts.slice(0, 3).map(impact => (
                        <Badge 
                          key={impact.asset} 
                          bg="success" 
                          className="asset-badge"
                          title={`Impact: ${(impact.impact_score * 100).toFixed(0)}%`}
                        >
                          {impact.asset}
                        </Badge>
                      ))}
                      {narrative.asset_impacts && narrative.asset_impacts.length > 3 && (
                        <Badge bg="secondary" className="asset-badge">
                          +{narrative.asset_impacts.length - 3}
                        </Badge>
                      )}
                    </div>
                    <ProgressBar 
                      variant={getHeatColor(narrative.heat_score)}
                      now={Math.min(narrative.priority_score * 50, 100)}
                      size="sm"
                      className="confidence-bar"
                    />
                  </Col>
                </Row>
              </div>
            ))
          ) : (
            <div className="text-center text-muted py-3">
              No major narratives detected
            </div>
          )}
        </Card.Body>
      </Card>

      {/* Breaking News Ticker */}
      {breakingNews.length > 0 && (
        <Card className="breaking-news-card">
          <Card.Header className="bg-danger text-white">
            <h6 className="mb-0">⚡ Breaking News</h6>
          </Card.Header>
          <Card.Body className="p-0">
            <div className="news-ticker-container">
              <div className="news-ticker">
                {breakingNews.map((news, index) => (
                  <div key={index} className="news-item">
                    <span className="sentiment-icon">{getSentimentIcon(news.sentiment_score)}</span>
                    <span className="source-badge">{news.source}</span>
                    <a href={news.link} target="_blank" rel="noopener noreferrer" className="news-link">
                      {news.title}
                    </a>
                  </div>
                ))}
              </div>
            </div>
          </Card.Body>
        </Card>
      )}
    </div>
  );
}

export default DashboardNewsWidget;