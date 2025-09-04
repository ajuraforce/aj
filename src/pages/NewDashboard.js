import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Table, Badge, Button, Alert, Spinner } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import api from '../services/api';
import socket from '../services/socket';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function NewDashboard() {
  const navigate = useNavigate();
  const [status, setStatus] = useState({});
  const [alerts, setAlerts] = useState({ total: 0, new: 0, recent: [] });
  const [trades, setTrades] = useState({ open: 0, labels: [], data: [], recent: [] });
  const [portfolio, setPortfolio] = useState({ pnl: 0, labels: [], data: [] });
  const [recentActivity, setRecentActivity] = useState([]);
  const [news, setNews] = useState({ narratives: [], breaking_news: [] });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAllData();
    
    socket.on('new_alert', (alert) => {
      setAlerts(prev => ({
        total: prev.total + 1,
        new: prev.new + 1,
        recent: [alert, ...prev.recent.slice(0,4)]
      }));
    });
    
    return () => socket.off('new_alert');
  }, []);

  async function loadAllData() {
    try {
      setLoading(true);
      
      const [
        statusRes, 
        alertsRes, 
        tradesRes, 
        portfolioRes, 
        communityRes,
        newsRes
      ] = await Promise.all([
        api.get('/api/status'),
        api.get('/api/alerts'),
        api.get('/api/trades'),
        api.get('/api/portfolio'),
        api.get('/api/community/posts?per_page=5'),
        api.get('/api/dashboard/narratives')
      ]);
      
      setStatus(statusRes.data);
      setAlerts(alertsRes.data);
      setTrades(tradesRes.data);
      setPortfolio(portfolioRes.data);
      setRecentActivity(communityRes.data.posts || []);
      setNews({
        narratives: newsRes.data.top_narratives || [],
        breaking_news: newsRes.data.breaking_news || []
      });
      
    } catch (e) {
      console.error('Error loading dashboard data:', e);
    } finally {
      setLoading(false);
    }
  }

  function getStatusColor(value) {
    if (value === 'Risk-On' || value > 80) return 'success';
    if (value === 'Risk-Off' || value < 20) return 'danger';
    return 'warning';
  }

  function formatTime(timestamp) {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return 'Invalid time';
    }
  }

  function getSentimentIcon(score) {
    if (score > 0.3) return '📈';
    if (score < -0.3) return '📉';
    return '➡️';
  }

  if (loading) {
    return (
      <Container fluid className="p-4 text-center">
        <Spinner animation="border" variant="primary" />
        <p className="mt-3">Loading real-time data...</p>
      </Container>
    );
  }

  return (
    <Container fluid className="p-4">
      {/* Header */}
      <Row className="mb-4">
        <Col>
          <h2>AJxAI Trading Platform</h2>
          <p className="text-muted">Real-time Market Intelligence & Trading Analytics</p>
        </Col>
      </Row>

      {/* Status Cards */}
      <Row className="g-4 mb-4">
        <Col xs={12} md={6} lg={3}>
          <Card className="text-center h-100">
            <Card.Body>
              <i className="bi bi-activity text-success fs-3 mb-2"></i>
              <Card.Title>Platform Status</Card.Title>
              <h4>
                <Badge bg="success">
                  {status.features_operational || 0}/15 Features
                </Badge>
              </h4>
              <small className="text-muted">
                Market: {status.market_regime || 'Unknown'}
              </small>
            </Card.Body>
          </Card>
        </Col>

        <Col xs={12} md={6} lg={3}>
          <Card className="text-center h-100">
            <Card.Body>
              <i className="bi bi-bell text-warning fs-3 mb-2"></i>
              <Card.Title>Live Alerts</Card.Title>
              <h4>{alerts.total || 0}</h4>
              <Badge bg="warning" className="me-1">{alerts.new || 0} New</Badge>
              <Line
                data={{ 
                  labels: ['1h', '2h', '3h', '4h', '5h'], 
                  datasets: [{ 
                    data: [2, 5, 3, 8, alerts.total], 
                    borderColor: '#ffc107', 
                    backgroundColor: 'rgba(255,193,7,0.2)', 
                    fill: true, 
                    pointRadius: 0 
                  }] 
                }}
                options={{ 
                  scales: { x: { display: false }, y: { display: false } }, 
                  plugins: { legend: { display: false } } 
                }}
                height={50}
              />
            </Card.Body>
          </Card>
        </Col>

        <Col xs={12} md={6} lg={3}>
          <Card className="text-center h-100">
            <Card.Body>
              <i className="bi bi-graph-up text-primary fs-3 mb-2"></i>
              <Card.Title>Open Trades</Card.Title>
              <h4>{trades.open || 0}</h4>
              <Line
                data={{ 
                  labels: trades.labels || ['1h', '2h', '3h', '4h', '5h'], 
                  datasets: [{ 
                    data: trades.data || [1, 2, 1, 3, trades.open], 
                    borderColor: '#0d6efd', 
                    backgroundColor: 'rgba(13,110,253,0.2)', 
                    fill: true, 
                    pointRadius: 0 
                  }] 
                }}
                options={{ 
                  scales: { x: { display: false }, y: { display: false } }, 
                  plugins: { legend: { display: false } } 
                }}
                height={50}
              />
            </Card.Body>
          </Card>
        </Col>

        <Col xs={12} md={6} lg={3}>
          <Card className="text-center h-100">
            <Card.Body>
              <i className="bi bi-currency-dollar text-info fs-3 mb-2"></i>
              <Card.Title>Total P&L</Card.Title>
              <h4 className={portfolio.pnl >= 0 ? 'text-success' : 'text-danger'}>
                ${portfolio.pnl?.toFixed(2) || '0.00'}
              </h4>
              <Line
                data={{ 
                  labels: portfolio.labels || ['1h', '2h', '3h', '4h', '5h'], 
                  datasets: [{ 
                    data: portfolio.data || [0, 50, -20, 100, portfolio.pnl], 
                    borderColor: portfolio.pnl >= 0 ? '#20c997' : '#dc3545', 
                    backgroundColor: portfolio.pnl >= 0 ? 'rgba(32,201,151,0.2)' : 'rgba(220,53,69,0.2)', 
                    fill: true, 
                    pointRadius: 0 
                  }] 
                }}
                options={{ 
                  scales: { x: { display: false }, y: { display: false } }, 
                  plugins: { legend: { display: false } } 
                }}
                height={50}
              />
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Breaking News Section - 30 Articles */}
      <Row className="mb-4">
        <Col xs={12}>
          <Card>
            <Card.Header className="d-flex justify-content-between align-items-center">
              <h5 className="mb-0">🔥 Breaking News ({news.breaking_news.length} Articles)</h5>
              <Button 
                variant="outline-primary" 
                size="sm"
                onClick={() => navigate('/all-news')}
              >
                View All Articles →
              </Button>
            </Card.Header>
            <Card.Body className="p-0">
              <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                <Table hover responsive className="mb-0">
                  <thead className="table-light sticky-top">
                    <tr>
                      <th width="10%">Sentiment</th>
                      <th width="15%">Source</th>
                      <th width="65%">Headline</th>
                      <th width="10%">Priority</th>
                    </tr>
                  </thead>
                  <tbody>
                    {news.breaking_news.length > 0 ? (
                      news.breaking_news.slice(0, 30).map((article, index) => (
                        <tr key={index}>
                          <td className="text-center">
                            {getSentimentIcon(article.sentiment_score || 0)}
                          </td>
                          <td>
                            <Badge bg="secondary" className="small">
                              {article.source || 'Unknown'}
                            </Badge>
                          </td>
                          <td>
                            <a 
                              href={article.link} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-decoration-none"
                            >
                              {article.title || 'No title'}
                            </a>
                          </td>
                          <td>
                            {article.urgency === 'high' && (
                              <Badge bg="danger">HIGH</Badge>
                            )}
                            {article.urgency === 'medium' && (
                              <Badge bg="warning">MED</Badge>
                            )}
                            {article.urgency === 'low' && (
                              <Badge bg="info">LOW</Badge>
                            )}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="4" className="text-center text-muted py-3">
                          <i className="bi bi-newspaper me-2"></i>
                          No breaking news available
                        </td>
                      </tr>
                    )}
                  </tbody>
                </Table>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Recent Activity */}
      <Row className="mb-4">
        <Col xs={12}>
          <Card>
            <Card.Header className="d-flex justify-content-between align-items-center">
              <h5 className="mb-0">📊 Recent Activity</h5>
              <Button 
                variant="outline-success" 
                size="sm"
                onClick={() => navigate('/live-alerts')}
              >
                View All Alerts →
              </Button>
            </Card.Header>
            <Card.Body className="p-0">
              <Table hover responsive className="mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Time</th>
                    <th>Type</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {recentActivity.length > 0 ? (
                    recentActivity.map((activity, i) => (
                      <tr key={i}>
                        <td>{formatTime(activity.timestamp)}</td>
                        <td>
                          <Badge bg={
                            activity.type === 'alert' ? 'danger' : 
                            activity.type === 'signal' ? 'success' : 'info'
                          }>
                            {activity.type.toUpperCase()}
                          </Badge>
                        </td>
                        <td>{activity.content}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="3" className="text-center text-muted py-3">
                        <i className="bi bi-clock me-2"></i>
                        No recent activity
                      </td>
                    </tr>
                  )}
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Platform Controls */}
      <Row>
        <Col xs={12}>
          <Card>
            <Card.Header>
              <h5 className="mb-0">🎛️ Platform Controls</h5>
            </Card.Header>
            <Card.Body className="text-center">
              <Button variant="success" className="me-2">
                <i className="bi bi-play-fill me-1"></i>
                Start Trading
              </Button>
              <Button variant="danger" className="me-2">
                <i className="bi bi-stop-fill me-1"></i>
                Stop Trading
              </Button>
              <Button variant="primary" className="me-2">
                <i className="bi bi-cloud-upload me-1"></i>
                Backup State
              </Button>
              <Button 
                variant="info" 
                onClick={() => navigate('/health')}
              >
                <i className="bi bi-heart-pulse me-1"></i>
                Health Monitor
              </Button>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default NewDashboard;