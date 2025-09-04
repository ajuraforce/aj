// src/pages/Dashboard.js
import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Badge, Table, Button } from 'react-bootstrap';
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
import DashboardNewsWidget from '../components/DashboardNewsWidget';

// Register Chart.js components
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

function Dashboard() {
  const navigate = useNavigate();
  const [status, setStatus] = useState({});
  const [alerts, setAlerts] = useState({ total: 0, new: 0, recent: [] });
  const [trades, setTrades] = useState({ open: 0, labels: [], data: [], recent: [] });
  const [portfolio, setPortfolio] = useState({ pnl: 0, labels: [], data: [] });
  const [recentActivity, setRecentActivity] = useState([]);

  useEffect(() => {
    loadData();
    socket.on('new_alert', (alert) => {
      setAlerts(prev => ({
        total: prev.total + 1,
        new: prev.new + 1,
        recent: [alert, ...prev.recent.slice(0,4)]
      }));
    });
    return () => socket.off('new_alert');
  }, []);

  async function loadData() {
    try {
      const [statusRes, alertsRes, tradesRes, portfolioRes, communityRes] = await Promise.all([
        api.get('/api/status'),
        api.get('/api/alerts'),
        api.get('/api/trades'),
        api.get('/api/portfolio'),
        api.get('/api/community/posts?per_page=5')
      ]);
      setStatus(statusRes.data);
      setAlerts(alertsRes.data);
      setTrades(tradesRes.data);
      setPortfolio(portfolioRes.data);
      setRecentActivity(communityRes.data.posts || []);
    } catch (e) {
      console.error('Error loading dashboard data:', e);
    }
  }

  return (
    <Container fluid className="p-4">
      <Row className="mb-4">
        <Col><h2>AjxAI Trading Dashboard</h2><p className="text-muted">AI-Powered Social Intelligence Platform</p></Col>
      </Row>

      <Row className="g-4 mb-4">
        {/* Platform Status */}
        <Col xs={12} md={6} lg={3}>
          <Card className="text-center">
            <Card.Body>
              <i className="bi bi-cpu text-primary fs-3 mb-2"></i>
              <Card.Title>Platform Status</Card.Title>
              <h4>{status.status || 'Loading...'}</h4>
              <Badge bg={status.status === 'Running' ? 'success' : 'danger'}>
                {status.status || '...'}
              </Badge>
              <div className="small text-muted">{status.last_backup}</div>
            </Card.Body>
          </Card>
        </Col>

        {/* Open Trades */}
        <Col xs={12} md={6} lg={3}>
          <Card className="text-center">
            <Card.Body>
              <i className="bi bi-graph-up text-success fs-3 mb-2"></i>
              <Card.Title>Open Trades</Card.Title>
              <h4>{trades.open}</h4>
              <Line
                data={{ labels: trades.labels, datasets: [{ data: trades.data, borderColor: '#198754', fill: false, pointRadius: 0 }] }}
                options={{ scales: { x: { display: false }, y: { display: false } }, plugins: { legend: { display: false } } }}
                height={50}
              />
            </Card.Body>
          </Card>
        </Col>

        {/* Recent Alerts */}
        <Col xs={12} md={6} lg={3}>
          <Card className="text-center">
            <Card.Body>
              <i className="bi bi-bell text-warning fs-3 mb-2"></i>
              <Card.Title>Recent Alerts</Card.Title>
              <h4>{alerts.total}</h4>
              <Badge bg="warning">+{alerts.new}</Badge>
            </Card.Body>
          </Card>
        </Col>

        {/* Total P&L */}
        <Col xs={12} md={6} lg={3}>
          <Card className="text-center">
            <Card.Body>
              <i className="bi bi-currency-dollar text-info fs-3 mb-2"></i>
              <Card.Title>Total P&L</Card.Title>
              <h4>${portfolio.pnl.toFixed(2)}</h4>
              <Line
                data={{ labels: portfolio.labels, datasets: [{ data: portfolio.data, borderColor: '#0d6efd', backgroundColor: 'rgba(13,110,253,0.2)', fill: true, pointRadius: 0 }] }}
                options={{ scales: { x: { display: false }, y: { display: false } }, plugins: { legend: { display: false } } }}
                height={50}
              />
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Recent Activity */}
      <Row className="mb-4">
        <Col xs={12}>
          <Card>
            <Card.Header>Recent Activity</Card.Header>
            <Card.Body className="p-0">
              <Table hover responsive className="mb-0">
                <thead className="table-light">
                  <tr><th>Time</th><th>Type</th><th>Details</th></tr>
                </thead>
                <tbody>
                  {recentActivity.length > 0 ? (
                    recentActivity.map((activity, i) => (
                      <tr key={i}>
                        <td>{new Date(activity.timestamp).toLocaleTimeString()}</td>
                        <td>
                          <Badge bg={activity.type === 'alert' ? 'danger' : activity.type === 'signal' ? 'success' : 'info'}>
                            {activity.type.toUpperCase()}
                          </Badge>
                        </td>
                        <td>{activity.content}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="3" className="text-center text-muted py-3">
                        <i className="bi bi-info-circle me-2"></i>
                        No recent activity
                      </td>
                    </tr>
                  )}
                </tbody>
              </Table>
            </Card.Body>
            <Card.Footer className="text-end">
              <Button variant="link" size="sm" onClick={() => navigate('/live-alerts')}>View All Alerts</Button>
            </Card.Footer>
          </Card>
        </Col>

      </Row>

      {/* News Intelligence Section */}
      <Row className="mb-4">
        <Col xs={12}>
          <DashboardNewsWidget />
        </Col>
      </Row>

      {/* Platform Controls */}
      <Card>
        <Card.Header>Platform Controls</Card.Header>
        <Card.Body className="text-center">
          <Button variant="success" className="me-2">Start</Button>
          <Button variant="danger" className="me-2">Stop</Button>
          <Button variant="primary" className="me-2">Backup</Button>
          <Button variant="info" onClick={() => navigate('/health')}>Health Monitor</Button>
        </Card.Body>
      </Card>
    </Container>
  );
}

export default Dashboard;