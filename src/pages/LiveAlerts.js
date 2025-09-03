import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Tabs, Tab, Card, Table, Badge, ProgressBar, Alert, Spinner, Form, Button, Modal } from 'react-bootstrap';
import { Bar, Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend } from 'chart.js';
import api from '../services/api';
import socket from '../services/socket';
import './LiveAlerts.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend);

// Helper functions
function getSignalBadgeVariant(signalType) {
  switch (signalType?.toLowerCase()) {
    case 'buy':
    case 'bullish':
      return 'success';
    case 'sell':
    case 'bearish':
      return 'danger';
    case 'hold':
    case 'neutral':
      return 'warning';
    default:
      return 'secondary';
  }
}

function getStatusBadgeVariant(status) {
  switch (status?.toLowerCase()) {
    case 'open': return 'primary';
    case 'closed': return 'success';
    case 'stopped': return 'danger';
    default: return 'secondary';
  }
}

function formatPnL(value) {
  if (typeof value !== 'number') return '0.00';
  return Math.abs(value).toFixed(2);
}

function formatTime(timestamp) {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: false 
    });
  } catch (e) {
    return 'Invalid time';
  }
}

function LiveAlerts() {
  const [alerts, setAlerts] = useState([]);
  const [trades, setTrades] = useState([]);
  const [analytics, setAnalytics] = useState({ 
    total_trades: 0, 
    winning_trades: 0, 
    win_rate: 0, 
    avg_confidence: 0,
    total_pnl: 0,
    open_trades: 0,
    weekly_pnl: 0,
    weekly_trades: 0,
    confidence_dist: [0, 0, 0, 0],
    cumulative_pnl: { dates: [], values: [] },
    sector_breakdown: { Crypto: 0, Equities: 0 },
    avg_ttp_hours: 0
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Modal state for alert-trade linking
  const [showModal, setShowModal] = useState(false);
  const [selectedTrades, setSelectedTrades] = useState([]);
  const [selectedAlertId, setSelectedAlertId] = useState(null);
  
  // Trading mode state
  const [currentMode, setCurrentMode] = useState('Paper');
  
  // Filter states
  const [filters, setFilters] = useState({
    priority: '',
    asset: '',
    type: ''
  });
  const [filteredAlerts, setFilteredAlerts] = useState([]);

  useEffect(() => {
    loadData();
    
    // Listen for new alerts via Socket.IO
    socket.on('new_alert', (signal) => {
      setAlerts(prev => [signal, ...prev.slice(0, 19)]);
      // Refresh trades and analytics when new alerts come in
      loadData();
    });
    
    // Refresh data every 30 seconds
    const interval = setInterval(loadData, 30000);
    
    return () => {
      socket.off('new_alert');
      clearInterval(interval);
    };
  }, []);
  
  // Apply filters when alerts or filters change
  useEffect(() => {
    applyFilters();
  }, [alerts, filters]);

  async function loadData() {
    try {
      setError(null);
      const [alertRes, tradeRes, analyticsRes, enhancedAnalyticsRes] = await Promise.all([
        api.get('/api/alerts/live'),
        api.get('/api/trades/paper'),
        api.get('/api/analytics/paper'),
        api.get('/api/analytics')
      ]);
      
      setAlerts(alertRes.data || []);
      setTrades(tradeRes.data || []);
      // Merge existing analytics with enhanced analytics
      setAnalytics({
        ...(analyticsRes.data || {}),
        ...(enhancedAnalyticsRes.data || {})
      });
      setLoading(false);
    } catch (e) {
      console.error('Error loading live alerts data:', e);
      setError('Failed to load data. Please try again.');
      setLoading(false);
    }
  }

  // Handle alert click to show linked trades
  async function handleAlertClick(alertId) {
    try {
      const res = await api.get(`/api/trades/paper?linked_alert=${alertId}`);
      setSelectedTrades(res.data);
      setSelectedAlertId(alertId);
      setShowModal(true);
    } catch (e) {
      console.error('Error fetching linked trades:', e);
    }
  }

  // Handle mode switch
  async function toggleMode() {
    const newMode = currentMode === 'Paper' ? 'Live' : 'Paper';
    try {
      const res = await api.post('/api/mode/switch', { mode: newMode });
      setCurrentMode(res.data.mode);
    } catch (e) {
      alert('Live mode is locked for now');
    }
  }
  
  function applyFilters() {
    let filtered = [...alerts];
    
    // Filter by priority
    if (filters.priority) {
      filtered = filtered.filter(alert => {
        const priority = alert.priority || calculateAlertPriority(alert);
        return priority.toString() === filters.priority;
      });
    }
    
    // Filter by asset
    if (filters.asset) {
      filtered = filtered.filter(alert => 
        alert.symbol && alert.symbol.toLowerCase().includes(filters.asset.toLowerCase())
      );
    }
    
    // Filter by type
    if (filters.type) {
      filtered = filtered.filter(alert => 
        alert.alert_type === filters.type
      );
    }
    
    setFilteredAlerts(filtered);
  }
  
  function calculateAlertPriority(alert) {
    // Calculate priority based on confidence and alert type
    if (alert.confidence > 0.8) return 3; // High
    if (alert.confidence > 0.6) return 2; // Medium
    return 1; // Low
  }
  
  function handleFilterChange(filterType, value) {
    setFilters(prev => ({
      ...prev,
      [filterType]: value
    }));
  }
  
  function clearFilters() {
    setFilters({ priority: '', asset: '', type: '' });
  }

  const formatTime = (timestamp) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch (e) {
      return 'Invalid Date';
    }
  };

  const formatPnL = (pnl) => {
    if (typeof pnl !== 'number') return '0.00';
    return pnl.toFixed(2);
  };

  const getConfidenceVariant = (confidence) => {
    if (confidence > 0.7) return 'success';
    if (confidence > 0.5) return 'warning';
    return 'danger';
  };

  const getSignalBadgeVariant = (signalType) => {
    return signalType === 'BUY' || signalType === 'LONG' ? 'success' : 'danger';
  };

  const getStatusBadgeVariant = (status) => {
    return status === 'OPEN' ? 'primary' : 'secondary';
  };
  
  const getPriorityBadgeVariant = (priority) => {
    switch(priority) {
      case 3: return 'danger';  // High
      case 2: return 'warning'; // Medium
      case 1: return 'secondary'; // Low
      default: return 'secondary';
    }
  };
  
  const getPriorityLabel = (priority) => {
    switch(priority) {
      case 3: return 'High';
      case 2: return 'Medium';
      case 1: return 'Low';
      default: return 'Unknown';
    }
  };

  if (loading) {
    return (
      <Container fluid className="p-4 text-center">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
        <p>Loading live alerts data...</p>
      </Container>
    );
  }

  return (
    <Container fluid className="p-4">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h2>🔔 Live Trading Alerts</h2>
        <small className="text-muted">Last updated: {formatTime(new Date())}</small>
      </div>

      {error && (
        <Alert variant="danger" dismissible onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {/* Alert Filters Section */}
      <Card className="mb-4">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">Filter Alerts</h5>
          <Badge bg="info">{filteredAlerts.length} / {alerts.length} Alerts</Badge>
        </Card.Header>
        <Card.Body>
          <Row className="g-3">
            <Col md={3}>
              <Form.Label>Priority</Form.Label>
              <Form.Select 
                value={filters.priority} 
                onChange={(e) => handleFilterChange('priority', e.target.value)}
              >
                <option value="">All Priorities</option>
                <option value="3">High</option>
                <option value="2">Medium</option>
                <option value="1">Low</option>
              </Form.Select>
            </Col>
            <Col md={3}>
              <Form.Label>Asset</Form.Label>
              <Form.Control 
                type="text" 
                placeholder="e.g., BTC, ETH" 
                value={filters.asset}
                onChange={(e) => handleFilterChange('asset', e.target.value)}
              />
            </Col>
            <Col md={3}>
              <Form.Label>Alert Type</Form.Label>
              <Form.Select 
                value={filters.type} 
                onChange={(e) => handleFilterChange('type', e.target.value)}
              >
                <option value="">All Types</option>
                <option value="TRADING_SIGNAL">Trading Signal</option>
                <option value="CORRELATION_BREAK">Correlation Break</option>
                <option value="INSTITUTIONAL_FLOW">Institutional Flow</option>
                <option value="SENTIMENT_FLOW">Sentiment Flow</option>
                <option value="ML_ANOMALY">ML Anomaly</option>
                <option value="PORTFOLIO_REBALANCING">Portfolio Rebalancing</option>
              </Form.Select>
            </Col>
            <Col md={3} className="d-flex align-items-end gap-2">
              <Button variant="primary" onClick={loadData} className="flex-fill">
                <i className="bi bi-arrow-clockwise me-2"></i>Refresh
              </Button>
              <Button variant="outline-secondary" onClick={clearFilters}>
                <i className="bi bi-x-circle me-1"></i>Clear
              </Button>
            </Col>
          </Row>
        </Card.Body>
      </Card>

      <Tabs defaultActiveKey="alerts" className="mb-4">
        <Tab eventKey="alerts" title={`🔔 Live Alerts (${filteredAlerts.length})`}>
          <Card>
            <Card.Header>Recent Trading Signals</Card.Header>
            <Card.Body>
              {filteredAlerts.length === 0 ? (
                <Alert variant="info">
                  {alerts.length === 0 
                    ? "No recent signals. The system is monitoring markets for opportunities."
                    : "No alerts found matching your criteria. Try adjusting the filters."
                  }
                </Alert>
              ) : (
                <Table striped hover responsive>
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Type</th>
                      <th>Priority</th>
                      <th>Confidence</th>
                      <th>Reason</th>
                      <th>Time</th>
                      <th>Timeframe</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredAlerts.map((alert, index) => (
                      <tr key={alert.id || index} onClick={() => handleAlertClick(alert.id)} style={{cursor: 'pointer'}} title="Click to view linked trades">
                        <td>
                          <strong>{alert.symbol}</strong>
                        </td>
                        <td>
                          <Badge bg={getSignalBadgeVariant(alert.signal_type)}>
                            {alert.signal_type}
                          </Badge>
                        </td>
                        <td>
                          <Badge bg={getPriorityBadgeVariant(alert.priority || calculateAlertPriority(alert))}>
                            {getPriorityLabel(alert.priority || calculateAlertPriority(alert))}
                          </Badge>
                        </td>
                        <td>
                          <ProgressBar 
                            now={alert.confidence * 100} 
                            variant={getConfidenceVariant(alert.confidence)}
                            label={`${(alert.confidence * 100).toFixed(0)}%`}
                            style={{ minWidth: '120px' }}
                          />
                        </td>
                        <td>
                          <small>
                            {Array.isArray(alert.reason) 
                              ? alert.reason.join(', ')
                              : alert.reason || 'No reason provided'
                            }
                          </small>
                        </td>
                        <td>
                          <small>{formatTime(alert.created_at)}</small>
                        </td>
                        <td>
                          <Badge variant="outline-primary">{alert.timeframe || 'N/A'}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
        </Tab>

        <Tab eventKey="trades" title={`📈 Paper Trades (${trades.length})`}>
          <Card>
            <Card.Header>Virtual Trading Portfolio</Card.Header>
            <Card.Body>
              {trades.length === 0 ? (
                <Alert variant="info">No paper trades yet. Trades will appear here when signals trigger.</Alert>
              ) : (
                <Table striped hover responsive>
                  <thead>
                    <tr>
                      <th>Trade ID</th>
                      <th>Symbol</th>
                      <th>Side</th>
                      <th>Entry Price</th>
                      <th>Size</th>
                      <th>Status</th>
                      <th>P&L</th>
                      <th>Entry Time</th>
                      <th>Linked Alert</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.map((trade, index) => (
                      <tr key={trade.trade_id || index}>
                        <td>
                          <small><code>{trade.trade_id}</code></small>
                        </td>
                        <td>
                          <strong>{trade.symbol}</strong>
                        </td>
                        <td>
                          <Badge bg={getSignalBadgeVariant(trade.side)}>
                            {trade.side}
                          </Badge>
                        </td>
                        <td>${trade.entry_price?.toFixed(2) || '0.00'}</td>
                        <td>{trade.size?.toFixed(4) || '0.0000'}</td>
                        <td>
                          <Badge bg={getStatusBadgeVariant(trade.status)}>
                            {trade.status}
                          </Badge>
                        </td>
                        <td>
                          <span style={{
                            color: trade.pnl > 0 ? 'green' : trade.pnl < 0 ? 'red' : 'black',
                            fontWeight: 'bold'
                          }}>
                            ${formatPnL(trade.pnl)}
                          </span>
                        </td>
                        <td>
                          <small>{formatTime(trade.entry_time)}</small>
                        </td>
                        <td>
                          <small><code>{trade.linked_alert || 'N/A'}</code></small>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
        </Tab>

        <Tab eventKey="analytics" title="📊 Analytics">
          <Card>
            <Card.Header className="d-flex justify-content-between align-items-center">
              <span>Advanced Analytics Dashboard</span>
              <Button variant="outline-primary" onClick={toggleMode} size="sm">
                Switch to {currentMode === 'Paper' ? 'Live (Locked)' : 'Paper'}
              </Button>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={6}>
                  <h6>Confidence Distribution</h6>
                  <Bar 
                    data={{
                      labels: ['0-50%', '50-70%', '70-90%', '90-100%'],
                      datasets: [{
                        label: 'Alerts',
                        data: analytics.confidence_dist || [0, 0, 0, 0],
                        backgroundColor: 'rgba(75,192,192,0.6)',
                        borderColor: 'rgba(75,192,192,1)',
                        borderWidth: 1
                      }]
                    }} 
                    options={{ 
                      responsive: true,
                      plugins: {
                        legend: {
                          display: false
                        }
                      },
                      scales: {
                        y: {
                          beginAtZero: true
                        }
                      }
                    }} 
                  />
                </Col>
                <Col md={6}>
                  <h6>Cumulative P&L</h6>
                  {analytics.cumulative_pnl && analytics.cumulative_pnl.dates.length > 0 ? (
                    <Line 
                      data={{
                        labels: analytics.cumulative_pnl.dates.map((date, index) => `Day ${index + 1}`),
                        datasets: [{
                          label: 'P&L',
                          data: analytics.cumulative_pnl.values,
                          borderColor: 'rgb(75,192,192)',
                          backgroundColor: 'rgba(75,192,192,0.2)',
                          tension: 0.1,
                          fill: true
                        }]
                      }} 
                      options={{ 
                        responsive: true,
                        plugins: {
                          legend: {
                            display: false
                          }
                        },
                        scales: {
                          y: {
                            beginAtZero: true
                          }
                        }
                      }} 
                    />
                  ) : (
                    <Alert variant="info">No P&L data available yet</Alert>
                  )}
                </Col>
              </Row>
              
              <div className="mt-4">
                <Row>
                  <Col md={3}>
                    <div className="text-center">
                      <h6>Win Rate</h6>
                      <h4 className={analytics.win_rate >= 60 ? 'text-success' : 'text-warning'}>
                        {analytics.win_rate || 0}%
                      </h4>
                    </div>
                  </Col>
                  <Col md={3}>
                    <div className="text-center">
                      <h6>Avg Confidence</h6>
                      <h4>{((analytics.avg_confidence || 0) * 100).toFixed(1)}%</h4>
                    </div>
                  </Col>
                  <Col md={3}>
                    <div className="text-center">
                      <h6>Sector Mix</h6>
                      <small>Crypto: {analytics.sector_breakdown?.Crypto || 0}%</small><br/>
                      <small>Equities: {analytics.sector_breakdown?.Equities || 0}%</small>
                    </div>
                  </Col>
                  <Col md={3}>
                    <div className="text-center">
                      <h6>Avg Time to Profit</h6>
                      <h4>{analytics.avg_ttp_hours || 0} hrs</h4>
                    </div>
                  </Col>
                </Row>
              </div>
              
              <hr className="my-4" />
              
              {/* Original Analytics Section */}
              <div className="row">
                <div className="col-md-6">
                  <h5>Overall Performance</h5>
                  <ul className="list-unstyled">
                    <li><strong>Total Trades:</strong> {analytics.total_trades}</li>
                    <li><strong>Winning Trades:</strong> {analytics.winning_trades}</li>
                    <li><strong>Losing Trades:</strong> {analytics.total_trades - analytics.winning_trades}</li>
                    <li>
                      <strong>Win Rate:</strong> 
                      <span className={`ms-2 ${analytics.win_rate >= 60 ? 'text-success' : analytics.win_rate >= 40 ? 'text-warning' : 'text-danger'}`}>
                        {analytics.win_rate}%
                      </span>
                    </li>
                    <li><strong>Average Confidence:</strong> {(analytics.avg_confidence * 100).toFixed(1)}%</li>
                  </ul>
                </div>
                <div className="col-md-6">
                  <h5>Profit & Loss</h5>
                  <ul className="list-unstyled">
                    <li>
                      <strong>Total P&L:</strong> 
                      <span className={`ms-2 ${analytics.total_pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                        ${formatPnL(analytics.total_pnl)}
                      </span>
                    </li>
                    <li>
                      <strong>Weekly P&L:</strong> 
                      <span className={`ms-2 ${analytics.weekly_pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                        ${formatPnL(analytics.weekly_pnl)}
                      </span>
                    </li>
                    <li><strong>Weekly Trades:</strong> {analytics.weekly_trades}</li>
                    <li><strong>Open Trades:</strong> {analytics.open_trades}</li>
                  </ul>
                </div>
              </div>
              
              {analytics.total_trades > 0 && (
                <div className="mt-4">
                  <h6>Performance Indicators</h6>
                  <div className="progress-stacked">
                    <div className="progress" style={{ height: '20px' }}>
                      <div 
                        className="progress-bar bg-success" 
                        style={{ width: `${analytics.win_rate}%` }}
                        aria-label={`${analytics.win_rate}% win rate`}
                      >
                        Win: {analytics.win_rate}%
                      </div>
                      <div 
                        className="progress-bar bg-danger" 
                        style={{ width: `${100 - analytics.win_rate}%` }}
                        aria-label={`${100 - analytics.win_rate}% loss rate`}
                      >
                        Loss: {(100 - analytics.win_rate).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </Card.Body>
          </Card>
        </Tab>
      </Tabs>

      {/* Modal for linked trades */}
      <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Trades for Alert {selectedAlertId}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedTrades.length === 0 ? (
            <Alert variant="info">No trades found for this alert.</Alert>
          ) : (
            <Table striped hover>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Entry</th>
                  <th>Status</th>
                  <th>P&L</th>
                  <th>Entry Time</th>
                </tr>
              </thead>
              <tbody>
                {selectedTrades.map(trade => (
                  <tr key={trade.trade_id}>
                    <td><strong>{trade.symbol}</strong></td>
                    <td>
                      <Badge bg={getSignalBadgeVariant(trade.side)}>
                        {trade.side}
                      </Badge>
                    </td>
                    <td>${trade.entry_price?.toFixed(2) || '0.00'}</td>
                    <td>
                      <Badge bg={getStatusBadgeVariant(trade.status)}>
                        {trade.status}
                      </Badge>
                    </td>
                    <td>
                      <span style={{
                        color: trade.pnl > 0 ? 'green' : trade.pnl < 0 ? 'red' : 'black',
                        fontWeight: 'bold'
                      }}>
                        ${formatPnL(trade.pnl)}
                      </span>
                    </td>
                    <td>
                      <small>{formatTime(trade.entry_time)}</small>
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Modal.Body>
      </Modal>
    </Container>
  );
}

export default LiveAlerts;