// src/pages/Trades.js
import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Table, Form, Button, Modal, Pagination, Spinner, Badge } from 'react-bootstrap';
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

function Trades() {
  const [trades, setTrades] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [perPage] = useState(20);
  const [statusFilter, setStatusFilter] = useState('open');  // Tabs control
  const [assetFilter, setAssetFilter] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedTrade, setSelectedTrade] = useState(null);  // For modal
  const [pnlData, setPnlData] = useState({ labels: [], data: [] });  // For chart

  useEffect(() => {
    loadTrades();
    socket.on('trade_update', (updatedTrade) => {
      setTrades(prev => prev.map(t => t.id === updatedTrade.id ? updatedTrade : t));
    });
    return () => socket.off('trade_update');
  }, [page, statusFilter, assetFilter, dateFrom, dateTo]);

  async function loadTrades() {
    setLoading(true);
    try {
      const params = { status: statusFilter, page, per_page: perPage };
      if (assetFilter) params.asset = assetFilter;
      if (dateFrom) params.date_from = dateFrom;
      if (dateTo) params.date_to = dateTo;
      
      const [tradesRes, portfolioRes] = await Promise.all([
        api.get('/api/trades', { params }),
        api.get('/api/portfolio')
      ]);
      
      setTrades(tradesRes.data.trades || []);
      setTotal(tradesRes.data.total || 0);

      // Set P&L data for chart
      const portfolio = portfolioRes.data;
      setPnlData({ 
        labels: portfolio.labels || [], 
        data: portfolio.pnl_history || portfolio.pnl_data || [] 
      });
    } catch (e) {
      console.error('Error loading trades', e);
    } finally {
      setLoading(false);
    }
  }

  async function closeTrade(id) {
    try {
      await api.post('/api/trades/close', { id });
      loadTrades();  // Refresh list
    } catch (e) {
      console.error('Error closing trade', e);
    }
  }

  function openDetails(trade) {
    setSelectedTrade(trade);
  }

  function closeDetails() {
    setSelectedTrade(null);
  }

  const totalPages = Math.ceil(total / perPage);
  const paginationItems = [];
  for (let number = 1; number <= totalPages; number++) {
    paginationItems.push(
      <Pagination.Item key={number} active={number === page} onClick={() => setPage(number)}>
        {number}
      </Pagination.Item>
    );
  }

  const chartData = {
    labels: pnlData.labels,
    datasets: [{
      data: pnlData.data,
      borderColor: '#198754',
      backgroundColor: 'rgba(25,135,84,0.2)',
      fill: true,
      tension: 0.4
    }]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { display: true },
      y: { display: true }
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `P&L: $${context.parsed.y.toFixed(2)}`;
          }
        }
      }
    }
  };

  return (
    <Container fluid className="p-4">
      <Row className="mb-4">
        <Col><h2>Trading Management</h2><p className="text-muted">Monitor and manage your trading positions</p></Col>
      </Row>

      <Card className="mb-4">
        <Card.Header as="h5">Trades</Card.Header>
        <Card.Body>
          {/* Tabs for Open/Closed */}
          <ul className="nav nav-tabs mb-3">
            <li className="nav-item">
              <Button 
                variant="link" 
                className={`nav-link ${statusFilter === 'open' ? 'active' : ''}`} 
                onClick={() => setStatusFilter('open')}
              >
                Open Trades
              </Button>
            </li>
            <li className="nav-item">
              <Button 
                variant="link" 
                className={`nav-link ${statusFilter === 'closed' ? 'active' : ''}`} 
                onClick={() => setStatusFilter('closed')}
              >
                Closed Trades
              </Button>
            </li>
          </ul>

          {/* Filters */}
          <Form className="row g-3 mb-3">
            <Col md={3}>
              <Form.Control 
                placeholder="Filter by Asset (e.g., BTC)" 
                value={assetFilter} 
                onChange={(e) => setAssetFilter(e.target.value)} 
              />
            </Col>
            <Col md={3}>
              <Form.Control 
                type="date" 
                placeholder="From Date" 
                value={dateFrom} 
                onChange={(e) => setDateFrom(e.target.value)} 
              />
            </Col>
            <Col md={3}>
              <Form.Control 
                type="date" 
                placeholder="To Date" 
                value={dateTo} 
                onChange={(e) => setDateTo(e.target.value)} 
              />
            </Col>
            <Col md={3}>
              <Button variant="primary" onClick={loadTrades}>Apply Filters</Button>
            </Col>
          </Form>

          {/* P&L Chart */}
          <Card className="mb-3">
            <Card.Header>P&L Trend</Card.Header>
            <Card.Body style={{ height: '200px' }}>
              <Line data={chartData} options={chartOptions} />
            </Card.Body>
          </Card>

          {/* Trades Table */}
          {loading ? (
            <div className="text-center"><Spinner animation="border" /></div>
          ) : trades.length ? (
            <Table striped hover responsive>
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Entry Price</th>
                  <th>Exit Price</th>
                  <th>Quantity</th>
                  <th>P&L</th>
                  <th>Timestamp</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {trades.map(t => (
                  <tr key={t.id}>
                    <td>{t.asset}</td>
                    <td>${t.entry_price}</td>
                    <td>{t.exit_price ? `$${t.exit_price}` : 'Open'}</td>
                    <td>{t.quantity}</td>
                    <td>
                      <Badge bg={t.pnl > 0 ? 'success' : t.pnl < 0 ? 'danger' : 'secondary'}>
                        ${t.pnl}
                      </Badge>
                    </td>
                    <td>{new Date(t.timestamp).toLocaleString()}</td>
                    <td>
                      <Button 
                        variant="outline-primary" 
                        size="sm" 
                        onClick={() => openDetails(t)}
                      >
                        Details
                      </Button>
                      {t.status === 'open' && (
                        <Button 
                          variant="outline-danger" 
                          size="sm" 
                          className="ms-2" 
                          onClick={() => closeTrade(t.id)}
                        >
                          Close
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          ) : (
            <div className="text-center text-muted">No trades found</div>
          )}

          {totalPages > 1 && (
            <Pagination className="justify-content-center mt-3">
              {paginationItems}
            </Pagination>
          )}
        </Card.Body>
      </Card>

      {/* Trade Details Modal */}
      <Modal show={!!selectedTrade} onHide={closeDetails}>
        <Modal.Header closeButton>
          <Modal.Title>Trade Details</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedTrade && (
            <div>
              <p><strong>Asset:</strong> {selectedTrade.asset}</p>
              <p><strong>Entry Price:</strong> ${selectedTrade.entry_price}</p>
              <p><strong>Exit Price:</strong> {selectedTrade.exit_price || 'N/A'}</p>
              <p><strong>Quantity:</strong> {selectedTrade.quantity}</p>
              <p><strong>P&L:</strong> ${selectedTrade.pnl}</p>
              <p><strong>Status:</strong> {selectedTrade.status}</p>
              <p><strong>Side:</strong> {selectedTrade.side || 'N/A'}</p>
              <p><strong>Confidence:</strong> {selectedTrade.confidence || 0}%</p>
              <p><strong>Timestamp:</strong> {new Date(selectedTrade.timestamp).toLocaleString()}</p>
            </div>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={closeDetails}>Close</Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
}

export default Trades;