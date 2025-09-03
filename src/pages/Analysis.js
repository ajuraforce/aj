// src/pages/Analysis.js
import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Accordion, Button, Table, Alert, Spinner } from 'react-bootstrap';
import { Line, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import HeatMapGrid from 'react-heatmap-grid';
import api from '../services/api';
import socket from '../services/socket';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function Analysis() {
  const [signals, setSignals] = useState({ signals: [], labels: [], data: [] });
  const [sentiment, setSentiment] = useState({ sentiment: [] });
  const [flow, setFlow] = useState({ flow: [], labels: [], data: [] });
  const [portfolio, setPortfolio] = useState({ allocation: [] });
  const [regimes, setRegimes] = useState({ regime: '', confidence: 0 });
  const [correlations, setCorrelations] = useState({ correlations: [] });
  const [timeframes, setTimeframes] = useState({ timeframes: [] });
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(null);
  
  // Analysis-to-Action Pipeline states
  const [analysisText, setAnalysisText] = useState('');
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [parsedAnalysis, setParsedAnalysis] = useState(null);
  const [analysisError, setAnalysisError] = useState('');
  const [autoExecute, setAutoExecute] = useState(false);
  const [mode, setMode] = useState('paper');

  useEffect(() => {
    loadAnalysis();
    
    // Set up real-time updates via socket
    socket.on('analysis_update', (update) => {
      // Update specific section in real-time
      if (update.type === 'signal') setSignals(prev => ({ ...prev, signals: [update.data, ...prev.signals] }));
      if (update.type === 'sentiment') setSentiment(prev => ({ ...prev, sentiment: update.data }));
      if (update.type === 'flow') setFlow(prev => ({ ...prev, flow: [update.data, ...prev.flow] }));
    });
    
    return () => socket.off('analysis_update');
  }, []);

  const processAnalysis = async () => {
    if (!analysisText.trim()) {
      setAnalysisError('Please enter analysis text');
      return;
    }
    
    setAnalysisLoading(true);
    setAnalysisError('');
    
    try {
      const response = await api.post('/api/analyze/paste', {
        text: analysisText,
        mode,
        auto_execute: autoExecute
      });
      
      setParsedAnalysis(response.data);
      console.log('Analysis processed:', response.data);
    } catch (error) {
      console.error('Analysis processing failed:', error);
      setAnalysisError(error.response?.data?.error || 'Failed to process analysis');
    } finally {
      setAnalysisLoading(false);
    }
  };

  const clearAnalysis = () => {
    setAnalysisText('');
    setParsedAnalysis(null);
    setAnalysisError('');
  };

  async function loadAnalysis() {
    setLoading(true);
    try {
      const [signalsRes, sentimentRes, flowRes, portfolioRes, regimesRes, correlationsRes, timeframesRes] = await Promise.all([
        api.get('/api/analysis/signals'),
        api.get('/api/analysis/sentiment'),
        api.get('/api/analysis/flow'),
        api.get('/api/analysis/portfolio'),
        api.get('/api/analysis/regimes'),
        api.get('/api/analysis/correlations'),
        api.get('/api/analysis/timeframes')
      ]);
      
      setSignals(signalsRes.data);
      setSentiment(sentimentRes.data);
      setFlow(flowRes.data);
      setPortfolio(portfolioRes.data);
      setRegimes(regimesRes.data);
      setCorrelations(correlationsRes.data);
      setTimeframes(timeframesRes.data);
    } catch (e) {
      console.error('Error loading analysis', e);
    } finally {
      setLoading(false);
    }
  }

  function toggleExpand(section) {
    setExpanded(expanded === section ? null : section);
  }

  // Chart configurations
  const signalConfidenceData = {
    labels: signals.labels,
    datasets: [{
      label: 'Signal Confidence',
      data: signals.data,
      borderColor: '#0d6efd',
      backgroundColor: 'rgba(13, 110, 253, 0.1)',
      fill: true,
      tension: 0.4
    }]
  };

  const allocationData = {
    labels: portfolio.allocation.map(a => a.asset),
    datasets: [{
      data: portfolio.allocation.map(a => a.percentage),
      backgroundColor: ['#0d6efd', '#198754', '#ffc107', '#dc3545', '#6f42c1']
    }]
  };

  const flowData = {
    labels: flow.labels,
    datasets: [{
      label: 'Institutional Flow Volume',
      data: flow.data,
      borderColor: '#198754',
      backgroundColor: 'rgba(25, 135, 84, 0.1)',
      fill: true,
      tension: 0.4
    }]
  };

  // Prepare heatmap data for sentiment
  const sentimentAssets = ['BTC', 'ETH', 'ADA', 'SOL', 'LINK'];
  const sentimentTypes = ['Positive', 'Negative', 'Neutral'];
  const sentimentMatrix = sentimentTypes.map(type => 
    sentimentAssets.map(asset => {
      const dataPoint = sentiment.sentiment.find(s => s.x === asset && s.y === type);
      return dataPoint ? dataPoint.value : 0;
    })
  );

  // Prepare heatmap data for correlations
  const correlationAssets = ['BTC', 'ETH', 'ADA', 'SOL', 'LINK'];
  const correlationMatrix = correlationAssets.map(asset1 =>
    correlationAssets.map(asset2 => {
      const dataPoint = correlations.correlations.find(c => c.x === asset1 && c.y === asset2);
      return dataPoint ? dataPoint.value : 0;
    })
  );

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
      },
    },
  };

  return (
    <Container fluid className="p-4">
      <Card className="mb-4">
        <Card.Header as="h5" className="d-flex justify-content-between align-items-center">
          <span>Advanced Analysis Dashboard</span>
          <Button variant="outline-primary" size="sm" onClick={loadAnalysis} disabled={loading}>
            {loading ? <Spinner animation="border" size="sm" /> : 'Refresh'}
          </Button>
        </Card.Header>
        <Card.Body>
          {loading ? (
            <div className="text-center p-5">
              <Spinner animation="border" role="status">
                <span className="visually-hidden">Loading...</span>
              </Spinner>
            </div>
          ) : (
            <Accordion>
              {/* Trading Signals Summary */}
              <Accordion.Item eventKey="0">
                <Accordion.Header>
                  <i className="fas fa-chart-line me-2"></i>
                  Trading Signals Summary ({signals.signals.length})
                </Accordion.Header>
                <Accordion.Body>
                  <Row>
                    <Col md={6}>
                      <Table striped hover responsive size="sm">
                        <thead>
                          <tr>
                            <th>Asset</th>
                            <th>Type</th>
                            <th>Confidence</th>
                            <th>Time</th>
                          </tr>
                        </thead>
                        <tbody>
                          {signals.signals.slice(0, 5).map((signal, idx) => (
                            <tr key={signal.id || idx}>
                              <td><strong>{signal.asset}</strong></td>
                              <td>
                                <span className={`badge ${signal.type === 'BUY' ? 'bg-success' : signal.type === 'SELL' ? 'bg-danger' : 'bg-secondary'}`}>
                                  {signal.type}
                                </span>
                              </td>
                              <td>{signal.confidence.toFixed(1)}%</td>
                              <td>{new Date(signal.timestamp).toLocaleTimeString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </Col>
                    <Col md={6}>
                      <div style={{ height: '250px' }}>
                        <Line data={signalConfidenceData} options={chartOptions} />
                      </div>
                    </Col>
                  </Row>
                </Accordion.Body>
              </Accordion.Item>

              {/* Analysis-to-Action Pipeline */}
              <Accordion.Item eventKey="analysis-pipeline">
                <Accordion.Header>
                  <i className="fas fa-magic me-2"></i>
                  Analysis-to-Action Pipeline
                  {parsedAnalysis && <Badge bg="success" className="ms-2">Active</Badge>}
                </Accordion.Header>
                <Accordion.Body>
                  <Row>
                    <Col md={6}>
                      <Card className="h-100">
                        <Card.Header>
                          <h6 className="mb-0">
                            <i className="fas fa-paste me-2"></i>
                            Paste Trading Analysis
                          </h6>
                        </Card.Header>
                        <Card.Body>
                          <form onSubmit={(e) => { e.preventDefault(); processAnalysis(); }}>
                            <div className="mb-3">
                              <textarea
                                className="form-control"
                                rows="8"
                                placeholder="Paste your trading analysis here...
                                
Example:
* NIFTY (Trade 1)
Direction: Long
Entry Zone: 24,600–24,620
Stop: 24,540
Targets: 24,750/24,800
R:R: 1:2.5

Invalidation: Daily close below 24,500"
                                value={analysisText}
                                onChange={(e) => setAnalysisText(e.target.value)}
                              />
                            </div>
                            
                            <Row className="mb-3">
                              <Col md={6}>
                                <div className="form-check form-switch">
                                  <input
                                    className="form-check-input"
                                    type="checkbox"
                                    id="paperMode"
                                    checked={mode === 'paper'}
                                    onChange={(e) => setMode(e.target.checked ? 'paper' : 'live')}
                                  />
                                  <label className="form-check-label" htmlFor="paperMode">
                                    Paper Trading Mode
                                  </label>
                                </div>
                              </Col>
                              <Col md={6}>
                                <div className="form-check form-switch">
                                  <input
                                    className="form-check-input"
                                    type="checkbox"
                                    id="autoExecute"
                                    checked={autoExecute}
                                    onChange={(e) => setAutoExecute(e.target.checked)}
                                  />
                                  <label className="form-check-label" htmlFor="autoExecute">
                                    Auto-Execute Triggered Trades
                                  </label>
                                </div>
                              </Col>
                            </Row>
                            
                            {analysisError && (
                              <Alert variant="danger" className="mb-3">
                                {analysisError}
                              </Alert>
                            )}
                            
                            <div className="d-grid gap-2 d-md-flex">
                              <Button 
                                variant="primary" 
                                type="submit" 
                                disabled={analysisLoading || !analysisText.trim()}
                              >
                                {analysisLoading ? <Spinner animation="border" size="sm" /> : 'Process Analysis'}
                              </Button>
                              <Button variant="outline-secondary" onClick={clearAnalysis}>
                                Clear
                              </Button>
                            </div>
                          </form>
                        </Card.Body>
                      </Card>
                    </Col>
                    
                    <Col md={6}>
                      <Card className="h-100">
                        <Card.Header>
                          <h6 className="mb-0">
                            <i className="fas fa-chart-bar me-2"></i>
                            Parsed Trades & Signals
                          </h6>
                        </Card.Header>
                        <Card.Body>
                          {parsedAnalysis ? (
                            <>
                              <div className="mb-3">
                                <small className="text-muted">
                                  Processing Time: {parsedAnalysis.processing_time} | 
                                  Active Monitoring: {parsedAnalysis.active_monitoring} signals
                                </small>
                              </div>
                              
                              {parsedAnalysis.parsed.trades.length > 0 && (
                                <div className="mb-4">
                                  <h6>Extracted Trades ({parsedAnalysis.parsed.trades.length})</h6>
                                  <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                    {parsedAnalysis.parsed.trades.map((trade, idx) => (
                                      <Card key={idx} className="mb-2 border-start-4 border-primary">
                                        <Card.Body className="py-2">
                                          <div className="d-flex justify-content-between align-items-start">
                                            <div>
                                              <strong>{trade.asset}</strong>
                                              <span className={`badge ms-2 ${trade.direction === 'LONG' ? 'bg-success' : 'bg-danger'}`}>
                                                {trade.direction}
                                              </span>
                                              <span className={`badge ms-1 ${
                                                trade.status === 'triggered' ? 'bg-warning' : 
                                                trade.status === 'pending' ? 'bg-secondary' : 'bg-info'
                                              }`}>
                                                {trade.status}
                                              </span>
                                            </div>
                                            <small className="text-muted">
                                              R:R {trade.calculated_rr || trade.rr}
                                            </small>
                                          </div>
                                          <small>
                                            Entry: {trade.entry_zone} | Stop: {trade.stop} | 
                                            Targets: {trade.targets.join('/')}
                                            {trade.current_price && (
                                              <> | Current: ${trade.current_price.toLocaleString()}</>
                                            )}
                                          </small>
                                        </Card.Body>
                                      </Card>
                                    ))}
                                  </div>
                                </div>
                              )}
                              
                              {parsedAnalysis.executed_trades.length > 0 && (
                                <div className="mb-4">
                                  <h6 className="text-success">Executed Trades ({parsedAnalysis.executed_trades.length})</h6>
                                  {parsedAnalysis.executed_trades.map((trade, idx) => (
                                    <Alert key={idx} variant="success" className="py-2">
                                      <small>
                                        <strong>{trade.symbol}</strong> executed at ${trade.price} 
                                        (Qty: {trade.quantity})
                                      </small>
                                    </Alert>
                                  ))}
                                </div>
                              )}
                              
                              {parsedAnalysis.parsed.invalidations.length > 0 && (
                                <div className="mb-3">
                                  <h6>Invalidation Conditions</h6>
                                  {parsedAnalysis.parsed.invalidations.map((inv, idx) => (
                                    <Alert key={idx} variant="warning" className="py-2">
                                      <small>{inv}</small>
                                    </Alert>
                                  ))}
                                </div>
                              )}
                            </>
                          ) : (
                            <div className="text-center text-muted py-5">
                              <i className="fas fa-upload fa-2x mb-3"></i>
                              <p>Paste your trading analysis to see parsed trades and automated signals</p>
                            </div>
                          )}
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Accordion.Body>
              </Accordion.Item>

              {/* Sentiment Tracking */}
              <Accordion.Item eventKey="1">
                <Accordion.Header>
                  <i className="fas fa-heart me-2"></i>
                  Market Sentiment Heatmap
                </Accordion.Header>
                <Accordion.Body>
                  <div className="text-center">
                    <HeatMapGrid
                      xLabels={sentimentAssets}
                      yLabels={sentimentTypes}
                      data={sentimentMatrix}
                      cellStyle={(background, value, min, max, data, x, y) => ({
                        background: `rgba(13, 110, 253, ${value / 100})`,
                        fontSize: '12px',
                        color: '#fff'
                      })}
                      cellRender={value => value && `${value}%`}
                      square={true}
                    />
                  </div>
                </Accordion.Body>
              </Accordion.Item>

              {/* Institutional Flow */}
              <Accordion.Item eventKey="2">
                <Accordion.Header>
                  <i className="fas fa-building me-2"></i>
                  Institutional Flow Analysis
                </Accordion.Header>
                <Accordion.Body>
                  <Row>
                    <Col md={6}>
                      <Table striped hover responsive size="sm">
                        <thead>
                          <tr>
                            <th>Asset</th>
                            <th>Volume</th>
                            <th>Direction</th>
                          </tr>
                        </thead>
                        <tbody>
                          {flow.flow.slice(0, 5).map((f, idx) => (
                            <tr key={idx}>
                              <td><strong>{f.asset}</strong></td>
                              <td>{f.volume.toLocaleString()}</td>
                              <td>
                                <span className={`badge ${f.direction === 'inflow' ? 'bg-success' : 'bg-danger'}`}>
                                  {f.direction}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </Col>
                    <Col md={6}>
                      <div style={{ height: '250px' }}>
                        <Line data={flowData} options={chartOptions} />
                      </div>
                    </Col>
                  </Row>
                </Accordion.Body>
              </Accordion.Item>

              {/* Portfolio Optimization */}
              <Accordion.Item eventKey="3">
                <Accordion.Header>
                  <i className="fas fa-pie-chart me-2"></i>
                  Portfolio Optimization
                </Accordion.Header>
                <Accordion.Body>
                  <Row>
                    <Col md={6}>
                      <div style={{ height: '300px' }}>
                        <Pie data={allocationData} options={pieOptions} />
                      </div>
                    </Col>
                    <Col md={6}>
                      <Table striped hover responsive>
                        <thead>
                          <tr>
                            <th>Asset</th>
                            <th>Allocation</th>
                          </tr>
                        </thead>
                        <tbody>
                          {portfolio.allocation.map((alloc, idx) => (
                            <tr key={idx}>
                              <td><strong>{alloc.asset}</strong></td>
                              <td>{alloc.percentage.toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </Col>
                  </Row>
                </Accordion.Body>
              </Accordion.Item>

              {/* Regime Detection */}
              <Accordion.Item eventKey="4">
                <Accordion.Header>
                  <i className="fas fa-thermometer-half me-2"></i>
                  Market Regime Detection
                </Accordion.Header>
                <Accordion.Body>
                  <Alert variant="info" className="text-center">
                    <h5 className="mb-1">Current Market Regime: <strong>{regimes.regime}</strong></h5>
                    <p className="mb-0">Confidence: <strong>{regimes.confidence}%</strong></p>
                  </Alert>
                </Accordion.Body>
              </Accordion.Item>

              {/* Correlation Matrix */}
              <Accordion.Item eventKey="5">
                <Accordion.Header>
                  <i className="fas fa-project-diagram me-2"></i>
                  Asset Correlation Matrix
                </Accordion.Header>
                <Accordion.Body>
                  <div className="text-center">
                    <HeatMapGrid
                      xLabels={correlationAssets}
                      yLabels={correlationAssets}
                      data={correlationMatrix}
                      cellStyle={(background, value, min, max, data, x, y) => ({
                        background: value > 0 ? `rgba(25, 135, 84, ${Math.abs(value)})` : `rgba(220, 53, 69, ${Math.abs(value)})`,
                        fontSize: '11px',
                        color: '#fff'
                      })}
                      cellRender={value => value && value.toFixed(2)}
                      square={true}
                    />
                  </div>
                </Accordion.Body>
              </Accordion.Item>

              {/* Multi-Timeframe Analysis */}
              <Accordion.Item eventKey="6">
                <Accordion.Header>
                  <i className="fas fa-clock me-2"></i>
                  Multi-Timeframe Analysis
                </Accordion.Header>
                <Accordion.Body>
                  <Table striped hover responsive>
                    <thead>
                      <tr>
                        <th>Timeframe</th>
                        <th>Trend</th>
                        <th>Strength</th>
                        <th>Signals</th>
                      </tr>
                    </thead>
                    <tbody>
                      {timeframes.timeframes.map((tf, idx) => (
                        <tr key={idx}>
                          <td><strong>{tf.timeframe}</strong></td>
                          <td>
                            <span className={`badge ${tf.trend === 'bullish' ? 'bg-success' : tf.trend === 'bearish' ? 'bg-danger' : 'bg-secondary'}`}>
                              {tf.trend}
                            </span>
                          </td>
                          <td>{tf.strength}%</td>
                          <td>{tf.signals}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </Accordion.Body>
              </Accordion.Item>
            </Accordion>
          )}
        </Card.Body>
      </Card>
    </Container>
  );
}

export default Analysis;