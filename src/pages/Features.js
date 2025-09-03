import React from 'react';
import { Container, Row, Col, Card, Badge, ListGroup } from 'react-bootstrap';

function Features() {
  return (
    <Container fluid className="p-4">
      <Row className="mb-4">
        <Col>
          <h2>Platform Features</h2>
          <p className="text-muted">Complete overview of AjxAI's AI-powered trading and social intelligence capabilities</p>
        </Col>
      </Row>

      {/* Data Layer - Scanners */}
      <Row className="mb-4">
        <Col xs={12}>
          <Card className="h-100">
            <Card.Header className="bg-primary text-white">
              <h5 className="mb-0">
                <i className="bi bi-database me-2"></i>
                Data Layer - Real-time Scanners
              </h5>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={6} lg={3} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body className="text-center">
                      <i className="bi bi-reddit text-danger fs-2 mb-2"></i>
                      <h6>Reddit Scanner</h6>
                      <small className="text-muted">Monitors crypto subreddits for sentiment and discussion patterns using PRAW</small>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6} lg={3} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body className="text-center">
                      <i className="bi bi-currency-bitcoin text-warning fs-2 mb-2"></i>
                      <h6>Binance Scanner</h6>
                      <small className="text-muted">Real-time crypto price movements and volume data via CCXT API</small>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6} lg={3} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body className="text-center">
                      <i className="bi bi-newspaper text-info fs-2 mb-2"></i>
                      <h6>News Scanner</h6>
                      <small className="text-muted">Financial news from RSS feeds (Yahoo Finance, BeInCrypto, Reuters)</small>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6} lg={3} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body className="text-center">
                      <i className="bi bi-bank text-success fs-2 mb-2"></i>
                      <h6>India Equity Scanner</h6>
                      <small className="text-muted">NIFTY50 and BankNifty market data from NSE/BSE exchanges</small>
                    </Card.Body>
                  </Card>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Decode Layer - AI Analyzers */}
      <Row className="mb-4">
        <Col xs={12}>
          <Card className="h-100">
            <Card.Header className="bg-success text-white">
              <h5 className="mb-0">
                <i className="bi bi-brain me-2"></i>
                Decode Layer - AI-Powered Analysis Engine
                <Badge bg="light" text="dark" className="ms-2">15 Features Operational</Badge>
              </h5>
            </Card.Header>
            <Card.Body>
              <Row>
                {/* Core Analysis Features */}
                <Col lg={6} className="mb-3">
                  <h6><i className="bi bi-gear-fill text-primary me-2"></i>Core Analysis Features</h6>
                  <ListGroup className="mb-3">
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Smart Alert Manager</strong>
                        <small className="d-block text-muted">GPT-5 powered intelligent alert filtering with risk scoring</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Portfolio Optimizer</strong>
                        <small className="d-block text-muted">Advanced portfolio analysis with AI market commentary</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Signal Engine</strong>
                        <small className="d-block text-muted">Trading signal generation with confidence scoring</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Pattern Analyzer</strong>
                        <small className="d-block text-muted">Cross-source correlation and pattern detection</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                  </ListGroup>
                </Col>

                {/* Advanced AI Features */}
                <Col lg={6} className="mb-3">
                  <h6><i className="bi bi-robot text-info me-2"></i>Advanced AI Features</h6>
                  <ListGroup className="mb-3">
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>ML Pattern Recognizer</strong>
                        <small className="d-block text-muted">Machine learning for pattern discovery and prediction</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Multi-Timeframe Analysis</strong>
                        <small className="d-block text-muted">Technical analysis across 1h, 4h, and 1d timeframes</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Sentiment Flow Analysis</strong>
                        <small className="d-block text-muted">Emotion tracking and trend detection with AI forecasting</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Institutional Flow Detection</strong>
                        <small className="d-block text-muted">Large player movement and whale activity identification</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                  </ListGroup>
                </Col>

                {/* Phase 5 Advanced Features */}
                <Col lg={6} className="mb-3">
                  <h6><i className="bi bi-rocket text-warning me-2"></i>Phase 5 Advanced Features</h6>
                  <ListGroup className="mb-3">
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>AI Predictive Forecasting</strong>
                        <small className="d-block text-muted">Prophet-powered market, geopolitical, and stock predictions (7-30 days)</small>
                      </div>
                      <Badge bg="warning">New</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Community Simulator</strong>
                        <small className="d-block text-muted">Advanced engagement modeling with viral detection</small>
                      </div>
                      <Badge bg="warning">New</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>User Acquisition Engine</strong>
                        <small className="d-block text-muted">Cross-platform automated growth campaigns</small>
                      </div>
                      <Badge bg="warning">New</Badge>
                    </ListGroup.Item>
                  </ListGroup>
                </Col>

                {/* AI Strategist Features */}
                <Col lg={6} className="mb-3">
                  <h6><i className="bi bi-lightbulb text-danger me-2"></i>AI Strategist Features</h6>
                  <ListGroup>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Knowledge Graph</strong>
                        <small className="d-block text-muted">Links events, assets, sectors using NetworkX</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Regime Engine</strong>
                        <small className="d-block text-muted">Market regime detection (risk-on/off, liquidity, volatility)</small>
                      </div>
                      <Badge bg="success">Active</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>GPT Integration</strong>
                        <small className="d-block text-muted">AI-powered event typing and narrative building</small>
                      </div>
                      <Badge bg="info">Enhanced</Badge>
                    </ListGroup.Item>
                    <ListGroup.Item className="d-flex justify-content-between align-items-center">
                      <div>
                        <strong>Advanced Models</strong>
                        <small className="d-block text-muted">SVAR and state-space models for macro analysis</small>
                      </div>
                      <Badge bg="info">Enhanced</Badge>
                    </ListGroup.Item>
                  </ListGroup>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Action Layer - Executors */}
      <Row className="mb-4">
        <Col xs={12}>
          <Card className="h-100">
            <Card.Header className="bg-warning text-dark">
              <h5 className="mb-0">
                <i className="bi bi-lightning me-2"></i>
                Action Layer - Execution Engine
              </h5>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={6} lg={4} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body>
                      <div className="d-flex align-items-center mb-2">
                        <i className="bi bi-graph-up text-success fs-4 me-2"></i>
                        <h6 className="mb-0">Paper Trading Engine</h6>
                      </div>
                      <small className="text-muted">Safe simulation mode for testing strategies with PnL tracking and performance analytics</small>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6} lg={4} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body>
                      <div className="d-flex align-items-center mb-2">
                        <i className="bi bi-bell text-primary fs-4 me-2"></i>
                        <h6 className="mb-0">Multi-Channel Alerts</h6>
                      </div>
                      <small className="text-muted">Telegram, Discord, email notifications with intelligent routing and risk scoring</small>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6} lg={4} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body>
                      <div className="d-flex align-items-center mb-2">
                        <i className="bi bi-reddit text-danger fs-4 me-2"></i>
                        <h6 className="mb-0">Reddit Automation</h6>
                      </div>
                      <small className="text-muted">Automated social media engagement with safety controls and rate limiting</small>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6} lg={4} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body>
                      <div className="d-flex align-items-center mb-2">
                        <i className="bi bi-people text-info fs-4 me-2"></i>
                        <h6 className="mb-0">Community Simulator</h6>
                      </div>
                      <small className="text-muted">Simulates community engagement and measures viral potential of content</small>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6} lg={4} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body>
                      <div className="d-flex align-items-center mb-2">
                        <i className="bi bi-megaphone text-warning fs-4 me-2"></i>
                        <h6 className="mb-0">User Acquisition</h6>
                      </div>
                      <small className="text-muted">Cross-platform posting campaigns with funnel analytics and conversion tracking</small>
                    </Card.Body>
                  </Card>
                </Col>
                <Col md={6} lg={4} className="mb-3">
                  <Card className="border-0 bg-light h-100">
                    <Card.Body>
                      <div className="d-flex align-items-center mb-2">
                        <i className="bi bi-telegram text-info fs-4 me-2"></i>
                        <h6 className="mb-0">Telegram Bot</h6>
                      </div>
                      <small className="text-muted">Premium subscriptions, auto-moderation, and community management</small>
                    </Card.Body>
                  </Card>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Platform Capabilities */}
      <Row className="mb-4">
        <Col xs={12}>
          <Card>
            <Card.Header className="bg-dark text-white">
              <h5 className="mb-0">
                <i className="bi bi-shield-check me-2"></i>
                Platform Capabilities & Safety
              </h5>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={6} className="mb-3">
                  <h6><i className="bi bi-cpu text-primary me-2"></i>Core Capabilities</h6>
                  <ul className="list-unstyled">
                    <li><i className="bi bi-check-circle text-success me-2"></i>Real-time multi-source data processing</li>
                    <li><i className="bi bi-check-circle text-success me-2"></i>AI-powered pattern recognition and forecasting</li>
                    <li><i className="bi bi-check-circle text-success me-2"></i>Cross-asset correlation analysis</li>
                    <li><i className="bi bi-check-circle text-success me-2"></i>Automated decision making with confidence scoring</li>
                    <li><i className="bi bi-check-circle text-success me-2"></i>Professional-grade risk management</li>
                  </ul>
                </Col>
                <Col md={6} className="mb-3">
                  <h6><i className="bi bi-shield text-success me-2"></i>Safety & Controls</h6>
                  <ul className="list-unstyled">
                    <li><i className="bi bi-check-circle text-success me-2"></i>Paper trading mode by default</li>
                    <li><i className="bi bi-check-circle text-success me-2"></i>Rate limiting and circuit breakers</li>
                    <li><i className="bi bi-check-circle text-success me-2"></i>Comprehensive permission system</li>
                    <li><i className="bi bi-check-circle text-success me-2"></i>State persistence and backup</li>
                    <li><i className="bi bi-check-circle text-success me-2"></i>Ban detection and safety controls</li>
                  </ul>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {/* Integration APIs */}
      <Row className="mb-4">
        <Col xs={12}>
          <Card>
            <Card.Header className="bg-info text-white">
              <h5 className="mb-0">
                <i className="bi bi-plug me-2"></i>
                API Integrations & External Services
              </h5>
            </Card.Header>
            <Card.Body>
              <Row>
                <Col md={4} className="mb-3">
                  <h6>Trading & Market Data</h6>
                  <ul className="list-unstyled small">
                    <li><i className="bi bi-circle-fill text-warning me-2" style={{fontSize: '6px'}}></i>Binance API (CCXT)</li>
                    <li><i className="bi bi-circle-fill text-success me-2" style={{fontSize: '6px'}}></i>Yahoo Finance</li>
                    <li><i className="bi bi-circle-fill text-info me-2" style={{fontSize: '6px'}}></i>NSE/BSE Market Data</li>
                  </ul>
                </Col>
                <Col md={4} className="mb-3">
                  <h6>Social & Communication</h6>
                  <ul className="list-unstyled small">
                    <li><i className="bi bi-circle-fill text-danger me-2" style={{fontSize: '6px'}}></i>Reddit API (PRAW)</li>
                    <li><i className="bi bi-circle-fill text-info me-2" style={{fontSize: '6px'}}></i>Telegram Bot API</li>
                    <li><i className="bi bi-circle-fill text-primary me-2" style={{fontSize: '6px'}}></i>Discord Webhooks</li>
                    <li><i className="bi bi-circle-fill text-secondary me-2" style={{fontSize: '6px'}}></i>Twitter API</li>
                  </ul>
                </Col>
                <Col md={4} className="mb-3">
                  <h6>AI & Analytics</h6>
                  <ul className="list-unstyled small">
                    <li><i className="bi bi-circle-fill text-success me-2" style={{fontSize: '6px'}}></i>OpenAI GPT-4/5</li>
                    <li><i className="bi bi-circle-fill text-warning me-2" style={{fontSize: '6px'}}></i>Prophet ML (Facebook)</li>
                    <li><i className="bi bi-circle-fill text-dark me-2" style={{fontSize: '6px'}}></i>GitHub API</li>
                    <li><i className="bi bi-circle-fill text-primary me-2" style={{fontSize: '6px'}}></i>RSS Feeds (Multiple)</li>
                  </ul>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default Features;