import React from 'react';
import { Navbar, Nav, Container } from 'react-bootstrap';
import { NavLink } from 'react-router-dom';

function Navigation() {
  return (
    <Navbar bg="dark" variant="dark" expand="lg" className="mb-4">
      <Container fluid>
        <Navbar.Brand as={NavLink} to="/">
          <i className="bi bi-cpu me-2"></i>
          AjxAI Trading Platform
        </Navbar.Brand>
        
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="me-auto">
            <Nav.Link 
              as={NavLink} 
              to="/" 
              className={({ isActive }) => isActive ? 'active' : ''}
            >
              <i className="bi bi-speedometer2 me-1"></i>
              Dashboard
            </Nav.Link>
            
            <Nav.Link 
              as={NavLink} 
              to="/trades" 
              className={({ isActive }) => isActive ? 'active' : ''}
            >
              <i className="bi bi-graph-up me-1"></i>
              Trades
            </Nav.Link>
            
            <Nav.Link 
              as={NavLink} 
              to="/analysis" 
              className={({ isActive }) => isActive ? 'active' : ''}
            >
              <i className="bi bi-bar-chart me-1"></i>
              Analysis
            </Nav.Link>
            
            <Nav.Link 
              as={NavLink} 
              to="/live-alerts" 
              className={({ isActive }) => isActive ? 'active' : ''}
            >
              <i className="bi bi-bell me-1"></i>
              Live Alerts
            </Nav.Link>
            
            <Nav.Link 
              as={NavLink} 
              to="/community" 
              className={({ isActive }) => isActive ? 'active' : ''}
            >
              <i className="bi bi-people me-1"></i>
              Community
            </Nav.Link>
            
            <Nav.Link 
              as={NavLink} 
              to="/features" 
              className={({ isActive }) => isActive ? 'active' : ''}
            >
              <i className="bi bi-grid me-1"></i>
              Features
            </Nav.Link>
            
            <Nav.Link 
              as={NavLink} 
              to="/health" 
              className={({ isActive }) => isActive ? 'active' : ''}
            >
              <i className="bi bi-heart-pulse me-1"></i>
              Health
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}

export default Navigation;