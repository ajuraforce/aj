// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Dashboard from './pages/Dashboard';
import Trades from './pages/Trades';
import Analysis from './pages/Analysis';
import Community from './pages/Community';
import Health from './pages/Health';
import LiveAlerts from './pages/LiveAlerts';
import Features from './pages/Features';

function App() {
  return (
    <Router>
      <Navigation />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/trades" element={<Trades />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/community" element={<Community />} />
        <Route path="/health" element={<Health />} />
        <Route path="/live-alerts" element={<LiveAlerts />} />
        <Route path="/features" element={<Features />} />
      </Routes>
    </Router>
  );
}

export default App;