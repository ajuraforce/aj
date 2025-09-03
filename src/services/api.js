// src/services/api.js
import axios from 'axios';

const api = axios.create({
  baseURL: '/',       // Proxy to Flask backend in development
  timeout: 10000
});

export default api;