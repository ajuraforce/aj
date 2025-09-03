// src/services/socket.js
import { io } from 'socket.io-client';

const socket = io('/', {        // Connect to Flask-SocketIO server
  transports: ['websocket']
});

export default socket;