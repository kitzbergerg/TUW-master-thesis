import WebSocket, { WebSocketServer } from 'ws';
import http from 'http';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';
import { Node, Graph } from './graph.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const end = new Node('end', [], '')
const start = new Node('start', [end], '')
const graph = new Graph([start, end])


function broadcastNumberOfConnectedUsers(clients) {
  Object.values(clients).forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({
        type: 'connectedUsers',
        message: Object.keys(clients).length
      }));
    } else {
      console.error("Unavailable socket");
    }
  });
}

// Set up Express app
const app = express();
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.static(path.join(__dirname, 'model')));

// Create HTTP server
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocketServer({ server });

// Store connected clients
const clients = {};

// Handle WebSocket connections
wss.on('connection', (ws) => {
  const userUuid = uuidv4();
  console.log('Client connected');
  clients[userUuid] = ws;
  broadcastNumberOfConnectedUsers(clients);
  graph.addUser(userUuid)

  // Handle messages from clients
  ws.on('message', (message) => {
    const data = JSON.parse(message)

    switch (data.type) {
      case 'inferenceRequest':
        const text = data.message;
        const tokenized = []
        for (let i = 0; i < text.length; i++) {
          tokenized[i] = text.charCodeAt(i);
        }
        console.log(tokenized)
        const client = graph.getWorker(graph.startNodeId);
        const clientWs = clients[client];
        const request = { type: 'computation', message: tokenized, nodeId: graph.startNodeId };
        clientWs.send(JSON.stringify(request));
        break;
      case 'computationResult':
        const result = data.message;
        const nextNodes = graph.getNextNodes(data.nodeId);
        if (nextNodes.length == 0) {
          console.log("Finished inference");
          console.log(`Final result: ${result}`);
        } else {
          nextNodes.forEach(node => {
            const client = graph.getWorker(node.id);
            if (!client) return
            const clientWs = clients[client];
            const request = { type: 'computation', message: result, nodeId: node.id };
            clientWs.send(JSON.stringify(request));
          })
        }
        break;
      default:
        console.error('Unknown type');
    }
  });

  // Handle client disconnection
  ws.on('close', () => {
    console.log('Client disconnected');
    delete clients[userUuid];
    graph.removeUser(userUuid);

    broadcastNumberOfConnectedUsers(clients)
  });
});

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
