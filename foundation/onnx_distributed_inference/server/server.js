import WebSocket, { WebSocketServer } from 'ws';
import http from 'http';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { v4 as uuidv4 } from 'uuid';
import { AutoTokenizer } from '@huggingface/transformers';

import { Node, Graph } from './graph.ts';

import externalDataP1 from './data_p1.json' with { type: "json" };
import externalDataP2 from './data_p2.json' with { type: "json" };

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const end = new Node('end', [], { modelURI: 'http://localhost:3000/phi/split/p2/model.onnx', externalData: externalDataP2 })
const start = new Node('start', [end], { modelURI: 'http://localhost:3000/phi/split/p1/model.onnx', externalData: externalDataP1 })
const graph = new Graph([start, end])

const tokenizer = await AutoTokenizer.from_pretrained('microsoft/phi-2');
const vocabSize = 51200;

function argMax(array) {
  return [].reduce.call(array, (m, c, i, arr) => c > arr[m] ? i : m, 0)
}

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
app.use(function (req, res, next) {
  res.header('Access-Control-Allow-Origin', '*');
  res.header(
    'Access-Control-Allow-Headers',
    'Origin, X-Requested-With, Content-Type, Accept'
  );
  next();
});
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
  const nodeId = graph.addUser(userUuid);
  if (nodeId) {
    const request = { type: 'initialize', message: graph.getNode(nodeId).data, };
    ws.send(JSON.stringify(request));
  }

  // Handle messages from clients
  ws.on('message', async (message) => {
    const data = JSON.parse(message)

    switch (data.type) {
      case 'inferenceRequest':
        const text = data.message;
        const tokenized = await tokenizer(text)
        const seqLen = tokenized.input_ids.data.length;
        const input = {
          input_ids: Array.from(tokenized.input_ids.data),
          attention_mask: Array.from(tokenized.attention_mask.data),
          position_ids: [...Array(seqLen).keys()].map(x => BigInt(x)),
        };
        console.log(input)
        const client = graph.getWorker(graph.startNodeId);
        const request = { type: 'computation', message: { nodeId: graph.startNodeId, requestId: uuidv4(), data: input } };
        // Need to handle bigint specially
        clients[client].send(JSON.stringify(request, (_, value) => typeof value === 'bigint' ? value.toString() : value));
        break;
      case 'computationResult':
        const result = data.message;
        const nextNodes = graph.getNextNodes(result.nodeId);
        if (nextNodes.length == 0) {
          console.log('Final result: ', result);
          const logits = result.data.logits.data.slice(-vocabSize);
          const nextToken = argMax(logits);
          console.log(`Generated token: ${tokenizer.decode([nextToken])}`);
        } else {
          console.log('Intermediate result: ', result);
          nextNodes.forEach(node => {
            const client = graph.getWorker(node.id);
            const request = { type: 'computation', message: { nodeId: node.id, requestId: result.requestId, data: result.data } };
            clients[client].send(JSON.stringify(request));
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
