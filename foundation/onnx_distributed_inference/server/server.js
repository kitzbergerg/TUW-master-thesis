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
const workerClients = {};

const active_requests = {};

// Handle WebSocket connections
wss.on('connection', (ws) => {
  const userUuid = uuidv4();
  console.log('Client connected');
  workerClients[userUuid] = ws;
  broadcastNumberOfConnectedUsers(workerClients);
  const nodeId = graph.addUser(userUuid);
  if (nodeId) {
    const request = { type: 'initialize', message: graph.getNode(nodeId).data, };
    ws.send(JSON.stringify(request));
  }

  // Handle messages from clients
  ws.on('message', async (message) => {
    const data = JSON.parse(message)

    switch (data.type) {
      case 'inferenceRequest': {
        const requestId = uuidv4();

        const text = data.message;
        const tokenized = await tokenizer(text)
        const seqLen = tokenized.input_ids.data.length;
        const input = {
          input_ids: Array.from(tokenized.input_ids.data),
          attention_mask: Array.from(tokenized.attention_mask.data),
          position_ids: [...Array(seqLen).keys()].map(x => BigInt(x)),
        };
        console.log('Input: ', input);
        const workerId = graph.getWorker(graph.startNodeId);
        active_requests[requestId] = { userId: userUuid, input_tokens: input.input_ids, generated_token: [] }
        const message = { type: 'computation', message: { nodeId: graph.startNodeId, requestId, data: input } };
        workerClients[workerId].send(JSON.stringify(message, (_, value) => typeof value === 'bigint' ? value.toString() : value));

        break;
      }
      case 'computationResult': {
        const result = data.message;
        const nextNodes = graph.getNextNodes(result.nodeId);
        if (nextNodes.length != 0) {
          nextNodes.forEach(node => {
            const workerId = graph.getWorker(node.id);
            const message = { type: 'computation', message: { nodeId: node.id, requestId: result.requestId, data: result.data } };
            workerClients[workerId].send(JSON.stringify(message));
          })
          return
        }

        // reached end node
        const logits = result.data.logits.data.slice(-vocabSize);
        const nextToken = argMax(logits);
        console.log(`Generated token: ${nextToken}, Decoded: ${tokenizer.decode([nextToken])}`);

        const active_request = active_requests[result.requestId];
        active_request.generated_token.push(nextToken)
        if (active_request.generated_token.length > 20) {
          const outputText = tokenizer.decode(active_request.generated_token);
          console.log(`Final result: ${outputText}`);
          workerClients[active_request.userId].send(JSON.stringify({ type: 'inferenceResult', message: outputText }));
          // TODO: think about caches for client, either keep them and allow long context or cleanup
          return
        }

        // generate next token
        const input = {
          input_ids: Array.from([BigInt(nextToken)]),
          attention_mask: Array.from([BigInt(1)]),
          position_ids: Array.from([BigInt(active_request.input_tokens.length + active_request.generated_token.length - 1)]),
        };
        console.log('Input: ', input);
        const client = graph.getWorker(graph.startNodeId);
        const request = { type: 'computation', message: { nodeId: graph.startNodeId, requestId: result.requestId, data: input } };
        workerClients[client].send(JSON.stringify(request, (_, value) => typeof value === 'bigint' ? value.toString() : value));

        break;
      }
      default:
        console.error('Unknown type');
    }
  });

  // Handle client disconnection
  ws.on('close', () => {
    console.log('Client disconnected');
    delete workerClients[userUuid];
    graph.removeUser(userUuid);

    broadcastNumberOfConnectedUsers(workerClients)
  });
});

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
