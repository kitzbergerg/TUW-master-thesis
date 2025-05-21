import { InferenceSession } from './src/onnx_session'
import { ComputationMessageSchema, FirstModelInput, IntermediateModelData, WebSocketMessageSchema } from "./src/gen/data_pb";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";

document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input') as HTMLInputElement;
    const connectionStatus = document.getElementById('connection-status');
    const connectedUsers = document.getElementById('connected-users');

    let session: InferenceSession = undefined;

    // Connect to WebSocket server
    const wsUrl = `ws://localhost:3000/ws`;
    const socket = new WebSocket(wsUrl);
    socket.binaryType = "arraybuffer";

    // Handle connection open
    socket.addEventListener('open', () => {
        connectionStatus.textContent = 'Connected';
        connectionStatus.style.color = 'green';

        // Enable the message form
        messageForm.querySelector('button').disabled = false;
    });

    // Handle connection close
    socket.addEventListener('close', () => {
        connectionStatus.textContent = 'Disconnected - Refresh to reconnect';
        connectionStatus.style.color = 'red';

        // Disable the message form
        messageForm.querySelector('button').disabled = true;
    });

    // Handle WebSocket errors
    socket.addEventListener('error', (error) => {
        console.error('WebSocket error:', error);
        connectionStatus.textContent = 'Connection error - See console for details';
        connectionStatus.style.color = 'red';
    });

    // Handle incoming messages
    socket.addEventListener('message', async (event) => {
        try {
            const data = fromBinary(WebSocketMessageSchema, new Uint8Array(event.data)).kind;

            switch (data.case) {
                case 'initialize': {
                    console.log('Loading model: ', data.value.message);
                    session = await InferenceSession.createSession(data.value.message.modelUri, data.value.message.externalData);
                    const response = create(WebSocketMessageSchema, {
                        kind: {
                            case: 'initializeDone',
                            value: {
                                nodeId: data.value.nodeId
                            }
                        }
                    });
                    socket.send(toBinary(WebSocketMessageSchema, response));
                    console.log('Finished loading model');
                    break;
                }
                case 'connectedUsers':
                    connectedUsers.textContent = data.value.message.toString();
                    break;
                case 'computation': {
                    console.log('Running computation: ', data);
                    const result = await session.runInference(data.value.message);
                    const response = create(WebSocketMessageSchema, {
                        kind: {
                            case: 'computation',
                            value: {
                                message: result
                            }
                        }
                    });
                    socket.send(toBinary(WebSocketMessageSchema, response));
                    break;
                }
                case 'inferenceResult':
                    displayMessage(data.value.message);
                    break;
                case 'invalidateCache':
                    session.invalidateCache(data.value.requestId);
                    break;
                default:
                    console.error('Unknown type');
            }
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    });

    // Handle form submission
    messageForm.addEventListener('submit', (event) => {
        event.preventDefault();

        const message = messageInput.value.trim();
        if (!message) return;

        const messageObj = create(WebSocketMessageSchema, {
            kind: {
                case: 'inferenceRequest',
                value: {
                    message
                }
            }
        });
        socket.send(toBinary(WebSocketMessageSchema, messageObj));

        // Clear input field
        messageInput.innerText = '';
    });

    // Display message in chat container
    function displayMessage(message: string) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';
        messageDiv.textContent = message;

        // Add message to chat container
        chatContainer.appendChild(messageDiv);

        // Auto-scroll to the bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});