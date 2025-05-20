import { InferenceSession } from './onnx_session'

document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const messageForm = document.getElementById('message-form');
    const messageInput = document.getElementById('message-input');
    const connectionStatus = document.getElementById('connection-status');
    const connectedUsers = document.getElementById('connected-users');

    let session = undefined;

    // Connect to WebSocket server
    const wsUrl = `ws://localhost:3000/ws`;
    const socket = new WebSocket(wsUrl);

    // Handle connection open
    socket.addEventListener('open', () => {
        connectionStatus.textContent = 'Connected';
        connectionStatus.style.color = 'green';

        // Enable the message form
        messageForm.querySelector('button').disabled = false;
        messageInput.disabled = false;
    });

    // Handle connection close
    socket.addEventListener('close', () => {
        connectionStatus.textContent = 'Disconnected - Refresh to reconnect';
        connectionStatus.style.color = 'red';

        // Disable the message form
        messageForm.querySelector('button').disabled = true;
        messageInput.disabled = true;
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
            const data = JSON.parse(event.data);
            console.log(data);

            switch (data.type) {
                case 'initialize': {
                    console.log('Loading model: ', data.message);
                    session = await InferenceSession.createSession(data.message.model_uri, data.message.external_data);
                    const response = { type: 'initializeDone' };
                    socket.send(JSON.stringify(response));
                    console.log('Finished loading model');
                    break;
                }
                case 'connectedUsers':
                    connectedUsers.textContent = data.message;
                    break;
                case 'computation': {
                    console.log('Running computation: ', data);
                    const result = await session.runInference(data.message.requestId, data.message.data);
                    const response = {
                        type: 'computationResult',
                        message: {
                            nodeId: data.message.nodeId,
                            requestId: data.message.requestId,
                            data: result,
                        },
                    };
                    socket.send(JSON.stringify(response));
                    break;
                }
                case 'inferenceResult':
                    displayMessage(data.message);
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

        const messageObj = { type: 'inferenceRequest', message };
        socket.send(JSON.stringify(messageObj));

        // Clear input field
        messageInput.value = '';
    });

    // Display message in chat container
    function displayMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';
        messageDiv.textContent = message;

        // Add message to chat container
        chatContainer.appendChild(messageDiv);

        // Auto-scroll to the bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});