document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const sendIcon = document.getElementById('send-icon');
    const loadingIcon = document.getElementById('loading-icon');

 const API_URL = 'https://hate-chat-backend.onrender.com'; 
    let isServerReady = false;

    // Check server health on startup
    checkServerHealth();

    function addMessage(text, isUser = true, result = null, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message p-4 rounded-lg ${
            isUser ? 'bg-indigo-50' : 'bg-gray-50'
        } ${
            result?.status === 'clean' ? 'clean-message' : 
            result?.status === 'inappropriate' ? 'inappropriate-message' : ''
        }`;

        if (isUser) {
            messageDiv.innerHTML = `
                <div class="font-medium text-indigo-700">You</div>
                <div class="text-gray-800">${text}</div>
            `;
        } else if (isError) {
            messageDiv.innerHTML = `
                <div class="font-medium text-red-700">Error</div>
                <div class="text-red-600">${text}</div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="font-medium text-gray-700">Moderation Result</div>
                ${result.source_language !== 'en' ? `
                    <div class="text-gray-600 mb-1">Detected: ${result.source_language.toUpperCase()}</div>
                    <div class="text-gray-600 mb-2">Translated: ${result.translated_text}</div>
                ` : ''}
                <div class="flex items-center">
                    <span class="font-medium">Status:</span>
                    <span class="ml-2 px-2 py-1 rounded text-sm ${
                        result.status === 'clean' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }">
                        ${result.status === 'clean' ? '✅ Clean' : '🚫 Inappropriate'}
                    </span>
                </div>
                <div class="text-gray-600 mt-1">Confidence: ${(result.score * 100).toFixed(2)}%</div>
            `;
        }

        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    async function checkServerHealth() {
        try {
            const response = await fetch(`${API_URL}/health`);
            const data = await response.json();
            isServerReady = data.status === 'ready';
            
            if (!isServerReady) {
                addMessage('Server is still loading models...', false, null, true);
                setTimeout(checkServerHealth, 3000); // Check again in 3 seconds
            }
        } catch (error) {
            addMessage('Cannot connect to server', false, null, true);
            setTimeout(checkServerHealth, 3000);
        }
    }

    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || !isServerReady) return;

        // UI Loading state
        sendIcon.classList.add('hidden');
        loadingIcon.classList.remove('hidden');
        sendButton.disabled = true;
        messageInput.disabled = true;

        addMessage(text, true);

        try {
            const response = await fetch(`${API_URL}/moderate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Moderation failed');
            }

            const result = await response.json();
            addMessage('Moderation result', false, result);
        } catch (error) {
            addMessage(error.message, false, null, true);
        } finally {
            // Reset UI
            sendIcon.classList.remove('hidden');
            loadingIcon.classList.add('hidden');
            sendButton.disabled = false;
            messageInput.disabled = false;
            messageInput.value = '';
            messageInput.focus();
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Initial focus
    messageInput.focus();
});
