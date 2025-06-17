document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const sendIcon = document.getElementById('send-icon');
    const loadingIcon = document.getElementById('loading-icon');

    const API_URL = 'https://hate-chat-backend.onrender.com'; 
    let isServerReady = false;

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
            const lang = result?.source_language ? result.source_language.toUpperCase() : 'N/A';
            const translatedText = result?.translated_text || '';
            const score = result?.score ? (result.score * 100).toFixed(2) : '0.00';

            messageDiv.innerHTML = `
                <div class="font-medium text-gray-700">Moderation Result</div>
                ${result?.source_language && result.source_language !== 'en' ? `
                    <div class="text-gray-600 mb-1">Detected: ${lang}</div>
                    <div class="text-gray-600 mb-2">Translated: ${translatedText}</div>
                ` : ''}
                <div class="flex items-center">
                    <span class="font-medium">Status:</span>
                    <span class="ml-2 px-2 py-1 rounded text-sm ${
                        result.status === 'clean' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }">
                        ${result.status === 'clean' ? '✅ Clean' : '🚫 Inappropriate'}
                    </span>
                </div>
                <div class="text-gray-600 mt-1">Confidence: ${score}%</div>
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
                setTimeout(checkServerHealth, 3000); // Retry in 3 seconds
            }
        } catch (error) {
            addMessage('Cannot connect to server', false, null, true);
            setTimeout(checkServerHealth, 3000);
        }
    }

    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || !isServerReady) return;

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

            const result = await response.json();

            if (!response.ok || !result || !result.status) {
                const errorMsg = result?.error || result?.detail || 'Unexpected server response';
                throw new Error(errorMsg);
            }

            addMessage('Moderation result', false, result);
        } catch (error) {
            addMessage(error.message, false, null, true);
        } finally {
            sendIcon.classList.remove('hidden');
            loadingIcon.classList.add('hidden');
            sendButton.disabled = false;
            messageInput.disabled = false;
            messageInput.value = '';
            messageInput.focus();
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    messageInput.focus();
});
