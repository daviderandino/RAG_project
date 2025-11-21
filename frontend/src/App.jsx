import { useState } from 'react';
import Header from './components/Header';
import FileUploader from './components/FileUploader';
import MessageList from './components/MessageList';
import ChatInput from './components/ChatInput';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleUploadSuccess = () => {
    setMessages(prev => [...prev, { 
      sender: 'bot', 
      text: 'Documento analizzato con successo! Sono pronto a rispondere alle tue domande.' 
    }]);
  };

  const handleSendMessage = async (text) => {
    const userMsg = { sender: 'user', text };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text }),
      });

      const data = await response.json();
      
      setMessages(prev => [...prev, { 
        sender: 'bot', 
        text: data.answer, 
        sources: data.sources 
      }]);
    } catch (error) {
      setMessages(prev => [...prev, { sender: 'bot', text: 'Errore di connessione.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-layout">
      <Header />
      
      <main className="main-content">
        <div className="top-bar">
          <FileUploader onUploadSuccess={handleUploadSuccess} />
        </div>

        <div className="chat-container">
          <MessageList messages={messages} isLoading={isLoading} />
          <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
        </div>
      </main>
    </div>
  );
}

export default App;