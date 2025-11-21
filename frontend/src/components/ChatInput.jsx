import { useState } from 'react';
import { FaPaperPlane } from 'react-icons/fa';

const ChatInput = ({ onSendMessage, disabled }) => {
  const [input, setInput] = useState('');

  const send = () => {
    if (!input.trim()) return;
    onSendMessage(input);
    setInput('');
  };

  return (
    <div className="input-area">
      <div className="input-wrapper">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="Fai una domanda specifica..."
          disabled={disabled}
        />
        <button onClick={send} disabled={disabled || !input.trim()}>
          <FaPaperPlane />
        </button>
      </div>
    </div>
  );
};

export default ChatInput;