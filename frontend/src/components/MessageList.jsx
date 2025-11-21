import { FaUser, FaRobot, FaBookOpen } from 'react-icons/fa';
import { useEffect, useRef } from 'react';

const MessageList = ({ messages, isLoading }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="empty-state">
          <FaRobot size={50} color="#cbd5e1" />
          <p>Carica un PDF e inizia a chattare con i tuoi dati.</p>
        </div>
      )}

      {messages.map((msg, idx) => (
        <div key={idx} className={`message-row ${msg.sender}`}>
          <div className="avatar">
            {msg.sender === 'user' ? <FaUser /> : <FaRobot />}
          </div>
          <div className="bubble">
            <p>{msg.text}</p>
            
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources-container">
                <div className="sources-title"><FaBookOpen /> Fonti rilevate:</div>
                <div className="sources-list">
                  {msg.sources.map((src, i) => (
                    <span key={i} className="source-tag">{src}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="message-row bot">
          <div className="avatar"><FaRobot /></div>
          <div className="bubble loading">
            <div className="dot"></div><div className="dot"></div><div className="dot"></div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
};

export default MessageList;