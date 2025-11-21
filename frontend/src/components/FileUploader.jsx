import { useState } from 'react';
import { FaCloudUploadAlt, FaCheckCircle, FaExclamationCircle, FaSpinner } from 'react-icons/fa';

const FileUploader = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState('idle'); // idle, success, error

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setStatus('idle');
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (res.ok) {
        setStatus('success');
        onUploadSuccess(); 
      } else {
        setStatus('error');
      }
    } catch (error) {
      setStatus('error');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="uploader-container">
      <label className={`upload-btn ${uploading ? 'disabled' : ''}`}>
        {uploading ? <FaSpinner className="spin" /> : <FaCloudUploadAlt />}
        <span>{uploading ? 'Analisi in corso...' : 'Carica Documento PDF'}</span>
        <input type="file" accept=".pdf" onChange={handleFileChange} disabled={uploading} hidden />
      </label>

      {status === 'success' && (
        <div className="status-badge success">
          <FaCheckCircle /> Documento pronto
        </div>
      )}
      {status === 'error' && (
        <div className="status-badge error">
          <FaExclamationCircle /> Errore caricamento
        </div>
      )}
    </div>
  );
};

export default FileUploader;