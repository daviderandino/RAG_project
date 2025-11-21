import { FaRobot, FaGithub } from 'react-icons/fa';

const Header = () => {
  return (
    <header className="header">
      <div className="logo-area">
        <FaRobot className="logo-icon" />
        <h1>RAG Assistant <span className="tag">Beta</span></h1>
      </div>
      <a href="#" className="github-link">
        <FaGithub /> Repo
      </a>
    </header>
  );
};

export default Header;