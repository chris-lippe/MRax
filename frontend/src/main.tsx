import { createRoot } from 'react-dom/client';
import { App } from './App';
import { FileSystemProvider } from './FileSystemDialog';
import './styles.css';

createRoot(document.getElementById('root')!).render(
  <FileSystemProvider>
    <App />
  </FileSystemProvider>,
);
