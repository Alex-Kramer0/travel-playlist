import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore, rehydrateAuth } from './store/authStore';
import Home from './pages/Home';
import Callback from './pages/Callback';
import Generate from './pages/Generate';
import Results from './pages/Results';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const canAccess = useAuthStore((state) => state.canAccess());
  
  if (!canAccess) {
    return <Navigate to="/" replace />;
  }
  
  return <>{children}</>;
}

function App() {
  useEffect(() => {
    rehydrateAuth();
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/callback" element={<Callback />} />
        <Route
          path="/generate"
          element={
            <ProtectedRoute>
              <Generate />
            </ProtectedRoute>
          }
        />
        <Route
          path="/results"
          element={
            <ProtectedRoute>
              <Results />
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
