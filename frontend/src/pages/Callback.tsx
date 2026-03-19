import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { authApi } from '@/api/auth';
import { userApi } from '@/api/user';
import { useAuthStore } from '@/store/authStore';
import { Loader2 } from 'lucide-react';

export default function Callback() {
  const navigate = useNavigate();
  const { setTokens, setUserProfile } = useAuthStore();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let executed = false;
    
    const handleCallback = async () => {
      if (executed) return;
      executed = true;
      
      try {
        // Parse URL parameters
        const params = new URLSearchParams(window.location.search);
        const code = params.get('code');
        const state = params.get('state');

        if (!code || !state) {
          throw new Error('Missing authorization code or state');
        }

        // Retrieve stored state and verifier
        const expectedState = sessionStorage.getItem('spotify_auth_state');
        const codeVerifier = sessionStorage.getItem('spotify_code_verifier');

        if (!expectedState || !codeVerifier) {
          throw new Error('Missing stored authentication data');
        }

        // Exchange code for tokens
        const tokens = await authApi.callback({
          code,
          state,
          expected_state: expectedState,
          code_verifier: codeVerifier,
        });

        // Store tokens
        setTokens(tokens);

        // Fetch user profile
        const profile = await userApi.getProfile(tokens.access_token);
        setUserProfile(profile);

        // Clean up sessionStorage
        sessionStorage.removeItem('spotify_auth_state');
        sessionStorage.removeItem('spotify_code_verifier');

        // Redirect to generate page
        navigate('/generate');
      } catch (err) {
        console.error('Callback error:', err);
        setError(err instanceof Error ? err.message : 'Authentication failed');
      }
    };

    handleCallback();
  }, []);

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-50 to-orange-50 flex items-center justify-center p-4">
        <div className="bg-white p-8 rounded-xl shadow-lg max-w-md w-full text-center">
          <div className="text-red-500 text-5xl mb-4">⚠️</div>
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Authentication Failed</h1>
          <p className="text-gray-600 mb-6">{error}</p>
          <a
            href="/"
            className="inline-block bg-primary hover:bg-primary/90 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
          >
            Back to Home
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-purple-50 flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-16 h-16 text-primary animate-spin mx-auto mb-4" />
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Completing authentication...</h2>
        <p className="text-gray-600">Please wait while we set up your account</p>
      </div>
    </div>
  );
}
