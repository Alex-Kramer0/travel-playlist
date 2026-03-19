import { Music, Sparkles, MapPin } from 'lucide-react';
import { authApi } from '@/api/auth';
import { useState } from 'react';

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async () => {
    try {
      setIsLoading(true);
      const response = await authApi.start();
      
      // Store state and code_verifier in sessionStorage
      sessionStorage.setItem('spotify_auth_state', response.state);
      sessionStorage.setItem('spotify_code_verifier', response.code_verifier);
      
      // Redirect to Spotify authorization
      window.location.href = response.authorization_url;
    } catch (error) {
      console.error('Failed to start auth:', error);
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-purple-50">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center">
          {/* Hero Section */}
          <div className="mb-12">
            <div className="flex justify-center mb-6">
              <div className="bg-primary/10 p-6 rounded-full">
                <Music className="w-16 h-16 text-primary" />
              </div>
            </div>
            <h1 className="text-5xl font-bold text-gray-900 mb-4">
              Travel Playlist
            </h1>
            <p className="text-xl text-gray-600 mb-8">
              Your Airbnb stay deserves a soundtrack
            </p>
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <MapPin className="w-10 h-10 text-primary mx-auto mb-4" />
              <h3 className="font-semibold text-lg mb-2">Location-Based</h3>
              <p className="text-gray-600 text-sm">
                Paste your Airbnb listing URL and we'll analyze the vibe
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <Sparkles className="w-10 h-10 text-primary mx-auto mb-4" />
              <h3 className="font-semibold text-lg mb-2">AI-Powered</h3>
              <p className="text-gray-600 text-sm">
                Advanced NLP extracts emotions and keywords from descriptions
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <Music className="w-10 h-10 text-primary mx-auto mb-4" />
              <h3 className="font-semibold text-lg mb-2">Personalized</h3>
              <p className="text-gray-600 text-sm">
                Curated playlists based on your listening history
              </p>
            </div>
          </div>

          {/* CTA */}
          <div className="bg-white p-8 rounded-xl shadow-lg">
            <h2 className="text-2xl font-semibold mb-4">Get Started</h2>
            <p className="text-gray-600 mb-6">
              Connect your Spotify account to generate personalized playlists for your travels
            </p>
            <button
              onClick={handleLogin}
              disabled={isLoading}
              className="bg-primary hover:bg-primary/90 text-white font-semibold py-3 px-8 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Connecting...
                </>
              ) : (
                <>
                  <Music className="w-5 h-5" />
                  Connect with Spotify
                </>
              )}
            </button>
          </div>

          {/* Footer */}
          <div className="mt-12 text-sm text-gray-500">
            <p>
              Powered by Spotify API, NLP emotion analysis, and machine learning clustering
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
