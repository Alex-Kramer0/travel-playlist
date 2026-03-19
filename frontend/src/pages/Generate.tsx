import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { recommendApi } from '@/api/recommend';
import { userApi } from '@/api/user';
import { useAuthStore } from '@/store/authStore';
import { usePlaylistStore } from '@/store/playlistStore';
import { RecommendWeights } from '@/types';
import { Music, Link as LinkIcon, FileText, Hash, Loader2, LogOut, Sliders } from 'lucide-react';

type InputMode = 'url' | 'description' | 'keywords';

export default function Generate() {
  const navigate = useNavigate();
  const { accessToken, userProfile, logout } = useAuthStore();
  const { setResult, setLoading } = usePlaylistStore();
  
  const [inputMode, setInputMode] = useState<InputMode>('url');
  const [urlInput, setUrlInput] = useState('');
  const [descriptionInput, setDescriptionInput] = useState('');
  const [keywordsInput, setKeywordsInput] = useState('');
  const [topN, setTopN] = useState(20);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [weights, setWeights] = useState<RecommendWeights>({
    lyrics: 0.35,
    emotion: 0.25,
    audio: 0.25,
    cluster: 0.15,
  });
  const [genres, setGenres] = useState<[string, number][]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchGenres = async () => {
      if (accessToken) {
        try {
          const response = await userApi.getGenres(accessToken);
          setGenres(response.genres);
        } catch (err) {
          console.error('Failed to fetch genres:', err);
        }
      }
    };
    fetchGenres();
  }, [accessToken]);

  const handleWeightChange = (key: keyof RecommendWeights, value: number) => {
    setWeights((prev) => ({ ...prev, [key]: value }));
  };

  const normalizeWeights = () => {
    const total = Object.values(weights).reduce((sum, val) => sum + val, 0);
    if (total === 0) return;
    const normalized = Object.fromEntries(
      Object.entries(weights).map(([key, val]) => [key, val / total])
    ) as RecommendWeights;
    setWeights(normalized);
  };

  const handleSubmit = async () => {
    setError(null);
    setIsSubmitting(true);
    setLoading(true);

    try {
      normalizeWeights();
      let result;

      if (inputMode === 'url') {
        if (!urlInput.trim()) {
          throw new Error('Please enter an Airbnb URL');
        }
        result = await recommendApi.fromUrl({
          url: urlInput.trim(),
          top_n: topN,
          weights: showAdvanced ? weights : undefined,
        });
      } else if (inputMode === 'description') {
        if (!descriptionInput.trim()) {
          throw new Error('Please enter a description');
        }
        result = await recommendApi.fromDescription({
          description: descriptionInput.trim(),
          top_n: topN,
          weights: showAdvanced ? weights : undefined,
        });
      } else {
        const keywords = keywordsInput
          .split(',')
          .map((k) => k.trim())
          .filter((k) => k.length > 0);
        if (keywords.length === 0) {
          throw new Error('Please enter at least one keyword');
        }
        result = await recommendApi.fromKeywords({
          keywords,
          top_n: topN,
          weights: showAdvanced ? weights : undefined,
        });
      }

      setResult(result);
      navigate('/results');
    } catch (err: any) {
      console.error('Generation error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to generate playlist');
      setLoading(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  const isInputValid = () => {
    if (inputMode === 'url') return urlInput.trim().length > 0;
    if (inputMode === 'description') return descriptionInput.trim().length > 0;
    return keywordsInput.trim().length > 0;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-purple-50">
      {/* Navbar */}
      <nav className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Music className="w-6 h-6 text-primary" />
            <span className="font-bold text-xl">Travel Playlist</span>
          </div>
          <div className="flex items-center gap-4">
            {userProfile && (
              <div className="flex items-center gap-2">
                {userProfile.images[0] && (
                  <img
                    src={userProfile.images[0].url}
                    alt={userProfile.display_name}
                    className="w-8 h-8 rounded-full"
                  />
                )}
                <span className="text-sm font-medium">{userProfile.display_name}</span>
              </div>
            )}
            <button
              onClick={logout}
              className="text-sm text-gray-600 hover:text-gray-900 flex items-center gap-1"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>
      </nav>

      <div className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Left Panel - Input */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold mb-6">Generate Your Playlist</h2>

              {/* Input Mode Tabs */}
              <div className="flex gap-2 mb-6 border-b">
                <button
                  onClick={() => setInputMode('url')}
                  className={`pb-3 px-4 font-medium transition-colors flex items-center gap-2 ${
                    inputMode === 'url'
                      ? 'border-b-2 border-primary text-primary'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <LinkIcon className="w-4 h-4" />
                  Airbnb URL
                </button>
                <button
                  onClick={() => setInputMode('description')}
                  className={`pb-3 px-4 font-medium transition-colors flex items-center gap-2 ${
                    inputMode === 'description'
                      ? 'border-b-2 border-primary text-primary'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <FileText className="w-4 h-4" />
                  Description
                </button>
                <button
                  onClick={() => setInputMode('keywords')}
                  className={`pb-3 px-4 font-medium transition-colors flex items-center gap-2 ${
                    inputMode === 'keywords'
                      ? 'border-b-2 border-primary text-primary'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <Hash className="w-4 h-4" />
                  Keywords
                </button>
              </div>

              {/* Input Fields */}
              <div className="mb-6">
                {inputMode === 'url' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Airbnb Listing URL
                    </label>
                    <input
                      type="text"
                      value={urlInput}
                      onChange={(e) => setUrlInput(e.target.value)}
                      placeholder="https://www.airbnb.com/rooms/..."
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                    />
                  </div>
                )}
                {inputMode === 'description' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Listing Description
                    </label>
                    <textarea
                      value={descriptionInput}
                      onChange={(e) => setDescriptionInput(e.target.value)}
                      placeholder="Paste the listing description here..."
                      rows={6}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                    />
                  </div>
                )}
                {inputMode === 'keywords' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Keywords (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={keywordsInput}
                      onChange={(e) => setKeywordsInput(e.target.value)}
                      placeholder="cozy, mountain view, fireplace, peaceful"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                    />
                  </div>
                )}
              </div>

              {/* Top N Slider */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of tracks: {topN}
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={topN}
                  onChange={(e) => setTopN(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Advanced Settings */}
              <div className="mb-6">
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-sm font-medium text-gray-700 hover:text-gray-900"
                >
                  <Sliders className="w-4 h-4" />
                  Advanced: Tune Weights
                  <span className="text-xs text-gray-500">
                    {showAdvanced ? '▼' : '▶'}
                  </span>
                </button>
                {showAdvanced && (
                  <div className="mt-4 space-y-4 p-4 bg-gray-50 rounded-lg">
                    {Object.entries(weights).map(([key, value]) => (
                      <div key={key}>
                        <label className="block text-sm font-medium text-gray-700 mb-1 capitalize">
                          {key}: {value.toFixed(2)}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={value}
                          onChange={(e) =>
                            handleWeightChange(key as keyof RecommendWeights, parseFloat(e.target.value))
                          }
                          className="w-full"
                        />
                      </div>
                    ))}
                    <p className="text-xs text-gray-500">
                      Weights will be normalized to sum to 1.0
                    </p>
                  </div>
                )}
              </div>

              {/* Error Display */}
              {error && (
                <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                  {error}
                </div>
              )}

              {/* Submit Button */}
              <button
                onClick={handleSubmit}
                disabled={!isInputValid() || isSubmitting}
                className="w-full bg-primary hover:bg-primary/90 text-white font-semibold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Music className="w-5 h-5" />
                    Generate Playlist
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Right Panel - Context */}
          <div className="space-y-6">
            {/* User Genres */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="font-semibold text-lg mb-4">Your Top Genres</h3>
              {genres.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {genres.map(([genre, count]) => (
                    <span
                      key={genre}
                      className="px-3 py-1 bg-primary/10 text-primary text-sm rounded-full"
                    >
                      {genre} ({count})
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">Loading your preferences...</p>
              )}
            </div>

            {/* How It Works */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="font-semibold text-lg mb-4">How It Works</h3>
              <ol className="space-y-3 text-sm text-gray-600">
                <li className="flex gap-2">
                  <span className="font-semibold text-primary">1.</span>
                  <span>We extract keywords and emotions from your input</span>
                </li>
                <li className="flex gap-2">
                  <span className="font-semibold text-primary">2.</span>
                  <span>AI maps keywords to audio features and emotions</span>
                </li>
                <li className="flex gap-2">
                  <span className="font-semibold text-primary">3.</span>
                  <span>Tracks are scored across 4 layers: lyrics, emotion, audio, cluster</span>
                </li>
                <li className="flex gap-2">
                  <span className="font-semibold text-primary">4.</span>
                  <span>Top matches are ranked and ready to save to Spotify</span>
                </li>
              </ol>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
