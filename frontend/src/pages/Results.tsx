import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/store/authStore';
import { usePlaylistStore } from '@/store/playlistStore';
import { playlistApi } from '@/api/playlist';
import { Music, ArrowLeft, Save, ExternalLink, Loader2, X } from 'lucide-react';
import { Track } from '@/types';

const EMOTION_COLORS: Record<string, string> = {
  joy: 'bg-yellow-100 text-yellow-800',
  sadness: 'bg-blue-100 text-blue-800',
  anger: 'bg-red-100 text-red-800',
  fear: 'bg-purple-100 text-purple-800',
  surprise: 'bg-pink-100 text-pink-800',
  neutral: 'bg-gray-100 text-gray-800',
};

export default function Results() {
  const navigate = useNavigate();
  const { accessToken, userProfile } = useAuthStore();
  const { tracks, resolved, keywordsUsed, reset } = usePlaylistStore();
  
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [playlistName, setPlaylistName] = useState('');
  const [playlistDescription, setPlaylistDescription] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [savedUrl, setSavedUrl] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  if (!tracks || tracks.length === 0) {
    navigate('/generate');
    return null;
  }

  const handleSave = async () => {
    if (!accessToken || !playlistName.trim()) return;

    setIsSaving(true);
    setSaveError(null);

    try {
      // Format track URIs as "track_name|artist" for backend to resolve
      const trackUris = tracks.map((t) => `${t.track_name}|${t.artist}`);

      const result = await playlistApi.save({
        access_token: accessToken,
        name: playlistName.trim(),
        description: playlistDescription.trim(),
        track_uris: trackUris,
        public: false,
      });

      setSavedUrl(result.playlist_url);
    } catch (err: any) {
      console.error('Save error:', err);
      setSaveError(err.response?.data?.detail || err.message || 'Failed to save playlist');
    } finally {
      setIsSaving(false);
    }
  };

  const handleStartOver = () => {
    reset();
    navigate('/generate');
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
        </div>
      </nav>

      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={handleStartOver}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Start Over
          </button>
          <h1 className="text-3xl font-bold mb-2">Your Playlist</h1>
          <p className="text-gray-600">{tracks.length} tracks generated</p>
        </div>

        {/* Vibe Summary */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Vibe Summary</h2>
          
          {/* Keywords */}
          <div className="mb-4">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Keywords</h3>
            <div className="flex flex-wrap gap-2">
              {keywordsUsed.map((keyword) => (
                <span
                  key={keyword}
                  className="px-3 py-1 bg-primary/10 text-primary text-sm rounded-full"
                >
                  {keyword}
                </span>
              ))}
            </div>
          </div>

          {/* Emotions */}
          {resolved && resolved.emotions.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Detected Emotions</h3>
              <div className="flex flex-wrap gap-2">
                {resolved.emotions.slice(0, 3).map((emotion) => (
                  <span
                    key={emotion}
                    className={`px-3 py-1 text-sm rounded-full ${
                      EMOTION_COLORS[emotion] || 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    {emotion} ({(resolved.emotion_weights[emotion] * 100).toFixed(0)}%)
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Location Terms */}
          {resolved && resolved.location_terms.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-700 mb-2">Location Terms</h3>
              <div className="flex flex-wrap gap-2">
                {resolved.location_terms.map((term) => (
                  <span
                    key={term}
                    className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
                  >
                    {term}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Track List */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Tracks</h2>
          <div className="space-y-3">
            {tracks.map((track, index) => (
              <TrackCard key={index} track={track} index={index} />
            ))}
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-4">
          <button
            onClick={() => {
              setPlaylistName(`${keywordsUsed.join(', ')} Vibes`);
              setPlaylistDescription(`Generated from keywords: ${keywordsUsed.join(', ')}`);
              setShowSaveModal(true);
            }}
            className="flex-1 bg-primary hover:bg-primary/90 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Save className="w-5 h-5" />
            Save to Spotify
          </button>
          <button
            onClick={handleStartOver}
            className="flex-1 bg-gray-200 hover:bg-gray-300 text-gray-900 font-semibold py-3 px-6 rounded-lg transition-colors"
          >
            Start Over
          </button>
        </div>
      </div>

      {/* Save Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-6">
            {savedUrl ? (
              <div className="text-center">
                <div className="text-green-500 text-5xl mb-4">✓</div>
                <h2 className="text-2xl font-bold mb-2">Playlist Saved!</h2>
                <p className="text-gray-600 mb-6">Your playlist has been created on Spotify</p>
                <a
                  href={savedUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 bg-primary hover:bg-primary/90 text-white font-semibold py-2 px-6 rounded-lg transition-colors mb-4"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open in Spotify
                </a>
                <button
                  onClick={() => {
                    setShowSaveModal(false);
                    setSavedUrl(null);
                  }}
                  className="block w-full text-gray-600 hover:text-gray-900"
                >
                  Close
                </button>
              </div>
            ) : (
              <>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold">Save Playlist</h2>
                  <button
                    onClick={() => setShowSaveModal(false)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Playlist Name
                    </label>
                    <input
                      type="text"
                      value={playlistName}
                      onChange={(e) => setPlaylistName(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                      placeholder="My Travel Playlist"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Description (optional)
                    </label>
                    <textarea
                      value={playlistDescription}
                      onChange={(e) => setPlaylistDescription(e.target.value)}
                      rows={3}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                      placeholder="A playlist for my upcoming trip..."
                    />
                  </div>
                  {saveError && (
                    <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                      {saveError}
                    </div>
                  )}
                  <button
                    onClick={handleSave}
                    disabled={!playlistName.trim() || isSaving}
                    className="w-full bg-primary hover:bg-primary/90 text-white font-semibold py-2 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isSaving ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="w-4 h-4" />
                        Save to Spotify
                      </>
                    )}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function TrackCard({ track, index }: { track: Track; index: number }) {
  const emotionColor = EMOTION_COLORS[track.emotion] || 'bg-gray-100 text-gray-800';
  
  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium text-gray-500">#{index + 1}</span>
            <h3 className="font-semibold text-lg">{track.track_name}</h3>
          </div>
          <p className="text-gray-600">{track.artist}</p>
          <div className="flex items-center gap-2 mt-2">
            <span className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded">
              {track.genre}
            </span>
            <span className={`text-xs px-2 py-1 rounded ${emotionColor}`}>
              {track.emotion}
            </span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-primary">
            {(track.score * 100).toFixed(0)}
          </div>
          <div className="text-xs text-gray-500">score</div>
        </div>
      </div>
      
      {/* Score Breakdown */}
      <div className="mt-3 space-y-1">
        <ScoreBar label="Lyrics" value={track.score_lyrics} color="bg-blue-500" />
        <ScoreBar label="Emotion" value={track.score_emotion} color="bg-purple-500" />
        <ScoreBar label="Audio" value={track.score_audio} color="bg-green-500" />
        <ScoreBar label="Cluster" value={track.score_cluster} color="bg-orange-500" />
      </div>
    </div>
  );
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-600 w-16">{label}</span>
      <div className="flex-1 bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full ${color} transition-all`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
      <span className="text-xs text-gray-500 w-8 text-right">
        {(value * 100).toFixed(0)}
      </span>
    </div>
  );
}
