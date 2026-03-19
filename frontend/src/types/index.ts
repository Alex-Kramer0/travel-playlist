export interface Track {
  track_name: string;
  artist: string;
  genre: string;
  emotion: string;
  cluster: number;
  score: number;
  score_lyrics: number;
  score_emotion: number;
  score_audio: number;
  score_cluster: number;
  danceability: number;
  energy: number;
  loudness: number;
  speechiness: number;
  acousticness: number;
  instrumentalness: number;
  liveness: number;
  valence: number;
  tempo: number;
  duration_s: number;
  popularity: number;
}

export interface ResolvedKeywords {
  emotions: string[];
  emotion_weights: Record<string, number>;
  audio_target: Record<string, number>;
  location_terms: string[];
}

export interface RecommendResponse {
  tracks: Track[];
  resolved: ResolvedKeywords;
  keywords_used: string[];
}

export interface SpotifyToken {
  access_token: string;
  refresh_token: string | null;
  expires_at: number;
  scope: string;
  token_type: string;
}

export interface SpotifyUserProfile {
  id: string;
  display_name: string;
  email: string;
  images: { url: string }[];
}

export interface RecommendWeights {
  lyrics: number;
  emotion: number;
  audio: number;
  cluster: number;
}
