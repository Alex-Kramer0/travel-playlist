import { create } from 'zustand';
import { SpotifyToken, SpotifyUserProfile } from '@/types';

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  expiresAt: number | null;
  userProfile: SpotifyUserProfile | null;
  isGuest: boolean;
  setTokens: (tokens: SpotifyToken) => void;
  setUserProfile: (profile: SpotifyUserProfile) => void;
  setGuest: () => void;
  logout: () => void;
  isAuthenticated: () => boolean;
  canAccess: () => boolean;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  accessToken: null,
  refreshToken: null,
  expiresAt: null,
  userProfile: null,
  isGuest: false,

  setTokens: (tokens: SpotifyToken) => {
    set({
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token,
      expiresAt: tokens.expires_at,
      isGuest: false,
    });
    
    // Persist to sessionStorage
    sessionStorage.setItem('spotify_token', JSON.stringify(tokens));
    sessionStorage.removeItem('guest_mode');
  },

  setUserProfile: (profile: SpotifyUserProfile) => {
    set({ userProfile: profile });
    sessionStorage.setItem('spotify_profile', JSON.stringify(profile));
  },

  setGuest: () => {
    set({ isGuest: true });
    sessionStorage.setItem('guest_mode', 'true');
  },

  logout: () => {
    set({
      accessToken: null,
      refreshToken: null,
      expiresAt: null,
      userProfile: null,
      isGuest: false,
    });
    sessionStorage.removeItem('spotify_token');
    sessionStorage.removeItem('spotify_profile');
    sessionStorage.removeItem('guest_mode');
  },

  isAuthenticated: () => {
    const state = get();
    if (!state.accessToken || !state.expiresAt) return false;
    return Date.now() / 1000 < state.expiresAt;
  },

  canAccess: () => {
    const state = get();
    return state.isGuest || state.isAuthenticated();
  },
}));

// Rehydrate from sessionStorage on load
export const rehydrateAuth = () => {
  const tokenStr = sessionStorage.getItem('spotify_token');
  const profileStr = sessionStorage.getItem('spotify_profile');
  
  if (tokenStr) {
    try {
      const token = JSON.parse(tokenStr) as SpotifyToken;
      useAuthStore.getState().setTokens(token);
    } catch (e) {
      console.error('Failed to rehydrate auth token', e);
    }
  }
  
  if (profileStr) {
    try {
      const profile = JSON.parse(profileStr) as SpotifyUserProfile;
      useAuthStore.getState().setUserProfile(profile);
    } catch (e) {
      console.error('Failed to rehydrate user profile', e);
    }
  }

  // Rehydrate guest mode
  if (sessionStorage.getItem('guest_mode') === 'true') {
    useAuthStore.setState({ isGuest: true });
  }
};
