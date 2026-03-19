import { create } from 'zustand';
import { Track, ResolvedKeywords } from '@/types';

interface PlaylistState {
  tracks: Track[];
  resolved: ResolvedKeywords | null;
  keywordsUsed: string[];
  isLoading: boolean;
  savedPlaylistUrl: string | null;
  setResult: (result: { tracks: Track[]; resolved: ResolvedKeywords; keywords_used: string[] }) => void;
  setLoading: (loading: boolean) => void;
  setSavedUrl: (url: string) => void;
  reset: () => void;
}

export const usePlaylistStore = create<PlaylistState>((set) => ({
  tracks: [],
  resolved: null,
  keywordsUsed: [],
  isLoading: false,
  savedPlaylistUrl: null,

  setResult: (result) => {
    set({
      tracks: result.tracks,
      resolved: result.resolved,
      keywordsUsed: result.keywords_used,
      isLoading: false,
    });
  },

  setLoading: (loading) => {
    set({ isLoading: loading });
  },

  setSavedUrl: (url) => {
    set({ savedPlaylistUrl: url });
  },

  reset: () => {
    set({
      tracks: [],
      resolved: null,
      keywordsUsed: [],
      isLoading: false,
      savedPlaylistUrl: null,
    });
  },
}));
