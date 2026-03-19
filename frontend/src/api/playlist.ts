import { apiClient } from './client';

export interface PlaylistSaveRequest {
  access_token: string;
  name: string;
  description?: string;
  track_uris: string[];
  public?: boolean;
}

export interface PlaylistSaveResponse {
  playlist_id: string;
  playlist_url: string;
  tracks_added: number;
}

export const playlistApi = {
  async save(request: PlaylistSaveRequest): Promise<PlaylistSaveResponse> {
    const response = await apiClient.post<PlaylistSaveResponse>('/api/playlist/save', request);
    return response.data;
  },
};
