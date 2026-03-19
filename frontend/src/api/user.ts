import { apiClient } from './client';
import { SpotifyUserProfile } from '@/types';

export interface UserGenresResponse {
  genres: [string, number][];
}

export const userApi = {
  async getProfile(accessToken: string): Promise<SpotifyUserProfile> {
    const response = await apiClient.get<SpotifyUserProfile>('/api/user/profile', {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
    return response.data;
  },

  async getGenres(accessToken: string, timeRange: string = 'medium_term'): Promise<UserGenresResponse> {
    const response = await apiClient.get<UserGenresResponse>('/api/user/genres', {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      params: {
        time_range: timeRange,
      },
    });
    return response.data;
  },
};
