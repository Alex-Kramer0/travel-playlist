import { apiClient } from './client';
import { SpotifyToken } from '@/types';

export interface AuthStartResponse {
  authorization_url: string;
  state: string;
  code_verifier: string;
}

export interface AuthCallbackRequest {
  code: string;
  state: string;
  expected_state: string;
  code_verifier: string;
}

export const authApi = {
  async start(): Promise<AuthStartResponse> {
    const response = await apiClient.get<AuthStartResponse>('/api/auth/start');
    return response.data;
  },

  async callback(request: AuthCallbackRequest): Promise<SpotifyToken> {
    const response = await apiClient.post<SpotifyToken>('/api/auth/callback', request);
    return response.data;
  },
};
