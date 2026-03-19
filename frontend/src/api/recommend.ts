import { apiClient } from './client';
import { RecommendResponse, RecommendWeights } from '@/types';

export interface RecommendFromKeywordsRequest {
  keywords: string[];
  top_n?: number;
  weights?: RecommendWeights;
  user_top_artists?: string[];
}

export interface RecommendFromDescriptionRequest {
  description: string;
  top_n?: number;
  weights?: RecommendWeights;
  user_top_artists?: string[];
}

export interface RecommendFromUrlRequest {
  url: string;
  top_n?: number;
  weights?: RecommendWeights;
  user_top_artists?: string[];
}

export const recommendApi = {
  async fromKeywords(request: RecommendFromKeywordsRequest): Promise<RecommendResponse> {
    const response = await apiClient.post<RecommendResponse>('/api/recommend/from-keywords', request);
    return response.data;
  },

  async fromDescription(request: RecommendFromDescriptionRequest): Promise<RecommendResponse> {
    const response = await apiClient.post<RecommendResponse>('/api/recommend/from-description', request);
    return response.data;
  },

  async fromUrl(request: RecommendFromUrlRequest): Promise<RecommendResponse> {
    const response = await apiClient.post<RecommendResponse>('/api/recommend/from-url', request);
    return response.data;
  },
};
