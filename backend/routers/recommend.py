"""
Recommend router - Playlist generation endpoints.
"""
from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException

from backend import state
from backend.models import (
    RecommendFromKeywordsRequest,
    RecommendFromDescriptionRequest,
    RecommendFromUrlRequest,
    RecommendResponse,
    Track,
    ResolvedKeywords,
)
from recommendation import recommender, keyword_embedder
from Airbnb import nlp_pipeline

router = APIRouter()


def _build_response(
    tracks_df,
    keywords: list[str],
    resolved: dict,
) -> RecommendResponse:
    """
    Convert recommender output DataFrame to API response format.
    """
    tracks = []
    for _, row in tracks_df.iterrows():
        tracks.append(Track(
            track_name=row["track_name"],
            artist=row["artist"],
            genre=row["genre"],
            emotion=row["emotion"],
            cluster=int(row["cluster"]),
            score=float(row["score"]),
            score_lyrics=float(row["score_lyrics"]),
            score_emotion=float(row["score_emotion"]),
            score_audio=float(row["score_audio"]),
            score_cluster=float(row["score_cluster"]),
            danceability=float(row["danceability"]),
            energy=float(row["energy"]),
            loudness=float(row["loudness"]),
            speechiness=float(row["speechiness"]),
            acousticness=float(row["acousticness"]),
            instrumentalness=float(row["instrumentalness"]),
            liveness=float(row["liveness"]),
            valence=float(row["valence"]),
            tempo=float(row["tempo"]),
            duration_s=float(row["duration_s"]),
            popularity=float(row["popularity"]),
        ))
    
    return RecommendResponse(
        tracks=tracks,
        resolved=ResolvedKeywords(
            emotions=resolved["emotions"],
            emotion_weights=resolved["emotion_weights"],
            audio_target=resolved["audio_target"],
            location_terms=resolved["location_terms"],
        ),
        keywords_used=keywords,
    )


@router.post("/from-keywords", response_model=RecommendResponse)
async def recommend_from_keywords(request: RecommendFromKeywordsRequest):
    """
    Generate playlist recommendations from a pre-formed keyword list.
    """
    if state.filtered_df is None or state.scaled_df is None:
        raise HTTPException(status_code=503, detail="Backend not ready")
    
    if not request.keywords:
        raise HTTPException(status_code=400, detail="Keywords list cannot be empty")
    
    try:
        weights = request.weights.model_dump() if request.weights else None
        
        # Get resolved keywords for response
        resolved = keyword_embedder.resolve_keywords(request.keywords)
        
        # Generate recommendations
        tracks_df = recommender.recommend(
            keywords=request.keywords,
            df=state.filtered_df,
            scaled_df=state.scaled_df,
            top_n=request.top_n,
            weights=weights,
        )
        
        return _build_response(tracks_df, request.keywords, resolved)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-description", response_model=RecommendResponse)
async def recommend_from_description(request: RecommendFromDescriptionRequest):
    """
    Generate playlist recommendations from raw listing description text.
    
    Extracts keywords via NLP pipeline, then passes to recommender.
    """
    if state.filtered_df is None or state.scaled_df is None:
        raise HTTPException(status_code=503, detail="Backend not ready")
    
    if not request.description or not request.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")
    
    try:
        # Extract keywords from description
        keywords = nlp_pipeline.extract_vibe_keywords(request.description, top_n=10)
        
        if not keywords:
            raise HTTPException(
                status_code=400,
                detail="Could not extract meaningful keywords from description"
            )
        
        # Get resolved keywords for response
        resolved = keyword_embedder.resolve_keywords(keywords)
        
        # Generate recommendations
        weights = request.weights.model_dump() if request.weights else None
        tracks_df = recommender.recommend(
            keywords=keywords,
            df=state.filtered_df,
            scaled_df=state.scaled_df,
            top_n=request.top_n,
            weights=weights,
        )
        
        return _build_response(tracks_df, keywords, resolved)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-url", response_model=RecommendResponse)
async def recommend_from_url(request: RecommendFromUrlRequest):
    """
    Generate playlist recommendations from an Airbnb listing URL.
    
    Looks up the listing in the local dataset and extracts keywords.
    """
    if state.filtered_df is None or state.scaled_df is None:
        raise HTTPException(status_code=503, detail="Backend not ready")
    
    if not request.url or not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    try:
        airbnb_dataset_path = os.getenv("AIRBNB_DATASET_PATH")
        if not airbnb_dataset_path:
            raise HTTPException(
                status_code=500,
                detail="AIRBNB_DATASET_PATH environment variable not configured"
            )
        
        nrc_path = os.getenv("NRC_PATH")
        if not nrc_path:
            raise HTTPException(
                status_code=500,
                detail="NRC_PATH environment variable not configured"
            )
        
        # Analyze listing from URL
        keyword_json, emotion_json = nlp_pipeline.analyze_listing_from_url(
            url=request.url,
            dataset_path=airbnb_dataset_path,
            nrc_path=nrc_path,
        )
        
        keywords = keyword_json.get("keywords", [])
        if not keywords:
            raise HTTPException(
                status_code=400,
                detail="Could not extract keywords from listing"
            )
        
        # Get resolved keywords for response
        resolved = keyword_embedder.resolve_keywords(keywords)
        
        # Generate recommendations
        weights = request.weights.model_dump() if request.weights else None
        tracks_df = recommender.recommend(
            keywords=keywords,
            df=state.filtered_df,
            scaled_df=state.scaled_df,
            top_n=request.top_n,
            weights=weights,
        )
        
        return _build_response(tracks_df, keywords, resolved)
    
    except ValueError as e:
        # URL not found in dataset
        if "URL not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
