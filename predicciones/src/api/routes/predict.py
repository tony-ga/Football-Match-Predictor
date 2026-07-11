from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ...ingestion.schemas import MatchInput
from ...pipeline.predict import predict_match_pipeline

router = APIRouter()

@router.post("/predict", response_model=Dict[str, Any])
def predict_match(match: MatchInput, refresh_data: bool = False):
    try:
        home_team = match.metadata.home_team
        away_team = match.metadata.away_team
        match_date = match.metadata.date
        neutral_venue = match.metadata.neutral_venue
        
        response = predict_match_pipeline(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            neutral_venue=neutral_venue,
            refresh_data=refresh_data
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

