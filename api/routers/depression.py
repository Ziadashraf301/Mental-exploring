"""
DEPRESSION DETECTION API Router
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import time
import logging

from core.schemas import DepressionRequest, DepressionResponse
from core.database import Database
from services import get_depression_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/depression", tags=["Depression Detection"])


# Dependency injection
def get_db():
    """Retrieve DB instance from main"""
    from main import db
    return db


@router.post("/predict", response_model=DepressionResponse)
async def predict_depression(
    request: DepressionRequest,
    db: Database = Depends(get_db),
):
    """
    Detect Depression in text.
    Request body:
      - text: str
      - user_id: optional
      - metadata: dict
    """
    
    if not request.text or not isinstance(request.text, str):
        raise HTTPException(status_code=400, detail="Text must be a non-empty string.")

    try:
        start_time = time.time()

        # User handling
        if not request.user_id:
            user_id = await db.create_user(name="Anonymous")
        else:
            user = await db.get_user(request.user_id)
            user_id = user.id if user else await db.create_user(name="Anonymous")

        # Load depression service
        depression_service = get_depression_service()

        # Run prediction
        result = depression_service.predict(request.text)

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Depression service failed")
            )

        inference_time = time.time() - start_time

        # Save prediction to DB
        prediction_id = await db.save_prediction(
            user_id=user_id,
            service_type="depression",
            input_text=request.text,
            input_length=len(request.text),
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities={
                "not_depressed": result["probability_not_depressed"],
                "depressed": result["probability_depressed"],
            },
            model_name=result["model_info"]["name"],
            model_version=result["model_info"].get("version", "1.0"),
            inference_time=inference_time,
        )

        # Build response
        return DepressionResponse(
            success=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            depression_probability=result["probability_depressed"],
            not_depression_probability=result["probability_not_depressed"],
            user_id=user_id,
            prediction_id=prediction_id,
            inference_time=inference_time,
            model_name=result["model_info"]["name"],
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Depression prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@router.get("/health")
async def depression_health_check():
    """Check model and tokenizer status"""
    try:
        service = get_depression_service()
        return service.health_check()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/model/info")
async def depression_model_info():
    """Return depression model metadata"""
    try:
        service = get_depression_service()
        return service.get_info()
    except Exception as e:
        return {"error": str(e)}
