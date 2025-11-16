"""
Users Management API Router
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime
import logging

from core.schemas import UserCreate, UserResponse
from core.database import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])


# Dependency injection
def get_db():
    """Get database instance"""
    from main import db
    return db


@router.post("", response_model=UserResponse)
async def create_user(
    user: UserCreate,
    db: Database = Depends(get_db)
):
    """
    Create a new user
    
    - **name**: User name (required)
    - **email**: User email (optional, must be unique)
    - **metadata**: Additional user metadata (optional)
    
    Returns the created user with generated user_id
    """
    try:
        # Check if email already exists
        if user.email:
            email_exists = await db.check_email_exists(user.email)
            if email_exists:
                raise HTTPException(
                    status_code=400,
                    detail=f"Email already exists: {user.email}"
                )
        
        user_id = await db.create_user(
            name=user.name,
            email=user.email,
            metadata=user.metadata
        )
        
        logger.info(f"Created new user: {user_id}")
        
        return UserResponse(
            user_id=user_id,
            name=user.name,
            email=user.email,
            created_at=datetime.now(),
            total_requests=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create user: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create user: {str(e)}"
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db: Database = Depends(get_db)
):
    """
    Get user information by user_id
    
    - **user_id**: User identifier
    
    Returns user details including total requests and activity
    """
    try:
        user = await db.get_user(user_id)
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User not found: {user_id}"
            )
        
        return UserResponse(
            user_id=user.id,
            name=user.name,
            email=user.email,
            created_at=user.created_at,
            total_requests=user.total_requests
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user: {str(e)}"
        )


@router.get("/{user_id}/predictions")
async def get_user_predictions(
    user_id: str,
    service_type: str = Query(None, description="Filter by service type (depression, emotion, sentiment)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of predictions to return"),
    skip: int = Query(0, ge=0, description="Number of predictions to skip"),
    db: Database = Depends(get_db)
):
    """
    Get user's prediction history
    
    - **user_id**: User identifier
    - **service_type**: Filter by service (optional)
    - **limit**: Maximum results (default: 50, max: 100)
    - **skip**: Pagination offset (default: 0)
    
    Returns list of user's predictions with details
    """
    try:
        # Check if user exists
        user = await db.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User not found: {user_id}"
            )
        
        # Get predictions
        predictions = await db.get_user_predictions(
            user_id=user_id,
            service_type=service_type,
            limit=limit,
            skip=skip
        )
        
        return {
            "user_id": user_id,
            "service_type": service_type,
            "predictions": predictions,
            "count": len(predictions),
            "limit": limit,
            "skip": skip
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get predictions for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve predictions: {str(e)}"
        )


@router.get("/{user_id}/stats")
async def get_user_stats(
    user_id: str,
    db: Database = Depends(get_db)
):
    """
    Get user statistics
    
    - **user_id**: User identifier
    
    Returns:
    - Total predictions by service type
    - Most common predictions
    - Average confidence scores
    - Activity timeline
    """
    try:
        # Check if user exists
        user = await db.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User not found: {user_id}"
            )
        
        # Get all user predictions
        all_predictions = await db.get_user_predictions(
            user_id=user_id,
            limit=10000  # High limit to get all
        )
        
        # Calculate statistics
        stats = {
            "user_id": user_id,
            "user_name": user.name,
            "total_predictions": user.total_requests,
            "member_since": user.created_at,
            "last_active": user.last_active,
            "by_service": {},
            "average_confidence": 0.0,
            "most_common_predictions": {}
        }
        
        if not all_predictions:
            return stats
        
        # Count by service type and prediction
        service_counts = {}
        prediction_counts = {}
        total_confidence = 0
        count = 0
        
        for pred in all_predictions:
            service = pred.get("service_type", "unknown")
            service_counts[service] = service_counts.get(service, 0) + 1
            
            prediction = pred.get("prediction", "unknown")
            prediction_counts[prediction] = prediction_counts.get(prediction, 0) + 1
            
            # Sum confidence for average
            if "confidence" in pred:
                total_confidence += pred["confidence"]
                count += 1
        
        stats["by_service"] = service_counts
        stats["average_confidence"] = total_confidence / count if count > 0 else 0.0
        stats["most_common_predictions"] = dict(
            sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    db: Database = Depends(get_db)
):
    """
    Delete a user and all their predictions
    
    - **user_id**: User identifier
    """
    try:
        # Check if user exists
        user = await db.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User not found: {user_id}"
            )
        
        # Delete user (cascade will delete predictions)
        await db.delete_user(user_id)
        
        logger.warning(f"User deleted: {user_id}")
        
        return {
            "message": "User deleted successfully",
            "user_id": user_id,
            "deleted_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete user: {str(e)}"
        )


@router.patch("/{user_id}")
async def update_user(
    user_id: str,
    name: str = Query(None, description="New user name"),
    email: str = Query(None, description="New user email"),
    db: Database = Depends(get_db)
):
    """
    Update user information
    
    - **user_id**: User identifier
    - **name**: New name (optional)
    - **email**: New email (optional)
    
    Returns updated user information
    """
    try:
        # Check if user exists
        user = await db.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail=f"User not found: {user_id}"
            )
        
        # Check if email is being changed and if it exists
        if email and email != user.email:
            email_exists = await db.check_email_exists(email)
            if email_exists:
                raise HTTPException(
                    status_code=400,
                    detail=f"Email already exists: {email}"
                )
        
        # Update user
        await db.update_user(user_id, name=name, email=email)
        
        logger.info(f"User updated: {user_id}")
        
        # Get updated user
        updated_user = await db.get_user(user_id)
        
        return {
            "message": "User updated successfully",
            "user_id": user_id,
            "name": updated_user.name,
            "email": updated_user.email,
            "updated_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update user: {str(e)}"
        )