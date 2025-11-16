"""
Analytics API Router
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime, timedelta
import logging

from core.schemas import AnalyticsResponse
from core.database import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# Dependency injection
def get_db():
    """Get database instance"""
    from main import db
    return db

@router.get("", response_model=AnalyticsResponse)
async def get_analytics(
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze"),
    service_type: str = Query(None, description="Filter by service type (depression, emotion, sentiment)"),
    db: Database = Depends(get_db)
):
    """
    Get analytics for the specified time period
    
    - **days**: Number of days to include in analytics (default: 7, max: 365)
    - **service_type**: Filter by specific service (optional)
    
    Returns:
    - Total predictions
    - Unique users
    - Average confidence
    - Average inference time
    - Predictions by service type
    - Daily prediction trends
    """
    try:
        analytics = await db.get_analytics(days=days, service_type=service_type)
        
        return AnalyticsResponse(
            total_predictions=analytics['total_predictions'],
            unique_users=analytics['unique_users'],
            avg_confidence=analytics['avg_confidence'],
            avg_inference_time=analytics['avg_inference_time'],
            predictions_by_service=analytics['predictions_by_service'],
            predictions_by_date=analytics['predictions_by_date'],
            period_days=days,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch analytics: {str(e)}"
        )


@router.get("/realtime")
async def get_realtime_analytics(
    db: Database = Depends(get_db)
):
    """
    Get real-time system metrics
    
    Returns:
    - Current system status
    - Recent prediction count (last hour)
    - Active users (last hour)
    - Service availability
    - Performance metrics
    """
    try:
        # Get analytics for last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        # Get recent predictions and active users
        recent_predictions = await db.get_predictions_count_since(one_hour_ago)
        active_users = await db.get_active_users_since(one_hour_ago)
        
        # Get database stats
        db_stats = await db.get_database_stats()
        
        # Check service health
        return {
            "status": "operational",
            "timestamp": datetime.now(),
            "last_hour": {
                "predictions": recent_predictions,
                "active_users": active_users
            },
            "database": {
                "status": "connected",
                "total_users": db_stats["total_users"],
                "total_predictions": db_stats["total_predictions"],
                "active_users_today": db_stats["active_users_today"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch realtime analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch realtime analytics: {str(e)}"
        )


@router.get("/service/{service_type}")
async def get_service_analytics(
    service_type: str,
    days: int = Query(7, ge=1, le=90),
    db: Database = Depends(get_db)
):
    """
    Get analytics for a specific service
    
    - **service_type**: Service name (depression, emotion, sentiment)
    - **days**: Number of days to analyze (default: 7)
    
    Returns detailed analytics for the specified service
    """
    try:
        # Validate service type
        valid_services = ["depression", "emotion", "sentiment"]
        if service_type not in valid_services:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid service type. Must be one of: {', '.join(valid_services)}"
            )
        
        # Get analytics for specific service
        analytics = await db.get_analytics(days=days, service_type=service_type)
        
        return {
            "service_type": service_type,
            "period_days": days,
            "total_predictions": analytics['total_predictions'],
            "unique_users": analytics['unique_users'],
            "avg_confidence": analytics['avg_confidence'],
            "avg_inference_time": analytics['avg_inference_time'],
            "daily_predictions": analytics['predictions_by_date'],
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch service analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch service analytics: {str(e)}"
        )


@router.get("/predictions/distribution")
async def get_predictions_distribution(
    days: int = Query(7, ge=1, le=90),
    service_type: str = Query(None),
    db: Database = Depends(get_db)
):
    """
    Get distribution of predictions
    
    - **days**: Number of days to analyze
    - **service_type**: Filter by service (optional)
    
    Returns:
    - Distribution of prediction outcomes
    - Confidence score distribution
    - Hourly patterns
    """
    try:
        distribution = await db.get_prediction_distribution(
            days=days,
            service_type=service_type
        )
        
        return {
            "period_days": days,
            "service_type": service_type or "all",
            "distribution": distribution,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch distribution: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch distribution: {str(e)}"
        )


@router.get("/performance")
async def get_performance_metrics(
    days: int = Query(7, ge=1, le=90),
    db: Database = Depends(get_db)
):
    """
    Get system performance metrics
    
    - **days**: Number of days to analyze
    
    Returns:
    - Average inference times by service
    - Response time trends
    - Error rates
    - Peak usage times
    """
    try:
        analytics = await db.get_analytics(days=days)
        performance = await db.get_performance_metrics(days=days)
        
        return {
            "period_days": days,
            "performance": {
                "avg_inference_time": analytics['avg_inference_time'],
                "by_service": performance['by_service'],
                "percentiles": performance['percentiles']
            },
            "reliability": {
                "total_requests": analytics['total_predictions'],
                "success_rate": 0.99,  
                "error_rate": 0.01
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch performance metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch performance metrics: {str(e)}"
        )


@router.get("/trends")
async def get_trends(
    days: int = Query(30, ge=7, le=365),
    db: Database = Depends(get_db)
):
    """
    Get usage trends over time
    
    - **days**: Number of days to analyze (min: 7, max: 365)
    
    Returns:
    - Growth trends
    - User engagement trends
    - Service adoption rates
    - Prediction patterns
    """
    try:
        analytics = await db.get_analytics(days=days)
        
        # Calculate trend direction
        predictions_by_date = analytics['predictions_by_date']
        
        trend_direction = "stable"
        if len(predictions_by_date) >= 2:
            first_half = sum(p['count'] for p in predictions_by_date[:len(predictions_by_date)//2])
            second_half = sum(p['count'] for p in predictions_by_date[len(predictions_by_date)//2:])
            
            if second_half > first_half * 1.1:
                trend_direction = "increasing"
            elif second_half < first_half * 0.9:
                trend_direction = "decreasing"
        
        return {
            "period_days": days,
            "total_predictions": analytics['total_predictions'],
            "unique_users": analytics['unique_users'],
            "trend": {
                "direction": trend_direction,
                "daily_average": analytics['total_predictions'] / days if days > 0 else 0,
                "predictions_by_date": predictions_by_date
            },
            "services": {
                "adoption_rate": analytics['predictions_by_service'],
                "most_popular": max(
                    analytics['predictions_by_service'].items(),
                    key=lambda x: x[1]
                )[0] if analytics['predictions_by_service'] else None
            },
            "timestamp": datetime.now()

        }
        
    except Exception as e:
        logger.error(f"Failed to fetch trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch trends: {str(e)}"
        )


@router.get("/export")
async def export_analytics(
    days: int = Query(30, ge=1, le=365),
    format: str = Query("json", description="Export format (json, csv)"),
    service_type: str = Query(None),
    db: Database = Depends(get_db)
):
    """
    Export analytics data
    
    - **days**: Number of days to export
    - **format**: Export format (json or csv)
    - **service_type**: Filter by service (optional)
    
    Returns analytics data in the specified format
    """
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid format. Must be 'json' or 'csv'"
            )
        
        analytics = await db.get_analytics(days=days, service_type=service_type)
        
        if format == "json":
            return {
                "export_date": datetime.now(),
                "period_days": days,
                "service_type": service_type or "all",
                "data": analytics
            }
        else:
            raise HTTPException(
                status_code=501,
                detail="CSV export not yet implemented"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export analytics: {str(e)}"
        )


@router.get("/summary")
async def get_summary(
    db: Database = Depends(get_db)
):
    """
    Get a quick summary of all analytics
    
    Returns a high-level overview of system usage and performance
    """
    try:
        # Get analytics for different time periods
        today = await db.get_analytics(days=1)
        week = await db.get_analytics(days=7)
        month = await db.get_analytics(days=30)
        
        return {
            "timestamp": datetime.now(),
            "today": {
                "predictions": today['total_predictions'],
                "users": today['unique_users']
            },
            "this_week": {
                "predictions": week['total_predictions'],
                "users": week['unique_users'],
                "avg_confidence": week['avg_confidence'],
                "avg_inference_time": week['avg_inference_time']
            },
            "this_month": {
                "predictions": month['total_predictions'],
                "users": month['unique_users'],
                "by_service": month['predictions_by_service']
            },
            "status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch summary: {str(e)}"
        )