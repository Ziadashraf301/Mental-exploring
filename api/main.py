"""
Main FastAPI Application
Unified API for Depression, Emotion, and Sentiment Detection
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import logging
from pathlib import Path
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from config import settings
from core.database import Database
from services import get_emotion_service, get_sentiment_service

# Import routers
from routers import emotion, sentiment, users, analytics
# from api.routers import depression, sentiment

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

# Global instances
db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    global db
    
    logger.info("STARTING MENTAL HEALTH DETECTION API")
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        db = Database()
        await db.connect()
        logger.info("✓ Database connected")
        
        # Initialize Emotion Detection Service
        logger.info("Initializing Emotion Detection Service...")
        emotion_service = get_emotion_service()
        emotion_service.initialize()
        logger.info("✓ Emotion Detection Service initialized")
        
        # Initialize Sentiment Analysis Service
        logger.info("Initializing Sentiment Analysis Service...")
        sentiment_service = get_sentiment_service()
        sentiment_service.initialize()
        logger.info("✓ Sentiment Analysis Service initialized")
          
        # TODO: Initialize Depression Detection Service

        logger.info("API STARTED SUCCESSFULLY")
        logger.info(f"Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("\nShutting down API...")
    if db:
        await db.disconnect()
        logger.info("✓ Database disconnected")
    
    logger.info("API shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from middleware.rate_limiter import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)

# EXCEPTION HANDLERS
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )


# ROUTERS
app.include_router(emotion.router)
# app.include_router(depression.router)
app.include_router(sentiment.router)
app.include_router(users.router)
app.include_router(analytics.router)


# ROOT ENDPOINTS
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "services": {
            "emotion_detection": "active",
            "depression_detection": "planned",
            "sentiment_analysis": "active"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "emotion": "/emotion",
            "depression": "/depression",
            "sentiment": "/sentiment"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Global health check
    
    Checks status of all services and dependencies
    """
    try:
        # Check database
        db_status = db is not None
                
        # Check services
        emotion_service = get_emotion_service()
        emotion_status = emotion_service.initialized

        sentiment_service = get_sentiment_service()
        sentiment_status = sentiment_service.initialized
        
        all_healthy = db_status and emotion_status and sentiment_status
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": {
                "emotion_detection": emotion_status,
                "depression_detection": False,  # TODO
                "sentiment_analysis": sentiment_status
            },
            "dependencies": {
                "database": db_status
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }


# RUN APPLICATION
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )