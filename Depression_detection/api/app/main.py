from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from datetime import datetime
import uuid

from app.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    UserCreate, 
    UserResponse,
    AnalyticsResponse,
    HealthResponse
)
from app.database import Database
from app.tracking import ModelMonitor
from app.config import settings
from app.middleware import RateLimitMiddleware

# Global variables for model
model = None
tokenizer = None
device = None
db = None
monitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, tokenizer, device, db, monitor
    
    print("🚀 Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(settings.MODEL_NAME)
    model.to(device)
    model.eval()
    
    # Initialize database and monitoring
    db = Database()
    await db.connect()
    monitor = ModelMonitor(project_name=settings.WANDB_PROJECT)
    
    print(f"✅ Model loaded on {device}")
    print(f"✅ Database connected")
    print(f"✅ Monitoring initialized")
    
    yield
    
    # Shutdown
    await db.disconnect()
    monitor.close()
    print("👋 Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Depression Detection API",
    description="BERT-based depression detection with MLOps tracking",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Depression Detection API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "analytics": "/analytics",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
        timestamp=datetime.now()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict depression from text
    
    - **text**: Input text to analyze
    - **user_id**: Optional user identifier for tracking
    - **metadata**: Optional metadata for additional context
    """
    try:
        start_time = time.time()
        
        # Determine user_id
        if request.user_id:
            user_id = request.user_id
            existing_user = await db.get_user(user_id)
            if not existing_user:
                # create user automatically if ID not found
                user_id = await db.create_user(name="Anonymous", metadata={})
        else:
            # No user_id provided, create new user
            user_id = await db.create_user(name="Anonymous", metadata={})

        # Tokenize
        encoding = tokenizer(
            request.text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        prediction = 'Depressed' if pred_class == 1 else 'Not Depressed'
        depression_prob = probs[0][1].item()
        not_depression_prob = probs[0][0].item()
        
        inference_time = time.time() - start_time
        
        # Create response
        response = PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            depression_probability=depression_prob,
            not_depression_probability=not_depression_prob,
            user_id=user_id,
            timestamp=datetime.now(),
            inference_time=inference_time,
            model_version=settings.MODEL_VERSION
        )
        
        # Background task: Save to database and log to W&B
        background_tasks.add_task(
            save_prediction,
            user_id=user_id,
            text=request.text,
            prediction=prediction,
            confidence=confidence,
            depression_prob=depression_prob,
            inference_time=inference_time,
            metadata=request.metadata
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

async def save_prediction(
    user_id: str,
    text: str,
    prediction: str,
    confidence: float,
    depression_prob: float,
    inference_time: float,
    metadata: dict = None
):
    """Background task to save prediction"""
    try:
        text_length = len(text)

        # Save to database
        await db.save_prediction(
            user_id=user_id,
            text=text,
            text_length=text_length,
            prediction=prediction,
            confidence=confidence,
            depression_prob=depression_prob,
            inference_time=inference_time,
            metadata=metadata
        )
        
        # Log to Weights & Biases
        monitor.log_prediction(
            user_id=user_id,
            text_length=text_length,
            prediction=1 if prediction == "Depressed" else 0,
            confidence=confidence,
            inference_time=inference_time
        )
        
    except Exception as e:
        print(f"Error saving prediction: {e}")

@app.post("/users", response_model=UserResponse, tags=["Users"])
async def create_user(user: UserCreate):
    """Create a new user"""
    try:
        user_id = await db.create_user(
            name=user.name,
            email=user.email,
            metadata=user.metadata
        )
        
        return UserResponse(
            user_id=user_id,
            name=user.name,
            email=user.email,
            created_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def get_user(user_id: str):
    """Get user information"""
    user = await db.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        user_id=user.id,
        name=user.name,
        email=user.email,
        created_at=user.first_interaction
    )

@app.get("/users/{user_id}/predictions", tags=["Users"])
async def get_user_predictions(user_id: str, limit: int = 50, skip: int = 0):
    """Get user's prediction history"""
    predictions = await db.get_user_predictions(user_id, limit=limit, skip=skip)
    
    return {
        "user_id": user_id,
        "predictions": predictions,
        "total": len(predictions)
    }

@app.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics(days: int = 7):
    """
    Get analytics for the specified time period
    
    - **days**: Number of days to include in analytics (default: 7)
    """
    try:
        analytics = await db.get_analytics(days=days)
        
        return AnalyticsResponse(
            total_predictions=analytics['total_predictions'],
            unique_users=analytics['unique_users'],
            depression_rate=analytics['depression_rate'],
            avg_confidence=analytics['avg_confidence'],
            avg_inference_time=analytics['avg_inference_time'],
            predictions_by_date=analytics['predictions_by_date'],
            period_days=days,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")

@app.get("/analytics/realtime", tags=["Analytics"])
async def get_realtime_analytics():
    """Get real-time system metrics"""
    return {
        "model": {
            "name": settings.MODEL_NAME,
            "version": settings.MODEL_VERSION,
            "device": str(device)
        },
        "system": {
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time()
        }
    }

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    prediction_id: str,
    correct_label: str,
    user_id: str = None,
    comments: str = None
):
    """
    Submit feedback for a prediction
    
    - **prediction_id**: ID of the prediction
    - **correct_label**: Correct label (Depressed/Not Depressed)
    - **user_id**: User who submitted feedback
    - **comments**: Additional comments
    """
    try:
        await db.save_feedback(
            prediction_id=prediction_id,
            correct_label=correct_label,
            user_id=user_id,
            comments=comments
        )
        
        # Log to W&B
        monitor.log_feedback(
            prediction_id=prediction_id,
            correct_label=correct_label
        )
        
        return {
            "message": "Feedback submitted successfully",
            "prediction_id": prediction_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get model information"""
    return {
        "model_name": settings.MODEL_NAME,
        "model_version": settings.MODEL_VERSION,
        "device": str(device),
        "max_length": 128,
        "classes": ["Not Depressed", "Depressed"],
        "architecture": "roberta-base with LoRA",
        "training_date": "2025-01-XX",
        "performance": {
            "accuracy": 0.9103,
            "f1_score": 0.9070,
            "precision": 0.9149,
            "recall": 0.8993,
            "auc_roc": 0.9685
        }
    }

# Batch prediction endpoint
@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(texts: list[str], user_id: str = None):
    """
    Batch prediction for multiple texts
    
    - **texts**: List of texts to analyze
    - **user_id**: Optional user identifier
    """
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    predictions = []

    # Determine user_id
    if user_id:
        existing_user = await db.get_user(user_id)
        if not existing_user:
            # create user automatically if ID not found
            user_id = await db.create_user(name="Anonymous", metadata={})
    else:
        # No user_id provided, create new user
        user_id = await db.create_user(name="Anonymous", metadata={})
    
    for text in texts:
        request = PredictionRequest(text=text, user_id=user_id)
        result = await predict(request, BackgroundTasks())

        # call save_prediction()
        await save_prediction(
            user_id=result.user_id,
            text=text,
            prediction=result.prediction,
            confidence=result.confidence,
            depression_prob=result.depression_probability,
            inference_time=result.inference_time,
            metadata=None
        )

        #log to W&B
        monitor.log_prediction(
            user_id=result.user_id,
            text_length=len(text),
            prediction=1 if result.prediction == "Depressed" else 0,
            confidence=result.confidence,
            inference_time=result.inference_time
        )

        predictions.append(result)

    
    return {
        "predictions": predictions,
        "total": len(predictions)
    }