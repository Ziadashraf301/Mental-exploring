"""
Database models and operations
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, JSON, Text, ForeignKey, func
)
from datetime import datetime, timedelta
from typing import Dict, List
import uuid

from config import settings

Base = declarative_base()


# ===========================================
# DATABASE MODELS
# ===========================================

class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now())
    total_requests = Column(Integer, default=0)
    meta = Column("metadata", JSON, nullable=True)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    """Prediction model - stores all predictions from all services"""
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), index=True)
    service_type = Column(String, index=True)  # "depression", "emotion", "sentiment"
    
    # Input data
    input_text = Column(Text, nullable=True)
    input_image_path = Column(String, nullable=True)
    input_length = Column(Integer, nullable=True)
    
    # Prediction results
    prediction = Column(String)
    confidence = Column(Float)
    probabilities = Column(JSON, nullable=True)
    
    # Model info
    model_name = Column(String)
    model_version = Column(String)
    
    # Performance
    inference_time = Column(Float)
    
    # Metadata
    meta = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", back_populates="predictions")


# ===========================================
# DATABASE CLASS
# ===========================================

class Database:
    """Database operations"""
    
    def __init__(self):
        self.engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False
        )
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def connect(self):
        """Create database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def disconnect(self):
        """Close database connection"""
        await self.engine.dispose()
    
    # ===========================================
    # USER OPERATIONS
    # ===========================================
    
    async def create_user(self, name: str, email: str = None, metadata: Dict = None) -> str:
        """Create new user"""
        async with self.async_session() as session:
            user = User(
                name=name,
                email=email,
                metadata=metadata
            )
            session.add(user)
            await session.commit()
            return user.id
    
    async def get_user(self, user_id: str):
        """Get user by ID"""
        from sqlalchemy import select
        
        async with self.async_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
    
    async def update_user_activity(self, user_id: str):
        """Update user last active time and request count"""
        from sqlalchemy import update
        
        async with self.async_session() as session:
            await session.execute(
                update(User)
                .where(User.id == user_id)
                .values(
                    last_active=func.now(),
                    total_requests=User.total_requests + 1
                )
            )
            await session.commit()
    
    # ===========================================
    # PREDICTION OPERATIONS
    # ===========================================
    
    async def save_prediction(
        self,
        user_id: str,
        service_type: str,
        prediction: str,
        confidence: float,
        model_name: str,
        model_version: str,
        inference_time: float,
        input_text: str = None,
        input_image_path: str = None,
        input_length: int = None,
        probabilities: Dict = None,
        metadata: Dict = None
    ):
        """Save prediction to database"""
        async with self.async_session() as session:
            pred = Prediction(
                user_id=user_id,
                service_type=service_type,
                input_text=input_text,
                input_image_path=input_image_path,
                input_length=input_length,
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                model_name=model_name,
                model_version=model_version,
                inference_time=inference_time,
                metadata=metadata
            )
            session.add(pred)
            await session.commit()
            
            # Update user activity
            await self.update_user_activity(user_id)
            
            return pred.id
    
    async def get_user_predictions(
        self,
        user_id: str,
        service_type: str = None,
        limit: int = 50,
        skip: int = 0
    ):
        """Get user's prediction history"""
        from sqlalchemy import select
        
        async with self.async_session() as session:
            query = select(Prediction).where(Prediction.user_id == user_id)
            
            if service_type:
                query = query.where(Prediction.service_type == service_type)
            
            query = query.order_by(Prediction.created_at.desc()).limit(limit).offset(skip)
            
            result = await session.execute(query)
            predictions = result.scalars().all()
            
            return [
                {
                    "id": p.id,
                    "service_type": p.service_type,
                    "prediction": p.prediction,
                    "confidence": p.confidence,
                    "created_at": p.created_at.isoformat()
                }
                for p in predictions
            ]
    
    # ===========================================
    # ANALYTICS OPERATIONS
    # ===========================================
    
    async def get_analytics(self, days: int = 7, service_type: str = None):
        """Get analytics for specified period"""
        from sqlalchemy import select, func
        
        start_date = datetime.now() - timedelta(days=days)
        
        async with self.async_session() as session:
            # Base query
            base_query = select(Prediction).where(Prediction.created_at >= start_date)
            if service_type:
                base_query = base_query.where(Prediction.service_type == service_type)
            
            # Total predictions
            total_result = await session.execute(
                select(func.count(Prediction.id))
                .select_from(base_query.subquery())
            )
            total_predictions = total_result.scalar() or 0
            
            # Unique users
            users_result = await session.execute(
                select(func.count(func.distinct(Prediction.user_id)))
                .select_from(base_query.subquery())
            )
            unique_users = users_result.scalar() or 0
            
            # Average confidence
            avg_conf_result = await session.execute(
                select(func.avg(Prediction.confidence))
                .select_from(base_query.subquery())
            )
            avg_confidence = avg_conf_result.scalar() or 0
            
            # Average inference time
            avg_time_result = await session.execute(
                select(func.avg(Prediction.inference_time))
                .select_from(base_query.subquery())
            )
            avg_inference_time = avg_time_result.scalar() or 0
            
            # Predictions by service type
            service_query = select(
                Prediction.service_type,
                func.count(Prediction.id).label('count')
            ).where(Prediction.created_at >= start_date).group_by(Prediction.service_type)
            
            service_result = await session.execute(service_query)
            predictions_by_service = {
                row.service_type: row.count
                for row in service_result
            }
            
            # Daily predictions
            daily_query = select(
                func.date(Prediction.created_at).label('date'),
                func.count(Prediction.id).label('count')
            ).where(Prediction.created_at >= start_date).group_by(
                func.date(Prediction.created_at)
            ).order_by(func.date(Prediction.created_at))
            
            if service_type:
                daily_query = daily_query.where(Prediction.service_type == service_type)
            
            daily_result = await session.execute(daily_query)
            predictions_by_date = [
                {"date": row.date.isoformat(), "count": row.count}
                for row in daily_result
            ]
            
            return {
                "total_predictions": total_predictions,
                "unique_users": unique_users,
                "avg_confidence": avg_confidence,
                "avg_inference_time": avg_inference_time,
                "predictions_by_service": predictions_by_service,
                "predictions_by_date": predictions_by_date
            }