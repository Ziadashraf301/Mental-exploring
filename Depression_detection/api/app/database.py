from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON, Text
from datetime import datetime, timedelta
from typing import Dict, List
import uuid
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from app.config import settings
from sqlalchemy.sql import func

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), index=True)
    text = Column(Text)
    text_length = Column(Integer)
    prediction = Column(String)
    confidence = Column(Float)
    depression_prob = Column(Float)
    inference_time = Column(Float)
    meta = Column("metadata", JSON)
    created_at = Column(DateTime, default=func.now(), index=True)
    user = relationship("User", back_populates="predictions")
    feedbacks = relationship("Feedback", back_populates="prediction")


class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    email = Column(String, unique=True, nullable=True)
    total_predictions = Column(Integer, default=0)
    first_interaction = Column(DateTime, default=func.now())
    last_interaction = Column(DateTime, default=func.now())
    meta = Column("metadata", JSON)
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id = Column(String, ForeignKey("predictions.id"), index=True)
    correct_label = Column(String)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    comments = Column(Text)
    created_at = Column(DateTime, default=func.now())
    prediction = relationship("Prediction", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")

class Database:
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
    
    async def save_prediction(
        self,
        user_id: str,
        text: str,
        text_length: int,
        prediction: str,
        confidence: float,
        depression_prob: float,
        inference_time: float,
        metadata: Dict = None
    ):
        """Save prediction to database"""
        async with self.async_session() as session:
            pred = Prediction(
                user_id=user_id,
                text=text,
                text_length=text_length,
                prediction=prediction,
                confidence=confidence,
                depression_prob=depression_prob,
                inference_time=inference_time,
                meta=metadata
            )
            session.add(pred)
            
            # Update user statistics
            from sqlalchemy import select, update
            
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                await session.execute(
                    update(User)
                    .where(User.id == user_id)
                    .values(
                        total_predictions=User.total_predictions + 1,
                        last_interaction=func.now()
                    )
                )
            
            await session.commit()
    
    async def create_user(self, name: str, email: str = None, metadata: Dict = None):
        """Create new user"""
        async with self.async_session() as session:
            user = User(
                name=name,
                email=email,
                meta=metadata
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
    
    async def get_user_predictions(self, user_id: str, limit: int = 50, skip: int = 0):
        """Get user's prediction history"""
        from sqlalchemy import select
        
        async with self.async_session() as session:
            result = await session.execute(
                select(Prediction)
                .where(Prediction.user_id == user_id)
                .order_by(Prediction.created_at.desc())
                .limit(limit)
                .offset(skip)
            )
            predictions = result.scalars().all()
            
            return [
                {
                    "id": p.id,
                    "prediction": p.prediction,
                    "confidence": p.confidence,
                    "created_at": p.created_at.isoformat()
                }
                for p in predictions
            ]
    
    async def get_analytics(self, days: int = 7):
        """Get analytics for specified period"""
        from sqlalchemy import select, func
        
        start_date = datetime.now() - timedelta(days=days)
        
        async with self.async_session() as session:
            # Total predictions
            total_result = await session.execute(
                select(func.count(Prediction.id))
                .where(Prediction.created_at >= start_date)
            )
            total_predictions = total_result.scalar()
            
            # Unique users
            users_result = await session.execute(
                select(func.count(func.distinct(Prediction.user_id)))
                .where(Prediction.created_at >= start_date)
            )
            unique_users = users_result.scalar()
            
            # Depression rate
            depressed_result = await session.execute(
                select(func.count(Prediction.id))
                .where(
                    Prediction.created_at >= start_date,
                    Prediction.prediction == "Depressed"
                )
            )
            depressed_count = depressed_result.scalar()
            depression_rate = depressed_count / total_predictions if total_predictions > 0 else 0
            
            # Average confidence
            avg_conf_result = await session.execute(
                select(func.avg(Prediction.confidence))
                .where(Prediction.created_at >= start_date)
            )
            avg_confidence = avg_conf_result.scalar() or 0
            
            # Average inference time
            avg_time_result = await session.execute(
                select(func.avg(Prediction.inference_time))
                .where(Prediction.created_at >= start_date)
            )
            avg_inference_time = avg_time_result.scalar() or 0
            
            # Predictions by date
            daily_result = await session.execute(
                select(
                    func.date(Prediction.created_at).label('date'),
                    func.count(Prediction.id).label('count')
                )
                .where(Prediction.created_at >= start_date)
                .group_by(func.date(Prediction.created_at))
                .order_by(func.date(Prediction.created_at))
            )
            
            predictions_by_date = [
                {"date": row.date.isoformat(), "count": row.count}
                for row in daily_result
            ]
            
            return {
                "total_predictions": total_predictions,
                "unique_users": unique_users,
                "depression_rate": depression_rate,
                "avg_confidence": avg_confidence,
                "avg_inference_time": avg_inference_time,
                "predictions_by_date": predictions_by_date
            }
    
    async def save_feedback(
        self,
        prediction_id: str,
        correct_label: str,
        user_id: str = None,
        comments: str = None
    ):
        """Save user feedback"""
        async with self.async_session() as session:
            feedback = Feedback(
                prediction_id=prediction_id,
                correct_label=correct_label,
                user_id=user_id,
                comments=comments
            )
            session.add(feedback)
            await session.commit()