"""Database models and configuration for the AI chatbot"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User model for storing user information"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    profile_data = Column(JSON, default=dict)  # Stores learned information about user

    conversations = relationship("Conversation", back_populates="user")
    journal_entries = relationship("JournalEntry", back_populates="user")
    images = relationship("UserImage", back_populates="user")
    insights = relationship("Insight", back_populates="user")


class Conversation(Base):
    """Store conversation history"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message_data = Column(JSON, default=dict)  # Store detected patterns, emotions, etc.

    user = relationship("User", back_populates="conversations")


class JournalEntry(Base):
    """Store journal entries"""
    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    analysis = Column(JSON, default=dict)  # AI analysis of the entry

    user = relationship("User", back_populates="journal_entries")


class UserImage(Base):
    """Store user images and their analysis"""
    __tablename__ = "user_images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    description = Column(Text)
    analysis = Column(JSON, default=dict)  # AI analysis of the image
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="images")


class Insight(Base):
    """Store generated insights about the user"""
    __tablename__ = "insights"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    category = Column(String)  # e.g., "personality", "interests", "patterns"
    insight = Column(Text)
    confidence = Column(Integer)  # 0-100
    generated_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="insights")


# Database setup
DATABASE_URL = "sqlite:///./chatbot.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize the database"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
