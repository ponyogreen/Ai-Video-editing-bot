"""Main FastAPI application for AI Self-Discovery Chatbot"""
import os
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
from pathlib import Path

from database import init_db, get_db, User, Conversation, JournalEntry, UserImage, Insight
from chatbot_engine import ChatbotEngine

# Initialize FastAPI app
app = FastAPI(title="AI Self-Discovery Chatbot", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot engine
chatbot = ChatbotEngine()

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic models for API
class UserCreate(BaseModel):
    username: str


class MessageRequest(BaseModel):
    username: str
    message: str


class JournalEntryCreate(BaseModel):
    username: str
    title: str
    content: str


class InsightResponse(BaseModel):
    category: str
    insight: str
    confidence: int
    generated_at: datetime


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    print("Database initialized")


# API Endpoints
@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")


@app.post("/api/users")
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    # Check if user exists
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        return {"message": "User already exists", "user_id": existing_user.id}

    # Create new user
    new_user = User(username=user.username, profile_data={})
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User created successfully", "user_id": new_user.id}


@app.get("/api/users/{username}")
async def get_user(username: str, db: Session = Depends(get_db)):
    """Get user information"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user.id,
        "username": user.username,
        "created_at": user.created_at,
        "profile_data": user.profile_data,
        "conversation_count": len(user.conversations),
        "journal_count": len(user.journal_entries),
        "image_count": len(user.images)
    }


@app.post("/api/chat")
async def chat(request: MessageRequest, db: Session = Depends(get_db)):
    """Process chat message and return response"""
    # Get or create user
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        user = User(username=request.username, profile_data={})
        db.add(user)
        db.commit()
        db.refresh(user)

    # Build user context from history
    user_context = build_user_context(user, db)

    # Generate response
    result = await chatbot.generate_response(request.message, user_context)

    # Store conversation
    conversation = Conversation(
        user_id=user.id,
        message=request.message,
        response=result["response"],
        message_data={"analysis": result["analysis"]}
    )
    db.add(conversation)

    # Update user profile with new information
    update_user_profile(user, result["analysis"], db)

    db.commit()

    return {
        "response": result["response"],
        "analysis": result["analysis"],
        "timestamp": datetime.utcnow()
    }


@app.get("/api/conversations/{username}")
async def get_conversations(username: str, limit: int = 50, db: Session = Depends(get_db)):
    """Get conversation history for a user"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    conversations = db.query(Conversation)\
        .filter(Conversation.user_id == user.id)\
        .order_by(Conversation.timestamp.desc())\
        .limit(limit)\
        .all()

    return [
        {
            "id": conv.id,
            "message": conv.message,
            "response": conv.response,
            "timestamp": conv.timestamp,
            "metadata": conv.message_data
        }
        for conv in conversations
    ]


@app.post("/api/journal")
async def create_journal_entry(entry: JournalEntryCreate, db: Session = Depends(get_db)):
    """Create a journal entry"""
    user = db.query(User).filter(User.username == entry.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Analyze journal entry
    analysis = chatbot.analyze_message(entry.content, {})

    # Create journal entry
    journal = JournalEntry(
        user_id=user.id,
        title=entry.title,
        content=entry.content,
        analysis=analysis
    )
    db.add(journal)
    db.commit()

    return {
        "message": "Journal entry created",
        "analysis": analysis
    }


@app.get("/api/journal/{username}")
async def get_journal_entries(username: str, db: Session = Depends(get_db)):
    """Get journal entries for a user"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    entries = db.query(JournalEntry)\
        .filter(JournalEntry.user_id == user.id)\
        .order_by(JournalEntry.timestamp.desc())\
        .all()

    return [
        {
            "id": entry.id,
            "title": entry.title,
            "content": entry.content,
            "timestamp": entry.timestamp,
            "analysis": entry.analysis
        }
        for entry in entries
    ]


@app.post("/api/upload-image")
async def upload_image(
    username: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and analyze an image"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Save file
    file_path = UPLOAD_DIR / f"{user.id}_{datetime.utcnow().timestamp()}_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Analyze image
    analysis = chatbot.analyze_image(str(file_path))

    # Store image record
    user_image = UserImage(
        user_id=user.id,
        filename=str(file_path),
        description=analysis.get("description", ""),
        analysis=analysis
    )
    db.add(user_image)
    db.commit()

    return {
        "message": "Image uploaded successfully",
        "analysis": analysis
    }


@app.get("/api/insights/{username}")
async def get_insights(username: str, db: Session = Depends(get_db)):
    """Get personalized insights for a user"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Gather all user data
    user_data = {
        "conversations": [
            {
                "message": conv.message,
                "response": conv.response,
                "metadata": conv.message_data,
                "timestamp": conv.timestamp
            }
            for conv in user.conversations
        ],
        "journal_entries": [
            {
                "title": entry.title,
                "content": entry.content,
                "analysis": entry.analysis
            }
            for entry in user.journal_entries
        ],
        "images": [
            {
                "filename": img.filename,
                "analysis": img.analysis
            }
            for img in user.images
        ]
    }

    # Generate new insights
    new_insights = chatbot.generate_insights(user_data)

    # Store insights
    for insight_data in new_insights:
        # Check if similar insight already exists
        existing = db.query(Insight).filter(
            Insight.user_id == user.id,
            Insight.category == insight_data["category"],
            Insight.insight == insight_data["insight"]
        ).first()

        if not existing:
            insight = Insight(
                user_id=user.id,
                category=insight_data["category"],
                insight=insight_data["insight"],
                confidence=insight_data["confidence"]
            )
            db.add(insight)

    db.commit()

    # Get all insights for user
    all_insights = db.query(Insight)\
        .filter(Insight.user_id == user.id)\
        .order_by(Insight.generated_at.desc())\
        .all()

    return [
        {
            "category": insight.category,
            "insight": insight.insight,
            "confidence": insight.confidence,
            "generated_at": insight.generated_at
        }
        for insight in all_insights
    ]


@app.get("/api/profile/{username}")
async def get_profile_summary(username: str, db: Session = Depends(get_db)):
    """Get complete profile summary"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "username": user.username,
        "member_since": user.created_at,
        "profile_data": user.profile_data,
        "statistics": {
            "total_conversations": len(user.conversations),
            "total_journal_entries": len(user.journal_entries),
            "total_images": len(user.images),
            "total_insights": len(user.insights)
        }
    }


# Helper functions
def build_user_context(user: User, db: Session) -> dict:
    """Build context about user from their data"""
    context = {
        "username": user.username,
        "profile_data": user.profile_data or {},
        "interests": user.profile_data.get("interests", []) if user.profile_data else [],
        "communication_style": user.profile_data.get("communication_style", "conversational") if user.profile_data else "conversational"
    }

    # Get recent topics from conversations
    recent_conversations = db.query(Conversation)\
        .filter(Conversation.user_id == user.id)\
        .order_by(Conversation.timestamp.desc())\
        .limit(10)\
        .all()

    recent_topics = []
    for conv in recent_conversations:
        if conv.message_data and "analysis" in conv.message_data:
            topics = conv.message_data["analysis"].get("topics", [])
            recent_topics.extend(topics)

    context["recent_topics"] = list(set(recent_topics))

    return context


def update_user_profile(user: User, analysis: dict, db: Session):
    """Update user profile based on conversation analysis"""
    profile = user.profile_data or {}

    # Update interests
    interests = set(profile.get("interests", []))
    interests.update(analysis.get("topics", []))
    profile["interests"] = list(interests)

    # Update communication style (use most recent)
    if "communication_style" in analysis:
        profile["communication_style"] = analysis["communication_style"]

    # Track emotional patterns
    if "sentiment" in analysis:
        sentiments = profile.get("sentiment_history", [])
        sentiments.append({
            "sentiment": analysis["sentiment"],
            "timestamp": datetime.utcnow().isoformat()
        })
        # Keep only last 50
        profile["sentiment_history"] = sentiments[-50:]

    user.profile_data = profile
    db.add(user)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
