"""Simple test script to verify the chatbot functionality"""
import asyncio
from chatbot_engine import ChatbotEngine
from database import init_db, SessionLocal, User, Conversation

def test_chatbot_engine():
    """Test the chatbot engine"""
    print("Testing Chatbot Engine...")

    engine = ChatbotEngine()

    # Test message analysis
    test_message = "I'm feeling really happy today! I had a great time working on my art project."
    user_context = {"interests": ["art"], "communication_style": "positive"}

    # Test response generation
    result = asyncio.run(engine.generate_response(test_message, user_context))

    print(f"\nTest Message: {test_message}")
    print(f"\nBot Response: {result['response']}")
    print(f"\nAnalysis: {result['analysis']}")

    # Test insight generation
    print("\n" + "="*50)
    print("Testing Insight Generation...")

    user_data = {
        "conversations": [
            {
                "message": "I love painting",
                "response": "That's wonderful!",
                "metadata": {
                    "analysis": {
                        "topics": ["hobbies", "emotions"],
                        "sentiment": "positive",
                        "communication_style": "brief"
                    }
                }
            },
            {
                "message": "I'm working on a new painting project",
                "response": "Tell me more!",
                "metadata": {
                    "analysis": {
                        "topics": ["hobbies", "work"],
                        "sentiment": "positive",
                        "communication_style": "conversational"
                    }
                }
            }
        ],
        "journal_entries": [
            {
                "title": "My Day",
                "content": "Had a great day painting",
                "analysis": {"sentiment": "positive"}
            }
        ]
    }

    insights = engine.generate_insights(user_data)

    print(f"\nGenerated {len(insights)} insights:")
    for insight in insights:
        print(f"  - [{insight['category']}] {insight['insight']} (confidence: {insight['confidence']}%)")

    print("\nAll tests passed! ✓")


def test_database():
    """Test database setup"""
    print("\n" + "="*50)
    print("Testing Database...")

    # Initialize database
    init_db()
    print("Database initialized ✓")

    # Test creating a user
    db = SessionLocal()
    try:
        test_user = User(username="test_user", profile_data={"test": True})
        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        print(f"Created test user: {test_user.username} (ID: {test_user.id}) ✓")

        # Test creating a conversation
        conversation = Conversation(
            user_id=test_user.id,
            message="Hello!",
            response="Hi there!",
            message_data={"test": True}
        )
        db.add(conversation)
        db.commit()

        print(f"Created test conversation ✓")

        # Clean up
        db.delete(conversation)
        db.delete(test_user)
        db.commit()
        print("Cleaned up test data ✓")

    finally:
        db.close()

    print("Database tests passed! ✓")


if __name__ == "__main__":
    print("="*50)
    print("AI Self-Discovery Chatbot - Test Suite")
    print("="*50)

    test_chatbot_engine()
    test_database()

    print("\n" + "="*50)
    print("All tests completed successfully! 🎉")
    print("="*50)
