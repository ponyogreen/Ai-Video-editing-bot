"""AI Chatbot Engine for self-discovery and user profiling"""
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter
import re

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ChatbotEngine:
    """Main chatbot engine for conversation and analysis"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_mock = not self.api_key or not OPENAI_AVAILABLE

        if not self.use_mock:
            openai.api_key = self.api_key

    async def generate_response(self, message: str, user_context: Dict) -> Dict:
        """Generate a response based on user message and context"""
        if self.use_mock:
            return self._mock_response(message, user_context)

        try:
            # Build context from user profile
            context_prompt = self._build_context_prompt(user_context)

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500
            )

            bot_response = response.choices[0].message.content

            # Analyze the conversation
            analysis = self.analyze_message(message, user_context)

            return {
                "response": bot_response,
                "analysis": analysis
            }
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return self._mock_response(message, user_context)

    def _build_context_prompt(self, user_context: Dict) -> str:
        """Build context prompt from user profile"""
        prompt = """You are a compassionate AI companion designed to help users understand themselves better.
Your role is to:
1. Listen actively and ask thoughtful questions
2. Help users discover patterns in their thoughts and behaviors
3. Provide gentle insights without being judgmental
4. Encourage self-reflection and growth

"""

        if user_context.get("interests"):
            prompt += f"User's interests: {', '.join(user_context['interests'])}\n"

        if user_context.get("communication_style"):
            prompt += f"Communication style: {user_context['communication_style']}\n"

        if user_context.get("recent_topics"):
            prompt += f"Recent topics: {', '.join(user_context['recent_topics'][:5])}\n"

        prompt += "\nBe warm, understanding, and help the user learn about themselves."

        return prompt

    def _mock_response(self, message: str, user_context: Dict) -> Dict:
        """Generate mock response when OpenAI is not available"""
        message_lower = message.lower()

        # Simple pattern-based responses
        if any(word in message_lower for word in ["feel", "feeling", "emotion"]):
            response = "I hear that you're sharing your feelings. It takes courage to express emotions. Can you tell me more about what triggered this feeling?"
        elif any(word in message_lower for word in ["think", "thought", "believe"]):
            response = "That's an interesting perspective. What experiences have shaped this belief for you?"
        elif any(word in message_lower for word in ["want", "wish", "hope", "dream"]):
            response = "Dreams and aspirations tell us a lot about ourselves. What would achieving this mean to you?"
        elif "?" in message:
            response = "That's a thoughtful question. Let me reflect that back to you - what do you think the answer might be?"
        else:
            response = f"Thank you for sharing that with me. I'm learning more about you with each conversation. How does talking about this make you feel?"

        # Add personalization if we know the user
        if user_context.get("interests"):
            interests = user_context["interests"][:2]
            response += f" I notice you're interested in {', '.join(interests)} - does this connect to that in any way?"

        analysis = self.analyze_message(message, user_context)

        return {
            "response": response,
            "analysis": analysis
        }

    def analyze_message(self, message: str, user_context: Dict) -> Dict:
        """Analyze message for patterns and insights"""
        analysis = {
            "word_count": len(message.split()),
            "sentiment": self._detect_sentiment(message),
            "topics": self._extract_topics(message),
            "questions_asked": message.count("?"),
            "emotional_words": self._detect_emotional_words(message),
            "communication_style": self._detect_communication_style(message)
        }

        return analysis

    def _detect_sentiment(self, text: str) -> str:
        """Simple sentiment detection"""
        positive_words = ["happy", "joy", "excited", "love", "great", "amazing", "wonderful", "good"]
        negative_words = ["sad", "angry", "upset", "hate", "terrible", "awful", "bad", "frustrated"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from text"""
        # Common topic keywords
        topic_keywords = {
            "work": ["work", "job", "career", "office", "boss", "colleague"],
            "family": ["family", "parent", "mother", "father", "sibling", "child"],
            "health": ["health", "exercise", "sleep", "diet", "wellness"],
            "relationships": ["friend", "relationship", "dating", "partner", "love"],
            "hobbies": ["hobby", "music", "art", "reading", "gaming", "sports"],
            "emotions": ["feel", "emotion", "anxiety", "stress", "happiness"],
            "goals": ["goal", "dream", "aspiration", "future", "plan"]
        }

        text_lower = text.lower()
        detected_topics = []

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)

        return detected_topics

    def _detect_emotional_words(self, text: str) -> List[str]:
        """Detect emotional words in text"""
        emotional_words = [
            "happy", "sad", "angry", "excited", "anxious", "stressed",
            "frustrated", "joy", "fear", "worried", "calm", "peaceful",
            "love", "hate", "confused", "confident", "insecure"
        ]

        words = re.findall(r'\b\w+\b', text.lower())
        found_emotions = [word for word in words if word in emotional_words]

        return found_emotions

    def _detect_communication_style(self, text: str) -> str:
        """Detect communication style"""
        if len(text.split()) < 5:
            return "brief"
        elif len(text.split()) > 50:
            return "detailed"
        elif "?" in text:
            return "inquisitive"
        elif any(word in text.lower() for word in ["i think", "i believe", "in my opinion"]):
            return "reflective"
        else:
            return "conversational"

    def generate_insights(self, user_data: Dict) -> List[Dict]:
        """Generate insights from accumulated user data"""
        insights = []

        # Analyze conversation patterns
        if user_data.get("conversations"):
            conversations = user_data["conversations"]

            # Topic frequency
            all_topics = []
            for conv in conversations:
                if conv.get("metadata", {}).get("analysis", {}).get("topics"):
                    all_topics.extend(conv["metadata"]["analysis"]["topics"])

            if all_topics:
                topic_counts = Counter(all_topics)
                most_common = topic_counts.most_common(3)
                topics_str = ", ".join([f"{topic} ({count} times)" for topic, count in most_common])

                insights.append({
                    "category": "interests",
                    "insight": f"You frequently discuss: {topics_str}. These topics seem important to you.",
                    "confidence": 80
                })

            # Sentiment analysis
            sentiments = [conv.get("metadata", {}).get("analysis", {}).get("sentiment")
                         for conv in conversations if conv.get("metadata", {}).get("analysis")]
            sentiment_counts = Counter(sentiments)

            if sentiment_counts:
                dominant_sentiment = sentiment_counts.most_common(1)[0][0]
                insights.append({
                    "category": "emotional_pattern",
                    "insight": f"Your conversations tend to be {dominant_sentiment}. This might reflect your current state of mind.",
                    "confidence": 70
                })

            # Communication style
            styles = [conv.get("metadata", {}).get("analysis", {}).get("communication_style")
                     for conv in conversations if conv.get("metadata", {}).get("analysis")]
            if styles:
                style_counts = Counter(styles)
                main_style = style_counts.most_common(1)[0][0]

                insights.append({
                    "category": "communication",
                    "insight": f"Your communication style is typically {main_style}. This is how you naturally express yourself.",
                    "confidence": 75
                })

        # Analyze journal entries
        if user_data.get("journal_entries"):
            journal_count = len(user_data["journal_entries"])
            insights.append({
                "category": "habits",
                "insight": f"You've written {journal_count} journal entries. Regular journaling shows self-awareness and introspection.",
                "confidence": 90
            })

        return insights

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze uploaded image (basic implementation)"""
        # This is a placeholder for image analysis
        # In a full implementation, you could use computer vision APIs
        return {
            "description": "Image uploaded by user",
            "detected_elements": ["personal photo"],
            "context": "The user is sharing visual aspects of their life"
        }
