"""AI Chatbot Engine for self-discovery and user profiling"""
import os
import json
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter
import re

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class TherapeuticChatbot:
    """Advanced therapeutic chatbot with cognitive awareness and empathy"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_mock = not self.api_key or not OPENAI_AVAILABLE

        if not self.use_mock:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
                self.use_mock = True
                self.client = None
        else:
            self.client = None

        # Track recently used response patterns to avoid repetition
        self.recent_response_types = []
        self.max_recent_memory = 5

    async def generate_response(self, message: str, user_context: Dict) -> Dict:
        """Generate a therapeutic response based on user message and context"""
        if self.use_mock:
            return self._therapeutic_response(message, user_context)

        try:
            # Build context from user profile
            context_prompt = self._build_therapeutic_context(user_context)

            # Include recent conversation history if available
            messages = [{"role": "system", "content": context_prompt}]

            # Add conversation history for context
            if user_context.get("recent_messages"):
                for past_msg in user_context["recent_messages"][-6:]:  # Last 3 exchanges
                    messages.append({"role": "user", "content": past_msg.get("message", "")})
                    messages.append({"role": "assistant", "content": past_msg.get("response", "")})

            messages.append({"role": "user", "content": message})

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.8,
                max_tokens=600
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
            return self._therapeutic_response(message, user_context)

    def _build_therapeutic_context(self, user_context: Dict) -> str:
        """Build therapeutic context prompt"""
        prompt = """You are a professional, empathetic therapist specializing in self-discovery and personal growth.

Your therapeutic approach:
- Use reflective listening: mirror back what you hear to show understanding
- Ask open-ended questions that promote deeper self-exploration
- Validate emotions without judgment
- Help identify patterns in thoughts and behaviors
- Use therapeutic techniques like CBT, mindfulness, and person-centered therapy
- Be warm, genuine, and non-directive
- Avoid being repetitive - each response should offer new perspective
- Never give the same response twice
- Focus on the person's strengths and resilience

"""

        if user_context.get("interests"):
            interests = list(set(user_context['interests']))[:5]
            prompt += f"\nClient's known interests: {', '.join(interests)}"

        if user_context.get("recent_topics"):
            topics = list(set(user_context['recent_topics']))[:5]
            prompt += f"\nRecent discussion topics: {', '.join(topics)}"

        if user_context.get("sentiment_pattern"):
            prompt += f"\nRecent emotional tone: {user_context['sentiment_pattern']}"

        prompt += "\n\nRespond thoughtfully, authentically, and therapeutically. Vary your approach based on what the client needs."

        return prompt

    def _therapeutic_response(self, message: str, user_context: Dict) -> Dict:
        """Generate sophisticated therapeutic response without API"""
        message_lower = message.lower()
        analysis = self.analyze_message(message, user_context)

        # Determine the type of response needed
        response_type, response = self._select_therapeutic_approach(
            message, message_lower, analysis, user_context
        )

        # Track this response type to avoid repetition
        self.recent_response_types.append(response_type)
        if len(self.recent_response_types) > self.max_recent_memory:
            self.recent_response_types.pop(0)

        return {
            "response": response,
            "analysis": analysis
        }

    def _select_therapeutic_approach(self, message: str, message_lower: str,
                                     analysis: Dict, user_context: Dict) -> Tuple[str, str]:
        """Select appropriate therapeutic approach based on message content"""

        # Detect what the user needs
        is_greeting = self._is_greeting(message_lower)
        is_emotional = len(analysis.get("emotional_words", [])) > 0 or analysis.get("sentiment") != "neutral"
        is_question = "?" in message
        is_seeking_advice = any(word in message_lower for word in ["should i", "what do you think", "advice", "help me decide"])
        is_sharing_experience = any(word in message_lower for word in ["today", "yesterday", "happened", "did", "went"])
        is_struggling = any(word in message_lower for word in ["can't", "difficult", "hard", "struggle", "problem", "stuck"])
        is_positive = analysis.get("sentiment") == "positive"
        is_negative = analysis.get("sentiment") == "negative"

        # Get recent messages to understand conversation flow
        recent_messages = user_context.get("recent_messages", [])
        conversation_count = len(recent_messages)

        # Choose approach based on context, avoiding recent patterns
        available_approaches = []

        # GREETING RESPONSES
        if is_greeting and "reflective_greeting" not in self.recent_response_types:
            available_approaches.append(("reflective_greeting", self._greeting_response(user_context, conversation_count)))

        # EMOTIONAL VALIDATION
        if is_emotional and is_negative and "validation" not in self.recent_response_types:
            available_approaches.append(("validation", self._validation_response(message, analysis, user_context)))

        # POSITIVE REINFORCEMENT
        if is_positive and "positive_reinforcement" not in self.recent_response_types:
            available_approaches.append(("positive_reinforcement", self._positive_reinforcement(message, analysis)))

        # REFLECTIVE LISTENING
        if is_sharing_experience and "reflective_listening" not in self.recent_response_types:
            available_approaches.append(("reflective_listening", self._reflective_listening(message, analysis)))

        # PROBLEM EXPLORATION
        if is_struggling and "problem_exploration" not in self.recent_response_types:
            available_approaches.append(("problem_exploration", self._problem_exploration(message, analysis)))

        # SOCRATIC QUESTIONING
        if is_question and "socratic" not in self.recent_response_types:
            available_approaches.append(("socratic", self._socratic_response(message, analysis)))

        # STRENGTH-BASED
        if "strength_based" not in self.recent_response_types:
            available_approaches.append(("strength_based", self._strength_based_response(message, analysis, user_context)))

        # PATTERN RECOGNITION
        if conversation_count > 3 and "pattern_recognition" not in self.recent_response_types:
            available_approaches.append(("pattern_recognition", self._pattern_recognition_response(user_context)))

        # MINDFULNESS/PRESENT MOMENT
        if is_emotional and "mindfulness" not in self.recent_response_types:
            available_approaches.append(("mindfulness", self._mindfulness_response(message, analysis)))

        # EXPLORATORY QUESTIONS
        if "exploratory" not in self.recent_response_types:
            available_approaches.append(("exploratory", self._exploratory_question(message, analysis, user_context)))

        # If we have available approaches, randomly select one
        if available_approaches:
            return random.choice(available_approaches)

        # Fallback to general empathetic response (reset if we've used everything)
        self.recent_response_types = []
        return ("empathetic_general", self._general_empathetic_response(message, analysis))

    def _is_greeting(self, message_lower: str) -> bool:
        """Check if message is a greeting"""
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        return any(greeting in message_lower for greeting in greetings) and len(message_lower.split()) < 10

    def _greeting_response(self, user_context: Dict, conversation_count: int) -> str:
        """Generate appropriate greeting response"""
        username = user_context.get("username", "")

        if conversation_count == 0:
            responses = [
                f"Hello{', ' + username if username else ''}. I'm glad you're here. This is a safe space for you to explore your thoughts and feelings. What's been on your mind lately?",
                f"Welcome{', ' + username if username else ''}. I'm here to listen and support you in understanding yourself better. What would you like to talk about today?",
                f"Hi{', ' + username if username else ''}. Thank you for trusting me with your thoughts. What brings you here today?"
            ]
        else:
            responses = [
                f"Hello again{', ' + username if username else ''}. How have you been since we last spoke?",
                f"Welcome back{', ' + username if username else ''}. What's been happening in your world?",
                f"Hi{', ' + username if username else ''}. It's good to see you again. What would you like to explore today?"
            ]

        return random.choice(responses)

    def _validation_response(self, message: str, analysis: Dict, user_context: Dict) -> str:
        """Validate user's emotions and experiences"""
        emotional_words = analysis.get("emotional_words", [])

        validations = [
            f"I can hear that you're experiencing some difficult emotions right now. It's completely understandable to feel this way, and I want you to know that your feelings are valid.",
            f"Thank you for sharing that with me. What you're feeling sounds really challenging, and it takes courage to acknowledge and express these emotions.",
            f"I appreciate you opening up about this. These feelings you're describing are a natural human response to what you're going through. You're not alone in feeling this way.",
            f"It sounds like you're dealing with something quite difficult. I want you to know that it's okay to feel what you're feeling - there's no wrong emotion here."
        ]

        response = random.choice(validations)

        # Add reflective component
        follow_ups = [
            " Can you tell me more about what's contributing to these feelings?",
            " What do you think might be at the root of this?",
            " How long have you been feeling this way?",
            " What have you noticed helps when you're feeling like this?"
        ]

        return response + random.choice(follow_ups)

    def _positive_reinforcement(self, message: str, analysis: Dict) -> str:
        """Reinforce positive experiences and emotions"""
        responses = [
            "I'm really glad to hear something positive is happening for you. It's important to recognize and celebrate these moments. What about this experience feels particularly meaningful to you?",
            "That sounds wonderful. When good things happen, it can teach us a lot about what brings us joy and fulfillment. What do you think this reveals about what matters to you?",
            "I can sense the positive energy in what you're sharing. These are the moments worth savoring. How does this experience connect with your broader goals or values?",
            "It's beautiful to hear you experiencing something positive. Taking time to really acknowledge and appreciate these feelings can be powerful. What makes this moment special for you?"
        ]

        return random.choice(responses)

    def _reflective_listening(self, message: str, analysis: Dict) -> str:
        """Use reflective listening techniques"""
        # Extract key phrases from the message
        key_topics = analysis.get("topics", [])

        reflections = [
            f"So if I'm understanding correctly, you're sharing that {message[:50]}... Is that right? Tell me more about that.",
            f"I'm hearing that this situation has really impacted you. What stands out most to you about this experience?",
            f"It sounds like this is something that's been weighing on you. What aspect of this feels most important to explore right now?",
            f"What I'm taking from what you're saying is that this matters deeply to you. Help me understand what this means for you."
        ]

        return random.choice(reflections)

    def _problem_exploration(self, message: str, analysis: Dict) -> str:
        """Help explore problems with CBT-inspired questions"""
        explorations = [
            "When you're facing something difficult, it can help to break it down. What do you think is the most challenging aspect of this situation for you?",
            "I hear that this is really hard for you. Let's explore this together - what have you already tried to address this? What worked, and what didn't?",
            "Challenges like this can feel overwhelming. If we were to imagine this situation improving, what would that look like for you? What would be different?",
            "This sounds frustrating. Sometimes our thoughts about a situation can intensify our distress. What thoughts are running through your mind when you're in the middle of this?",
            "When you encounter this difficulty, what do you notice happening in your body and mind? Understanding our reactions can be the first step toward managing them differently."
        ]

        return random.choice(explorations)

    def _socratic_response(self, message: str, analysis: Dict) -> str:
        """Respond to questions with gentle reflection back"""
        responses = [
            "That's a really important question. Before I share my thoughts, I'm curious - what do you think about this? What does your intuition tell you?",
            "I appreciate you asking that. Often, the answers we seek are already within us. What possibilities have you been considering?",
            "That's something worth exploring together. What would it mean for you if you discovered the answer to that question?",
            "I'm glad you're thinking about this. What draws you to ask this particular question right now? What would help you most in answering it?"
        ]

        return random.choice(responses)

    def _strength_based_response(self, message: str, analysis: Dict, user_context: Dict) -> str:
        """Focus on user's strengths and resilience"""
        responses = [
            "I notice that despite what you're going through, you're here, talking about it, and seeking to understand yourself better. That takes real strength and self-awareness.",
            "You're showing a lot of resilience in how you're approaching this. The fact that you're reflecting on these experiences shows growth and maturity.",
            "I want to acknowledge something - the way you're engaging with these challenges demonstrates courage. Many people avoid looking inward like this.",
            "What strikes me is your willingness to be vulnerable and honest about your experience. That's a powerful quality that will serve you well in your personal growth."
        ]

        response = random.choice(responses)

        follow_ups = [
            " What strengths have helped you cope with difficulties in the past?",
            " How have you managed to get through challenges before?",
            " What resources, internal or external, do you have available to you?"
        ]

        return response + random.choice(follow_ups)

    def _pattern_recognition_response(self, user_context: Dict) -> str:
        """Help user recognize patterns in their behavior/thoughts"""
        topics = user_context.get("recent_topics", [])

        if topics:
            topic_counts = Counter(topics)
            common_topics = [t for t, c in topic_counts.most_common(2)]

            if common_topics:
                responses = [
                    f"I've noticed we've been discussing {common_topics[0]} quite a bit in our conversations. What do you think this topic represents for you in your life right now?",
                    f"There seems to be a pattern emerging around themes of {common_topics[0]}. Have you noticed this focus? What might it be telling you about your current needs or concerns?",
                    f"I'm wondering if you've noticed that {common_topics[0]} keeps coming up for you. Sometimes the things we return to repeatedly hold important messages. What do you make of that?"
                ]
                return random.choice(responses)

        return "As we've been talking, I'm noticing certain themes in what you share. What patterns, if any, have you observed in your own thoughts or behaviors lately?"

    def _mindfulness_response(self, message: str, analysis: Dict) -> str:
        """Encourage present-moment awareness"""
        responses = [
            "Let's pause for a moment. Right now, in this present moment, what are you noticing in your body? Sometimes our physical sensations can tell us a lot about our emotional state.",
            "I hear there's a lot going on. If you take a breath and check in with yourself right now, what feeling is most prominent for you in this very moment?",
            "When emotions feel intense, grounding ourselves in the present can help. What's one thing you can notice with your senses right now - something you see, hear, or feel?",
            "Before we go further, I'm curious - if you tune into this very moment, setting aside past and future, what's the strongest sensation or feeling you notice?"
        ]

        return random.choice(responses)

    def _exploratory_question(self, message: str, analysis: Dict, user_context: Dict) -> str:
        """Ask deep exploratory questions"""
        topics = analysis.get("topics", [])

        questions = [
            "I'm interested in understanding more about your inner world. What values or beliefs are most important to you in how you navigate life?",
            "When you think about the person you're becoming, what qualities do you most want to develop or strengthen?",
            "We all have different parts of ourselves. Which part of you is speaking right now, and what does it want you to know?",
            "If you could wave a magic wand and change one thing about your current situation, what would it be? And what would that change make possible for you?",
            "Sometimes we carry stories about ourselves from our past. What story about yourself are you ready to let go of?",
            "What do you think you're learning about yourself through this experience we're discussing?"
        ]

        # Topic-specific questions
        if "work" in topics:
            questions.append("How does your work align with your deeper sense of purpose? What role does it play in your identity?")
        if "relationships" in topics:
            questions.append("In your relationships, what patterns do you notice? What do you tend to give, and what do you tend to receive?")
        if "emotions" in topics:
            questions.append("How did you learn to relate to your emotions? What messages did you receive about expressing feelings?")

        return random.choice(questions)

    def _general_empathetic_response(self, message: str, analysis: Dict) -> str:
        """General empathetic fallback response"""
        responses = [
            "Thank you for sharing that with me. I'm here with you as you explore these thoughts and feelings. What else would you like me to know?",
            "I'm listening carefully to what you're saying, and I want to understand your experience fully. What feels most important to talk about right now?",
            "I appreciate your openness in sharing this. Every experience you describe helps paint a fuller picture of who you are. What comes to mind as you reflect on this?",
            "Being heard and understood is so important. I'm here to listen without judgment. What would be most helpful for us to focus on together?"
        ]

        return random.choice(responses)

    def analyze_message(self, message: str, user_context: Dict) -> Dict:
        """Comprehensive message analysis"""
        analysis = {
            "word_count": len(message.split()),
            "sentiment": self._detect_sentiment(message),
            "topics": self._extract_topics(message),
            "questions_asked": message.count("?"),
            "emotional_words": self._detect_emotional_words(message),
            "communication_style": self._detect_communication_style(message),
            "emotional_intensity": self._assess_emotional_intensity(message),
            "is_crisis": self._detect_crisis_language(message)
        }

        return analysis

    def _detect_sentiment(self, text: str) -> str:
        """Enhanced sentiment detection"""
        positive_words = [
            "happy", "joy", "excited", "love", "great", "amazing", "wonderful", "good",
            "fantastic", "excellent", "grateful", "thankful", "blessed", "proud", "hopeful",
            "delighted", "pleased", "cheerful", "content", "satisfied"
        ]
        negative_words = [
            "sad", "angry", "upset", "hate", "terrible", "awful", "bad", "frustrated",
            "depressed", "anxious", "worried", "scared", "lonely", "hopeless", "worthless",
            "miserable", "devastated", "overwhelmed", "exhausted", "disappointed"
        ]

        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        # Weight by intensity
        if negative_count > positive_count + 1:
            return "negative"
        elif positive_count > negative_count + 1:
            return "positive"
        else:
            return "neutral"

    def _extract_topics(self, text: str) -> List[str]:
        """Enhanced topic extraction"""
        topic_keywords = {
            "work": ["work", "job", "career", "office", "boss", "colleague", "employee", "workplace", "professional"],
            "family": ["family", "parent", "mother", "father", "sibling", "child", "mom", "dad", "brother", "sister"],
            "health": ["health", "exercise", "sleep", "diet", "wellness", "fitness", "medical", "doctor", "therapy"],
            "relationships": ["friend", "relationship", "dating", "partner", "love", "boyfriend", "girlfriend", "marriage", "connection"],
            "hobbies": ["hobby", "music", "art", "reading", "gaming", "sports", "creative", "passion", "interest"],
            "emotions": ["feel", "emotion", "anxiety", "stress", "happiness", "sadness", "anger", "fear", "worry"],
            "goals": ["goal", "dream", "aspiration", "future", "plan", "ambition", "hope", "want", "achieve"],
            "self": ["myself", "identity", "who i am", "self", "me", "i'm", "personal", "my life"],
            "past": ["childhood", "past", "history", "remember", "used to", "before", "trauma"],
            "change": ["change", "different", "transition", "growth", "evolve", "transform", "new"]
        }

        text_lower = text.lower()
        detected_topics = []

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)

        return detected_topics

    def _detect_emotional_words(self, text: str) -> List[str]:
        """Enhanced emotional word detection"""
        emotional_words = [
            "happy", "sad", "angry", "excited", "anxious", "stressed",
            "frustrated", "joy", "fear", "worried", "calm", "peaceful",
            "love", "hate", "confused", "confident", "insecure", "lonely",
            "hopeful", "hopeless", "grateful", "ashamed", "guilty", "proud",
            "overwhelmed", "content", "disappointed", "scared", "nervous"
        ]

        words = re.findall(r'\b\w+\b', text.lower())
        found_emotions = [word for word in words if word in emotional_words]

        return list(set(found_emotions))  # Remove duplicates

    def _detect_communication_style(self, text: str) -> str:
        """Detect communication style"""
        word_count = len(text.split())

        if word_count < 5:
            return "brief"
        elif word_count > 50:
            return "detailed"
        elif "?" in text:
            return "inquisitive"
        elif any(word in text.lower() for word in ["i think", "i believe", "in my opinion", "i feel like"]):
            return "reflective"
        else:
            return "conversational"

    def _assess_emotional_intensity(self, text: str) -> str:
        """Assess the intensity of emotions expressed"""
        intense_words = [
            "extremely", "very", "really", "so", "absolutely", "completely",
            "totally", "utterly", "devastated", "ecstatic", "terrified"
        ]

        text_lower = text.lower()
        intense_count = sum(1 for word in intense_words if word in text_lower)

        # Check for ALL CAPS (indicates strong emotion)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # Check for multiple exclamation marks
        exclamation_count = text.count("!")

        if intense_count > 2 or caps_ratio > 0.3 or exclamation_count > 2:
            return "high"
        elif intense_count > 0 or exclamation_count > 0:
            return "moderate"
        else:
            return "low"

    def _detect_crisis_language(self, text: str) -> bool:
        """Detect if user might be in crisis (simplified check)"""
        crisis_words = [
            "suicide", "kill myself", "end it all", "want to die",
            "no reason to live", "better off dead", "harm myself"
        ]

        text_lower = text.lower()
        return any(phrase in text_lower for phrase in crisis_words)

    def generate_insights(self, user_data: Dict) -> List[Dict]:
        """Generate deep, meaningful insights from user data"""
        insights = []

        # Analyze conversation patterns
        if user_data.get("conversations"):
            conversations = user_data["conversations"]

            # Topic frequency analysis
            all_topics = []
            for conv in conversations:
                if conv.get("metadata", {}).get("analysis", {}).get("topics"):
                    all_topics.extend(conv["metadata"]["analysis"]["topics"])

            if all_topics and len(all_topics) > 3:
                topic_counts = Counter(all_topics)
                most_common = topic_counts.most_common(3)

                # Create more meaningful insights
                primary_topic = most_common[0][0]
                primary_count = most_common[0][1]

                topic_insights = {
                    "work": f"You've discussed work-related topics {primary_count} times. This suggests your professional life is significantly on your mind. Consider whether work is fulfilling your deeper needs or if there's a balance issue to explore.",
                    "relationships": f"Relationships have come up {primary_count} times in our conversations. Connection with others seems to be a central theme for you. This shows emotional intelligence and a value for meaningful bonds.",
                    "emotions": f"You've explored your emotions {primary_count} times, showing strong self-awareness and emotional intelligence. This capacity for introspection is a valuable strength.",
                    "goals": f"You've mentioned goals and aspirations {primary_count} times, indicating you're future-oriented and growth-minded. You're actively thinking about who you want to become.",
                    "family": f"Family has been a recurring theme ({primary_count} times). These relationships clearly hold significant meaning in your life and may be shaping your current experiences."
                }

                insight_text = topic_insights.get(
                    primary_topic,
                    f"You frequently discuss {primary_topic}, which appears {primary_count} times. This recurring theme suggests it plays an important role in your life right now."
                )

                insights.append({
                    "category": "core_themes",
                    "insight": insight_text,
                    "confidence": 85
                })

            # Emotional pattern analysis
            sentiments = []
            emotional_words_all = []
            for conv in conversations:
                metadata = conv.get("metadata", {})
                if metadata.get("analysis"):
                    sentiments.append(metadata["analysis"].get("sentiment", "neutral"))
                    emotional_words_all.extend(metadata["analysis"].get("emotional_words", []))

            if len(sentiments) > 3:
                sentiment_counts = Counter(sentiments)
                dominant_sentiment = sentiment_counts.most_common(1)[0][0]
                sentiment_percentage = (sentiment_counts[dominant_sentiment] / len(sentiments)) * 100

                sentiment_insights = {
                    "positive": f"Your conversations tend to be positive ({sentiment_percentage:.0f}% of the time). This suggests an optimistic outlook or that you're experiencing a good phase in life. Continue nurturing what brings you joy.",
                    "negative": f"I notice {sentiment_percentage:.0f}% of our conversations have a negative emotional tone. You may be going through a challenging period. Remember, seeking support and expressing difficult emotions is a sign of strength, not weakness.",
                    "neutral": f"Your emotional expression tends to be balanced and measured. This could indicate emotional regulation skills, or it might be worth exploring if you feel comfortable fully expressing your emotions."
                }

                insights.append({
                    "category": "emotional_patterns",
                    "insight": sentiment_insights.get(dominant_sentiment, "Your emotional patterns show variety."),
                    "confidence": 80
                })

            # Communication style insight
            if len(conversations) > 5:
                styles = [conv.get("metadata", {}).get("analysis", {}).get("communication_style")
                         for conv in conversations if conv.get("metadata", {}).get("analysis")]

                if styles:
                    style_counts = Counter(styles)
                    main_style = style_counts.most_common(1)[0][0]

                    style_insights = {
                        "brief": "You tend to communicate briefly and concisely. This efficiency is valuable, though sometimes elaborating on your feelings might help you process them more deeply.",
                        "detailed": "You express yourself in detail, which shows thoughtfulness and a desire to be understood fully. This depth of communication can lead to profound insights.",
                        "reflective": "Your communication is often reflective and introspective. This self-awareness is a powerful tool for personal growth and understanding.",
                        "inquisitive": "You frequently ask questions, showing curiosity and a desire to understand. This openness to learning about yourself is admirable."
                    }

                    insights.append({
                        "category": "communication_style",
                        "insight": style_insights.get(main_style, f"Your communication style is typically {main_style}."),
                        "confidence": 75
                    })

        # Journal entry insights
        if user_data.get("journal_entries"):
            journal_count = len(user_data["journal_entries"])

            if journal_count >= 5:
                insights.append({
                    "category": "self_care_practices",
                    "insight": f"You've written {journal_count} journal entries, demonstrating commitment to self-reflection and personal growth. This practice of regular introspection is associated with increased self-awareness, emotional processing, and mental clarity.",
                    "confidence": 90
                })
            elif journal_count > 0:
                insights.append({
                    "category": "emerging_practices",
                    "insight": f"You've started journaling ({journal_count} entries so far). This is a wonderful beginning to a practice that can deepen self-understanding. Consider making it a regular habit.",
                    "confidence": 85
                })

        # Growth and resilience insight
        if len(insights) > 0:
            insights.append({
                "category": "personal_growth",
                "insight": "Your willingness to engage in self-discovery through our conversations shows courage and a commitment to personal growth. This investment in understanding yourself is one of the most valuable things you can do.",
                "confidence": 95
            })

        return insights

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze uploaded image with more meaningful context"""
        return {
            "description": "Thank you for sharing this image with me",
            "detected_elements": ["personal moment captured"],
            "context": "Visual memories like this can be powerful. What does this image represent for you? What feelings or memories does it bring up?"
        }


# Backward compatibility alias
ChatbotEngine = TherapeuticChatbot
