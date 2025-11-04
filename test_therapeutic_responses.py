"""Test the therapeutic chatbot's varied responses and cognitive awareness"""
import asyncio
from chatbot_engine import TherapeuticChatbot

def test_varied_responses():
    """Test that the chatbot provides varied, non-repetitive responses"""
    print("="*70)
    print("Testing Therapeutic Chatbot - Varied Responses & Cognitive Awareness")
    print("="*70)

    bot = TherapeuticChatbot()

    # Test different scenarios
    test_messages = [
        ("Hi there", "Greeting"),
        ("I'm feeling really stressed about work", "Negative emotion"),
        ("I had an amazing day today!", "Positive emotion"),
        ("I can't seem to make progress on my goals", "Struggle"),
        ("What do you think I should do about my relationship?", "Seeking advice"),
        ("Yesterday I went to the park and felt peaceful", "Sharing experience"),
        ("I'm feeling anxious and overwhelmed", "Strong negative emotion"),
        ("I keep thinking about the same problems", "Pattern discussion"),
        ("I don't know who I am anymore", "Identity question"),
        ("Things have been pretty normal lately", "Neutral")
    ]

    user_context = {
        "username": "TestUser",
        "interests": ["reading", "hiking"],
        "recent_topics": ["work", "emotions"],
        "recent_messages": []
    }

    print("\nTesting 10 different therapeutic responses:\n")

    responses = []
    for i, (message, category) in enumerate(test_messages, 1):
        result = asyncio.run(bot.generate_response(message, user_context))
        response = result["response"]
        responses.append(response)

        print(f"{i}. [{category}]")
        print(f"   User: {message}")
        print(f"   Bot: {response[:150]}...")
        print(f"   Analysis: {result['analysis']['sentiment']} sentiment, " +
              f"{len(result['analysis']['emotional_words'])} emotional words")
        print()

        # Add to context for next iteration
        user_context["recent_messages"].append({
            "message": message,
            "response": response
        })

    # Check for repetition
    print("="*70)
    print("Checking for repetition...")
    print("="*70)

    # Check if any responses are exactly the same
    unique_responses = set(responses)
    repetition_rate = (len(responses) - len(unique_responses)) / len(responses) * 100

    print(f"\nTotal responses: {len(responses)}")
    print(f"Unique responses: {len(unique_responses)}")
    print(f"Repetition rate: {repetition_rate:.1f}%")

    if repetition_rate < 10:
        print("✓ PASSED: Responses are highly varied (< 10% repetition)")
    else:
        print("✗ WARNING: Some repetition detected")

    # Check response diversity
    avg_length = sum(len(r) for r in responses) / len(responses)
    print(f"\nAverage response length: {avg_length:.0f} characters")
    print(f"Shortest response: {min(len(r) for r in responses)} characters")
    print(f"Longest response: {max(len(r) for r in responses)} characters")

    # Test that responses address the user's specific situation
    print("\n" + "="*70)
    print("Response Quality Check")
    print("="*70)

    quality_checks = [
        ("Empathy indicators", ["understand", "hear", "appreciate", "notice", "sounds", "seems"]),
        ("Questions (encouraging exploration)", ["?", "what", "how", "why"]),
        ("Validation words", ["valid", "okay", "natural", "understandable", "courage"]),
        ("Growth language", ["learn", "grow", "discover", "insight", "understand yourself"])
    ]

    for check_name, keywords in quality_checks:
        count = sum(1 for response in responses if any(keyword in response.lower() for keyword in keywords))
        percentage = (count / len(responses)) * 100
        print(f"\n{check_name}: {count}/{len(responses)} responses ({percentage:.0f}%)")
        if percentage >= 50:
            print(f"  ✓ Good therapeutic quality")
        elif percentage >= 30:
            print(f"  ~ Moderate presence")
        else:
            print(f"  ✗ Could be improved")

    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)


if __name__ == "__main__":
    test_varied_responses()
