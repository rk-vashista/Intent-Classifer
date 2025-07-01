#!/usr/bin/env python3
"""
Demo script comparing Rule-based vs Transformer-based Intent Classification
"""

import json
import time
from intent_classifier_transformer import TransformerIntentClassifier

def load_sample_conversations():
    """Load sample conversations for testing"""
    with open('sample_conversations.json', 'r') as f:
        return json.load(f)

def rule_based_classification(text):
    """Simple rule-based classification for comparison"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['visit', 'appointment', 'meet', 'schedule', 'when can']):
        return "Book Appointment", "Keyword-based classification"
    elif any(word in text_lower for word in ['price', 'cost', 'budget', 'expensive', 'cheap']):
        return "Pricing Negotiation", "Keyword-based classification"
    elif any(word in text_lower for word in ['help', 'problem', 'issue', 'broken', 'not working']):
        return "Support Request", "Keyword-based classification"
    elif any(word in text_lower for word in ['specifications', 'features', 'details', 'what']):
        return "Product Information", "Keyword-based classification"
    else:
        return "General Inquiry", "Default classification"

def main():
    """Main comparison demo"""
    print("🤖 Intent Classification Comparison Demo")
    print("=" * 50)
    
    # Load sample conversations
    conversations = load_sample_conversations()
    
    # Initialize transformer classifier
    print("Loading transformer models...")
    start_time = time.time()
    transformer_classifier = TransformerIntentClassifier()
    load_time = time.time() - start_time
    print(f"✅ Models loaded in {load_time:.1f} seconds\n")
    
    # Process each conversation
    for i, conv in enumerate(conversations, 1):
        print(f"📞 Conversation {i}: {conv['conversation_id']}")
        print("-" * 30)
        
        # Extract conversation text
        messages = conv['messages']
        customer_text = " ".join([msg['text'] for msg in messages if msg['sender'] == 'user'])
        
        print(f"Customer says: {customer_text}")
        print()
        
        # Rule-based classification
        rule_intent, rule_rationale = rule_based_classification(customer_text)
        print(f"🔧 Rule-based: {rule_intent}")
        print(f"   Reason: {rule_rationale}")
        print()
        
        # Transformer-based classification
        start_time = time.time()
        transformer_result = transformer_classifier.classify_intent(messages)
        process_time = time.time() - start_time
        
        print(f"🧠 Transformer: {transformer_result['predicted_intent']}")
        print(f"   Reason: {transformer_result['rationale']}")
        print(f"   Processing time: {process_time:.2f}s")
        print()
        
        # Comparison
        if rule_intent == transformer_result['predicted_intent']:
            print("✅ Both methods agree!")
        else:
            print("❌ Methods disagree - Transformer likely more accurate")
        
        print("=" * 50)
        print()

    print("🎯 Summary:")
    print("- Rule-based: Fast but limited to keywords")
    print("- Transformer: Slower but understands context and meaning")
    print("- Transformer provides confidence scores and detailed reasoning")
    print("- For production use, transformer approach is recommended")

if __name__ == "__main__":
    main()
