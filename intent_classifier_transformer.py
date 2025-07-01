"""
Multi-Turn Intent Classification System using Transformers
A system to classify customer intents from WhatsApp-style conversations
using state-of-the-art transformer models.
"""

import json
import pandas as pd
from typing import List, Dict, Any
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    pipeline,
    BartForSequenceClassification
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerIntentClassifier:
    def __init__(self):
        # Define the 5 intent categories
        self.intent_categories = [
            "Book Appointment",
            "Support Request", 
            "Pricing Negotiation",
            "General Inquiry",
            "Product Information"
        ]
        
        # Initialize transformer models
        self.setup_models()
        
        # Define intent descriptions for better semantic matching
        self.intent_descriptions = {
            "Book Appointment": "Customer wants to schedule a meeting, appointment, or visit. They are looking for available time slots or dates to meet.",
            "Support Request": "Customer needs help with a problem, issue, or technical difficulty. They are seeking assistance, troubleshooting, or customer service support.",
            "Pricing Negotiation": "Customer is discussing costs, prices, discounts, or budget-related concerns. They want pricing information, deals, or are negotiating costs.",
            "General Inquiry": "Customer is asking general questions or seeking basic information about services, processes, or company policies.",
            "Product Information": "Customer wants detailed information about products, features, specifications, availability, or available options and models."
        }
    
    def setup_models(self):
        """
        Initialize transformer models for intent classification
        """
        try:
            logger.info("Loading transformer models...")
            
            # Use BART for zero-shot classification
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Use sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Use a conversation-aware model for context understanding
            self.conversation_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("✅ All transformer models loaded successfully!")
            
        except Exception as e:
            logger.error(f"⚠️ Error loading transformer models: {e}")
            logger.info("💡 Installing required packages...")
            self._install_dependencies()
            
    def _install_dependencies(self):
        """Install required packages if not available"""
        import subprocess
        import sys
        
        packages = [
            "torch",
            "transformers",
            "sentence-transformers", 
            "scikit-learn",
            "numpy"
        ]
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError:
                logger.error(f"Failed to install {package}")
    
    def preprocess_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """
        Convert multi-turn conversation into a comprehensive context string
        """
        if not conversation:
            return ""
        
        # Separate customer and agent messages
        customer_messages = []
        agent_messages = []
        
        for message in conversation:
            # Handle different field names (role/sender and message/text)
            sender = message.get('sender', message.get('role', '')).lower()
            text = message.get('text', message.get('message', '')).strip()
            
            if text:  # Only include non-empty messages
                if sender in ['customer', 'user']:
                    customer_messages.append(text)
                elif sender in ['agent', 'assistant', 'support']:
                    agent_messages.append(text)
        
        # Create a comprehensive context
        context_parts = []
        
        # Add customer intent summary
        if customer_messages:
            customer_context = " ".join(customer_messages)
            context_parts.append(f"Customer says: {customer_context}")
        
        # Add agent responses for context
        if agent_messages:
            agent_context = " ".join(agent_messages[-2:])  # Last 2 agent responses
            context_parts.append(f"Agent responded: {agent_context}")
        
        return " | ".join(context_parts)
    
    def classify_with_zero_shot(self, text: str) -> Dict[str, Any]:
        """
        Classify intent using zero-shot classification
        """
        try:
            # Create candidate labels with better descriptions
            candidate_labels = [
                "booking appointment scheduling meeting visit",
                "requesting help support assistance problem issue",
                "discussing price cost budget negotiation discount",
                "asking general questions basic information inquiry",
                "seeking product information features specifications details"
            ]
            
            result = self.zero_shot_classifier(text, candidate_labels)
            
            # Map back to intent categories
            label_mapping = {
                "booking appointment scheduling meeting visit": "Book Appointment",
                "requesting help support assistance problem issue": "Support Request",
                "discussing price cost budget negotiation discount": "Pricing Negotiation",
                "asking general questions basic information inquiry": "General Inquiry",
                "seeking product information features specifications details": "Product Information"
            }
            
            predicted_label = result['labels'][0]
            predicted_intent = label_mapping.get(predicted_label, "General Inquiry")
            
            return {
                "predicted_intent": predicted_intent,
                "confidence": result['scores'][0],
                "all_scores": {label_mapping.get(label, label): score 
                             for label, score in zip(result['labels'], result['scores'])},
                "method": "zero-shot"
            }
        except Exception as e:
            logger.error(f"Zero-shot classification failed: {e}")
            return None
    
    def classify_with_semantic_similarity(self, text: str) -> Dict[str, Any]:
        """
        Classify intent using semantic similarity with intent descriptions
        """
        try:
            # Encode the input text
            text_embedding = self.sentence_model.encode([text])
            
            # Encode intent descriptions
            description_embeddings = self.sentence_model.encode(
                list(self.intent_descriptions.values())
            )
            
            # Calculate similarities
            similarities = cosine_similarity(text_embedding, description_embeddings)[0]
            
            # Get best match
            best_match_idx = np.argmax(similarities)
            intent_names = list(self.intent_descriptions.keys())
            
            return {
                "predicted_intent": intent_names[best_match_idx],
                "confidence": float(similarities[best_match_idx]),
                "all_scores": {intent: float(sim) for intent, sim in zip(intent_names, similarities)},
                "method": "semantic_similarity"
            }
        except Exception as e:
            logger.error(f"Semantic similarity classification failed: {e}")
            return None
    
    def ensemble_classification(self, text: str) -> Dict[str, str]:
        """
        Combine multiple transformer approaches for robust classification
        """
        # Get predictions from different methods
        zero_shot_result = self.classify_with_zero_shot(text)
        semantic_result = self.classify_with_semantic_similarity(text)
        
        # Decision logic for ensemble
        final_intent = None
        final_confidence = 0
        method_used = "ensemble"
        
        if zero_shot_result and semantic_result:
            # If both methods agree, use that with higher confidence
            if zero_shot_result['predicted_intent'] == semantic_result['predicted_intent']:
                final_intent = zero_shot_result['predicted_intent']
                final_confidence = max(zero_shot_result['confidence'], semantic_result['confidence'])
                method_used = "consensus"
            
            # If they disagree, use the one with higher confidence
            elif zero_shot_result['confidence'] > semantic_result['confidence']:
                final_intent = zero_shot_result['predicted_intent']
                final_confidence = zero_shot_result['confidence']
                method_used = "zero-shot (higher confidence)"
            else:
                final_intent = semantic_result['predicted_intent']
                final_confidence = semantic_result['confidence']
                method_used = "semantic similarity (higher confidence)"
        
        elif zero_shot_result:
            final_intent = zero_shot_result['predicted_intent']
            final_confidence = zero_shot_result['confidence']
            method_used = "zero-shot only"
        
        elif semantic_result:
            final_intent = semantic_result['predicted_intent']
            final_confidence = semantic_result['confidence']
            method_used = "semantic similarity only"
        
        else:
            # Fallback to most common intent
            final_intent = "General Inquiry"
            final_confidence = 0.5
            method_used = "fallback"
        
        # Generate rationale
        rationale = self._generate_rationale(final_intent, final_confidence, method_used)
        
        return {
            "predicted_intent": final_intent,
            "rationale": rationale
        }
    
    def classify_intent(self, conversation: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Main method to classify the intent of a multi-turn conversation
        """
        # Preprocess conversation
        text = self.preprocess_conversation(conversation)
        
        if not text.strip():
            return {
                "predicted_intent": "General Inquiry",
                "rationale": "Empty or invalid conversation content"
            }
        
        # Use ensemble classification
        return self.ensemble_classification(text)
    
    def _generate_rationale(self, intent: str, confidence: float, method: str) -> str:
        """
        Generate a detailed rationale for the prediction
        """
        base_rationales = {
            "Book Appointment": "Customer expressed intent to schedule, book, or arrange a meeting/appointment",
            "Support Request": "Customer is seeking help, reporting issues, or requesting technical assistance",
            "Pricing Negotiation": "Customer is discussing, negotiating, or inquiring about costs and pricing", 
            "General Inquiry": "Customer made general questions or requests for basic information",
            "Product Information": "Customer is seeking detailed information about products, features, or specifications"
        }
        
        base_rationale = base_rationales.get(intent, "Intent classified using transformer models")
        
        # Add confidence and method information
        confidence_level = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
        
        return f"{base_rationale}. Classified using {method} with {confidence_level} confidence ({confidence:.2f})"


def process_conversations(input_file: str, output_json: str = "predictions.json", output_csv: str = "predictions.csv"):
    """
    Process conversations from input file and generate predictions
    """
    logger.info(f"Loading conversations from {input_file}")
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different input formats
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict) and 'conversations' in data:
        conversations = data['conversations']
    else:
        conversations = [data]  # Single conversation
    
    logger.info(f"Found {len(conversations)} conversations")
    
    # Initialize classifier
    classifier = TransformerIntentClassifier()
    
    # Process each conversation
    results = []
    for conv in tqdm(conversations, desc="Classifying intents"):
        conv_id = conv.get('conversation_id', f"conv_{len(results)+1:03d}")
        messages = conv.get('messages', [])
        
        # Classify intent
        prediction = classifier.classify_intent(messages)
        
        result = {
            "conversation_id": conv_id,
            "predicted_intent": prediction["predicted_intent"],
            "rationale": prediction["rationale"]
        }
        results.append(result)
    
    # Save results
    logger.info(f"Saving results to {output_json} and {output_csv}")
    
    # Save JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Print summary
    logger.info("Classification Summary:")
    intent_counts = df['predicted_intent'].value_counts()
    for intent, count in intent_counts.items():
        logger.info(f"  {intent}: {count}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Transformer-based Multi-Turn Intent Classification")
    parser.add_argument("input_file", help="Path to input JSON file with conversations")
    parser.add_argument("--output-json", default="predictions.json", help="Output JSON file path")
    parser.add_argument("--output-csv", default="predictions.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        return
    
    # Process conversations
    process_conversations(args.input_file, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()
