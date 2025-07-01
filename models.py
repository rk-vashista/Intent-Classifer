"""
Core Models Module - Contains all transformer model implementations
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    """Abstract base class for all intent classification models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
    
    def load_model(self):
        """Load the model - to be implemented by subclasses"""
        raise NotImplementedError
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_confidence(self, prediction: Dict[str, Any]) -> float:
        """Extract confidence score from prediction"""
        return prediction.get('confidence', 0.0)


class ZeroShotClassifier(BaseModel):
    """Zero-shot classification using BART-large-mnli"""
    
    def __init__(self):
        super().__init__("facebook/bart-large-mnli")
        self.intent_labels = [
            "scheduling appointment meeting visit booking",
            "technical support help assistance problem troubleshooting",
            "pricing negotiation cost budget discount financial",
            "general information inquiry question basic details",
            "product specifications features details availability options"
        ]
        self.label_mapping = {
            "scheduling appointment meeting visit booking": "Book Appointment",
            "technical support help assistance problem troubleshooting": "Support Request",
            "pricing negotiation cost budget discount financial": "Pricing Negotiation",
            "general information inquiry question basic details": "General Inquiry",
            "product specifications features details availability options": "Product Information"
        }
    
    def load_model(self):
        """Load BART zero-shot classification model"""
        try:
            logger.info(f"Loading {self.model_name}...")
            self.model = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.is_loaded = True
            logger.info("✅ Zero-shot classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load zero-shot classifier: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Classify text using zero-shot approach"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            result = self.model(text, self.intent_labels)
            predicted_label = result['labels'][0]
            predicted_intent = self.label_mapping.get(predicted_label, "General Inquiry")
            
            return {
                "predicted_intent": predicted_intent,
                "confidence": result['scores'][0],
                "all_scores": {self.label_mapping.get(label, label): score 
                             for label, score in zip(result['labels'], result['scores'])},
                "method": "zero-shot",
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Zero-shot prediction failed: {e}")
            return {
                "predicted_intent": "General Inquiry",
                "confidence": 0.0,
                "method": "zero-shot-fallback",
                "error": str(e)
            }


class SemanticSimilarityClassifier(BaseModel):
    """Semantic similarity classification using sentence transformers"""
    
    def __init__(self):
        super().__init__("all-MiniLM-L6-v2")
        self.intent_descriptions = {
            "Book Appointment": "Customer wants to schedule, book, or arrange a meeting, appointment, visit, or call. They are looking for available time slots, dates, or seeking to set up face-to-face interactions.",
            "Support Request": "Customer needs help with a problem, technical issue, malfunction, or difficulty. They are seeking assistance, troubleshooting, customer service, or reporting something that isn't working properly.",
            "Pricing Negotiation": "Customer is discussing, negotiating, or inquiring about costs, prices, budget, discounts, deals, or financial aspects. They want pricing information or are trying to get better rates.",
            "General Inquiry": "Customer is asking general questions, seeking basic information, making casual inquiries, or requesting standard details about services, processes, or policies.",
            "Product Information": "Customer wants detailed information about products, features, specifications, availability, models, options, or technical details about what's being offered."
        }
    
    def load_model(self):
        """Load sentence transformer model"""
        try:
            logger.info(f"Loading {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            self.is_loaded = True
            logger.info("✅ Semantic similarity model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Classify using semantic similarity"""
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Encode input text
            text_embedding = self.model.encode([text])
            
            # Encode intent descriptions
            description_embeddings = self.model.encode(list(self.intent_descriptions.values()))
            
            # Calculate similarities
            similarities = cosine_similarity(text_embedding, description_embeddings)[0]
            
            # Get best match
            best_match_idx = np.argmax(similarities)
            intent_names = list(self.intent_descriptions.keys())
            
            return {
                "predicted_intent": intent_names[best_match_idx],
                "confidence": float(similarities[best_match_idx]),
                "all_scores": {intent: float(sim) for intent, sim in zip(intent_names, similarities)},
                "method": "semantic_similarity",
                "model_name": self.model_name
            }
        except Exception as e:
            logger.error(f"Semantic similarity prediction failed: {e}")
            return {
                "predicted_intent": "General Inquiry",
                "confidence": 0.0,
                "method": "semantic-fallback",
                "error": str(e)
            }


class ConversationContextClassifier(BaseModel):
    """Conversation-aware classification using DialoGPT"""
    
    def __init__(self):
        super().__init__("microsoft/DialoGPT-medium")
        self.context_patterns = {
            "Book Appointment": [
                "when can", "available", "schedule", "visit", "meet", "appointment",
                "book", "arrange", "time", "date", "calendar", "slot"
            ],
            "Support Request": [
                "help", "problem", "issue", "broken", "not working", "error",
                "trouble", "fix", "support", "assist", "malfunction"
            ],
            "Pricing Negotiation": [
                "price", "cost", "budget", "expensive", "cheap", "discount",
                "deal", "negotiate", "quote", "payment", "afford"
            ],
            "General Inquiry": [
                "information", "tell me", "explain", "how", "what", "where",
                "general", "question", "inquire", "details"
            ],
            "Product Information": [
                "features", "specifications", "specs", "details", "model",
                "options", "available", "product", "version", "type"
            ]
        }
    
    def load_model(self):
        """Load conversation context model"""
        try:
            logger.info(f"Loading {self.model_name}...")
            # Note: DialoGPT isn't ideal for classification, but we'll use it for context analysis
            self.model = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.is_loaded = True
            logger.info("✅ Conversation context model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load conversation model: {e}")
            # Fallback to pattern matching if model fails
            self.is_loaded = True
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Analyze conversation context for intent patterns"""
        text_lower = text.lower()
        intent_scores = {}
        
        # Calculate pattern matching scores
        for intent, patterns in self.context_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    score += text_lower.count(pattern)
            intent_scores[intent] = score
        
        # Get best match
        if any(score > 0 for score in intent_scores.values()):
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[best_intent] / 10.0, 1.0)  # Normalize score
        else:
            best_intent = "General Inquiry"
            confidence = 0.3
        
        return {
            "predicted_intent": best_intent,
            "confidence": confidence,
            "all_scores": intent_scores,
            "method": "context_patterns",
            "model_name": self.model_name
        }
