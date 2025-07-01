"""
Ensemble Classification Engine - Advanced model combination and decision making
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence level categories"""
    VERY_HIGH = "very_high"  # 0.8-1.0
    HIGH = "high"           # 0.7-0.8
    MEDIUM = "medium"       # 0.5-0.7
    LOW = "low"            # 0.3-0.5
    VERY_LOW = "very_low"  # 0.0-0.3

@dataclass
class PredictionResult:
    """Structured prediction result"""
    intent: str
    confidence: float
    method: str
    model_name: str
    all_scores: Dict[str, float]
    reasoning: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category"""
        if self.confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

class EnsembleStrategy(Enum):
    """Different ensemble combination strategies"""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    MAJORITY_VOTE = "majority_vote"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"

class AdvancedEnsembleClassifier:
    """Advanced ensemble classifier with multiple combination strategies"""
    
    def __init__(self, strategy: EnsembleStrategy = EnsembleStrategy.ADAPTIVE):
        self.strategy = strategy
        self.intent_categories = [
            "Book Appointment",
            "Support Request", 
            "Pricing Negotiation",
            "General Inquiry",
            "Product Information"
        ]
        
        # Model weights based on expected performance
        self.model_weights = {
            "zero-shot": 0.4,
            "semantic_similarity": 0.35,
            "context_patterns": 0.25
        }
        
        # Confidence thresholds for different decision strategies
        self.confidence_thresholds = {
            "high_confidence": 0.75,
            "medium_confidence": 0.5,
            "low_confidence": 0.3
        }
        
        # Performance tracking for adaptive strategy
        self.model_performance_history = {}
    
    def combine_predictions(self, predictions: List[PredictionResult], 
                          conversation_metadata: Dict[str, Any] = None) -> PredictionResult:
        """
        Combine multiple model predictions using the selected strategy
        """
        if not predictions:
            return self._create_fallback_prediction()
        
        # Filter out failed predictions
        valid_predictions = [p for p in predictions if p.confidence > 0]
        
        if not valid_predictions:
            return self._create_fallback_prediction()
        
        # Apply ensemble strategy
        if self.strategy == EnsembleStrategy.ADAPTIVE:
            return self._adaptive_combination(valid_predictions, conversation_metadata)
        elif self.strategy == EnsembleStrategy.CONFIDENCE_BASED:
            return self._confidence_based_combination(valid_predictions)
        elif self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_combination(valid_predictions)
        elif self.strategy == EnsembleStrategy.MAJORITY_VOTE:
            return self._majority_vote_combination(valid_predictions)
        elif self.strategy == EnsembleStrategy.HIERARCHICAL:
            return self._hierarchical_combination(valid_predictions)
        else:
            return self._confidence_based_combination(valid_predictions)
    
    def _adaptive_combination(self, predictions: List[PredictionResult], 
                            conversation_metadata: Dict[str, Any]) -> PredictionResult:
        """Adaptive strategy that chooses best approach based on conversation characteristics"""
        
        if not conversation_metadata:
            return self._confidence_based_combination(predictions)
        
        # Analyze conversation characteristics
        total_turns = conversation_metadata.get("total_turns", 1)
        evolution = conversation_metadata.get("intent_evolution", {})
        escalation = evolution.get("escalation_detected", False)
        
        # For complex conversations, use hierarchical approach
        if total_turns > 5 or escalation:
            return self._hierarchical_combination(predictions)
        
        # For simple conversations, use confidence-based
        elif total_turns <= 2:
            return self._confidence_based_combination(predictions)
        
        # For moderate conversations, use weighted average
        else:
            return self._weighted_average_combination(predictions)
    
    def _confidence_based_combination(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Select prediction with highest confidence, with tie-breaking"""
        
        # Sort by confidence
        sorted_predictions = sorted(predictions, key=lambda x: x.confidence, reverse=True)
        best_prediction = sorted_predictions[0]
        
        # Check for ties (within 0.05 confidence)
        tied_predictions = [p for p in sorted_predictions 
                          if abs(p.confidence - best_prediction.confidence) < 0.05]
        
        if len(tied_predictions) > 1:
            # Break ties using model preference order
            model_preference = ["zero-shot", "semantic_similarity", "context_patterns"]
            for preferred_method in model_preference:
                for pred in tied_predictions:
                    if preferred_method in pred.method:
                        best_prediction = pred
                        break
                if best_prediction.method and preferred_method in best_prediction.method:
                    break
        
        # Create enhanced result
        return PredictionResult(
            intent=best_prediction.intent,
            confidence=best_prediction.confidence,
            method=f"confidence_based({best_prediction.method})",
            model_name=best_prediction.model_name,
            all_scores=best_prediction.all_scores,
            reasoning=self._generate_ensemble_reasoning(predictions, best_prediction, "confidence_based"),
            metadata={
                "ensemble_strategy": "confidence_based",
                "num_models": len(predictions),
                "agreement_level": self._calculate_agreement(predictions)
            }
        )
    
    def _weighted_average_combination(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Combine predictions using weighted average of scores"""
        
        # Initialize score accumulator
        combined_scores = {intent: 0.0 for intent in self.intent_categories}
        total_weight = 0.0
        
        for prediction in predictions:
            # Get model weight
            method_key = prediction.method.split('(')[0]  # Remove parenthetical parts
            weight = self.model_weights.get(method_key, 0.2)
            
            # Add weighted scores
            for intent in self.intent_categories:
                score = prediction.all_scores.get(intent, 0.0)
                combined_scores[intent] += score * weight
            
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            combined_scores = {intent: score / total_weight 
                             for intent, score in combined_scores.items()}
        
        # Get best intent
        best_intent = max(combined_scores, key=combined_scores.get)
        best_confidence = combined_scores[best_intent]
        
        return PredictionResult(
            intent=best_intent,
            confidence=best_confidence,
            method="weighted_average_ensemble",
            model_name="ensemble",
            all_scores=combined_scores,
            reasoning=self._generate_ensemble_reasoning(predictions, None, "weighted_average"),
            metadata={
                "ensemble_strategy": "weighted_average",
                "model_weights": self.model_weights,
                "num_models": len(predictions)
            }
        )
    
    def _majority_vote_combination(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Combine using majority vote with confidence weighting"""
        
        # Count votes for each intent
        intent_votes = {}
        confidence_sum = {}
        
        for prediction in predictions:
            intent = prediction.intent
            confidence = prediction.confidence
            
            if intent not in intent_votes:
                intent_votes[intent] = 0
                confidence_sum[intent] = 0.0
            
            intent_votes[intent] += 1
            confidence_sum[intent] += confidence
        
        # Find majority with highest average confidence
        max_votes = max(intent_votes.values())
        majority_intents = [intent for intent, votes in intent_votes.items() if votes == max_votes]
        
        if len(majority_intents) == 1:
            winning_intent = majority_intents[0]
        else:
            # Break ties using average confidence
            winning_intent = max(majority_intents, 
                               key=lambda intent: confidence_sum[intent] / intent_votes[intent])
        
        avg_confidence = confidence_sum[winning_intent] / intent_votes[winning_intent]
        
        return PredictionResult(
            intent=winning_intent,
            confidence=avg_confidence,
            method="majority_vote_ensemble",
            model_name="ensemble",
            all_scores=intent_votes,
            reasoning=self._generate_ensemble_reasoning(predictions, None, "majority_vote"),
            metadata={
                "ensemble_strategy": "majority_vote",
                "vote_counts": intent_votes,
                "num_models": len(predictions)
            }
        )
    
    def _hierarchical_combination(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Hierarchical decision making based on model reliability"""
        
        # Order models by reliability for different scenarios
        model_hierarchy = [
            ("zero-shot", 0.8),      # Most reliable for clear intents
            ("semantic_similarity", 0.7),  # Good for semantic understanding
            ("context_patterns", 0.6)      # Fallback for pattern matching
        ]
        
        # Find the highest confidence prediction from most reliable available model
        for method, min_confidence in model_hierarchy:
            method_predictions = [p for p in predictions if method in p.method]
            if method_predictions:
                best_from_method = max(method_predictions, key=lambda x: x.confidence)
                if best_from_method.confidence >= min_confidence:
                    return PredictionResult(
                        intent=best_from_method.intent,
                        confidence=best_from_method.confidence,
                        method=f"hierarchical({best_from_method.method})",
                        model_name=best_from_method.model_name,
                        all_scores=best_from_method.all_scores,
                        reasoning=self._generate_ensemble_reasoning(predictions, best_from_method, "hierarchical"),
                        metadata={
                            "ensemble_strategy": "hierarchical",
                            "selected_hierarchy_level": method,
                            "num_models": len(predictions)
                        }
                    )
        
        # Fallback to confidence-based if no model meets hierarchy thresholds
        return self._confidence_based_combination(predictions)
    
    def _calculate_agreement(self, predictions: List[PredictionResult]) -> float:
        """Calculate agreement level between models"""
        if len(predictions) < 2:
            return 1.0
        
        # Count how many models agree on the top intent
        intents = [p.intent for p in predictions]
        most_common_intent = max(set(intents), key=intents.count)
        agreement_count = intents.count(most_common_intent)
        
        return agreement_count / len(predictions)
    
    def _generate_ensemble_reasoning(self, predictions: List[PredictionResult], 
                                   selected_prediction: Optional[PredictionResult],
                                   strategy: str) -> str:
        """Generate detailed reasoning for ensemble decision"""
        
        reasoning_parts = []
        
        # Strategy explanation
        strategy_explanations = {
            "confidence_based": "Selected prediction with highest confidence score",
            "weighted_average": "Combined predictions using weighted average of model scores",
            "majority_vote": "Selected intent based on majority vote with confidence weighting",
            "hierarchical": "Applied hierarchical model selection based on reliability",
            "adaptive": "Used adaptive strategy based on conversation characteristics"
        }
        
        reasoning_parts.append(strategy_explanations.get(strategy, f"Applied {strategy} ensemble strategy"))
        
        # Model agreement analysis
        agreement = self._calculate_agreement(predictions)
        if agreement >= 0.8:
            reasoning_parts.append(f"High model agreement ({agreement:.1%})")
        elif agreement >= 0.6:
            reasoning_parts.append(f"Moderate model agreement ({agreement:.1%})")
        else:
            reasoning_parts.append(f"Low model agreement ({agreement:.1%}) - decision required careful analysis")
        
        # Confidence analysis
        confidences = [p.confidence for p in predictions]
        avg_confidence = np.mean(confidences)
        
        if avg_confidence >= 0.7:
            reasoning_parts.append(f"Strong overall confidence (avg: {avg_confidence:.2f})")
        elif avg_confidence >= 0.5:
            reasoning_parts.append(f"Moderate overall confidence (avg: {avg_confidence:.2f})")
        else:
            reasoning_parts.append(f"Lower confidence requiring ensemble approach (avg: {avg_confidence:.2f})")
        
        # Model contribution summary
        model_methods = [p.method.split('(')[0] for p in predictions]
        reasoning_parts.append(f"Based on {len(predictions)} models: {', '.join(set(model_methods))}")
        
        return ". ".join(reasoning_parts)
    
    def _create_fallback_prediction(self) -> PredictionResult:
        """Create fallback prediction when all models fail"""
        return PredictionResult(
            intent="General Inquiry",
            confidence=0.3,
            method="fallback",
            model_name="ensemble_fallback",
            all_scores={"General Inquiry": 0.3},
            reasoning="Fallback prediction used due to model failures or insufficient input",
            metadata={"ensemble_strategy": "fallback", "num_models": 0}
        )
