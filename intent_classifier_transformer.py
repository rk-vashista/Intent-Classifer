"""
Multi-Turn Intent Classification System using Transformers
A modular, high-performance system for classifying customer intents from conversations
using state-of-the-art transformer models with advanced reasoning capabilities.
"""

import json
import pandas as pd
from typing import List, Dict, Any
import logging
from pathlib import Path
import argparse
from tqdm import tqdm
from dataclasses import dataclass

# Import modular components
from models import ZeroShotClassifier, SemanticSimilarityClassifier, ConversationContextClassifier
from conversation_processor import ConversationPreprocessor
from ensemble import AdvancedEnsembleClassifier, EnsembleStrategy
from reasoning_engine import IntelligentReasoningEngine, ReasoningDepth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerIntentClassifier:
    """
    Main classifier that orchestrates all modular components for maximum accuracy
    """
    
    def __init__(self, ensemble_strategy: EnsembleStrategy = EnsembleStrategy.ADAPTIVE,
                 reasoning_depth: ReasoningDepth = ReasoningDepth.COMPREHENSIVE):
        """
        Initialize the modular intent classification system
        
        Args:
            ensemble_strategy: Strategy for combining model predictions
            reasoning_depth: Level of detail in reasoning explanations
        """
        logger.info("Initializing Modular Transformer Intent Classification System...")
        
        # Initialize all modular components
        self.conversation_processor = ConversationPreprocessor(
            max_turns=10, 
            enable_context_analysis=True
        )
        
        self.ensemble_classifier = AdvancedEnsembleClassifier(strategy=ensemble_strategy)
        
        self.reasoning_engine = IntelligentReasoningEngine(reasoning_depth=reasoning_depth)
        
        # Initialize transformer models
        self.models = {
            "zero_shot": ZeroShotClassifier(),
            "semantic": SemanticSimilarityClassifier(), 
            "context": ConversationContextClassifier()
        }
        
        self.intent_categories = [
            "Book Appointment",
            "Support Request", 
            "Pricing Negotiation",
            "General Inquiry",
            "Product Information"
        ]
        
        logger.info("Loading transformer models...")
        self._load_all_models()
        logger.info("✅ All components initialized successfully!")
    
    def _load_all_models(self):
        """Load all transformer models with error handling"""
        for name, model in self.models.items():
            try:
                model.load_model()
                logger.info(f"✅ {name} model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {name} model: {e}")
                # Continue with other models
    
    def classify_intent(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Main method to classify intent with comprehensive analysis
        
        Args:
            conversation: List of message dictionaries
            
        Returns:
            Comprehensive classification result with reasoning
        """
        try:
            # 1. Advanced conversation preprocessing
            conversation_text, conversation_metadata = self.conversation_processor.preprocess_conversation(conversation)
            
            if not conversation_text:
                return {
                    "predicted_intent": "General Inquiry",
                    "confidence": 0.3,
                    "rationale": "Empty or invalid conversation - defaulted to General Inquiry",
                    "metadata": conversation_metadata
                }
            
            # 2. Get predictions from all available models
            model_predictions = []
            
            # Convert model prediction dicts to PredictionResult objects
            from ensemble import PredictionResult
            
            # Zero-shot classification
            if self.models["zero_shot"].is_loaded:
                zero_shot_result = self.models["zero_shot"].predict(conversation_text)
                if zero_shot_result and "error" not in zero_shot_result:
                    prediction_obj = PredictionResult(
                        intent=zero_shot_result["predicted_intent"],
                        confidence=zero_shot_result["confidence"],
                        method=zero_shot_result["method"],
                        model_name=zero_shot_result.get("model_name", "facebook/bart-large-mnli"),
                        all_scores=zero_shot_result["all_scores"]
                    )
                    model_predictions.append(prediction_obj)
            
            # Semantic similarity classification
            if self.models["semantic"].is_loaded:
                semantic_result = self.models["semantic"].predict(conversation_text)
                if semantic_result and "error" not in semantic_result:
                    prediction_obj = PredictionResult(
                        intent=semantic_result["predicted_intent"],
                        confidence=semantic_result["confidence"],
                        method=semantic_result["method"],
                        model_name=semantic_result.get("model_name", "all-MiniLM-L6-v2"),
                        all_scores=semantic_result["all_scores"]
                    )
                    model_predictions.append(prediction_obj)
            
            # Conversation context classification
            if self.models["context"].is_loaded:
                context_result = self.models["context"].predict(conversation_text)
                if context_result and "error" not in context_result:
                    prediction_obj = PredictionResult(
                        intent=context_result["predicted_intent"],
                        confidence=context_result["confidence"],
                        method=context_result["method"],
                        model_name=context_result.get("model_name", "microsoft/DialoGPT-medium"),
                        all_scores=context_result.get("all_scores", {context_result["predicted_intent"]: context_result["confidence"]})
                    )
                    model_predictions.append(prediction_obj)
            
            if not model_predictions:
                from ensemble import PredictionResult
                return {
                    "predicted_intent": "General Inquiry",
                    "confidence": 0.2,
                    "rationale": "All models failed - using fallback classification",
                    "metadata": {"conversation_metadata": conversation_metadata, "error": "all_models_failed"}
                }
            
            # 3. Advanced ensemble decision making
            ensemble_result = self.ensemble_classifier.combine_predictions(
                model_predictions, conversation_metadata
            )
            
            # 4. Generate comprehensive reasoning
            comprehensive_rationale = self.reasoning_engine.generate_comprehensive_reasoning(
                intent=ensemble_result.intent,
                confidence=ensemble_result.confidence,
                conversation_text=conversation_text,
                conversation_metadata=conversation_metadata,
                model_predictions=model_predictions,
                ensemble_metadata=ensemble_result.metadata
            )
            
            # 5. Return comprehensive result
            return {
                "predicted_intent": ensemble_result.intent,
                "confidence": ensemble_result.confidence,
                "rationale": comprehensive_rationale,
                "metadata": {
                    "conversation_metadata": conversation_metadata,
                    "ensemble_metadata": ensemble_result.metadata,
                    "model_predictions_count": len(model_predictions),
                    "conversation_complexity": self._assess_complexity(conversation_metadata),
                    "confidence_level": ensemble_result.confidence_level.value
                }
            }
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return {
                "predicted_intent": "General Inquiry",
                "confidence": 0.1,
                "rationale": f"Classification failed due to error: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def _assess_complexity(self, conversation_metadata: Dict[str, Any]) -> str:
        """Assess conversation complexity for metadata"""
        total_turns = conversation_metadata.get("total_turns", 1)
        evolution = conversation_metadata.get("intent_evolution", {})
        
        if total_turns >= 8 or evolution.get("escalation_detected", False):
            return "high"
        elif total_turns >= 4 or evolution.get("intent_stability") == "evolving":
            return "medium"
        else:
            return "low"
    
    def batch_classify(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple conversations with progress tracking
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            List of classification results
        """
        results = []
        
        for conv in tqdm(conversations, desc="🧠 Classifying intents with advanced AI"):
            conv_id = conv.get('conversation_id', f"conv_{len(results)+1:03d}")
            messages = conv.get('messages', [])
            
            # Classify intent
            prediction = self.classify_intent(messages)
            
            result = {
                "conversation_id": conv_id,
                "predicted_intent": prediction["predicted_intent"],
                "confidence": prediction["confidence"],
                "rationale": prediction["rationale"],
                "metadata": prediction.get("metadata", {})
            }
            results.append(result)
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the classification system"""
        return {
            "system_name": "Modular Transformer Intent Classifier",
            "version": "2.0",
            "components": {
                "conversation_processor": "Advanced multi-turn analysis",
                "ensemble_classifier": f"Strategy: {self.ensemble_classifier.strategy.value}",
                "reasoning_engine": f"Depth: {self.reasoning_engine.reasoning_depth.value}",
                "models": {
                    name: model.is_loaded for name, model in self.models.items()
                }
            },
            "intent_categories": self.intent_categories,
            "features": [
                "Multi-turn conversation analysis",
                "Intent evolution tracking", 
                "Advanced ensemble methods",
                "Comprehensive reasoning generation",
                "Confidence calibration",
                "Context-aware processing"
            ]
        }

def process_conversations(input_file: str, output_json: str = "results/predictions.json", output_csv: str = "results/predictions.csv"):
    """
    Process conversations from input file and generate predictions with advanced analytics
    """
    logger.info(f"📁 Loading conversations from {input_file}")
    
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
    
    logger.info(f"📊 Found {len(conversations)} conversations to analyze")
    
    # Initialize advanced classifier system
    classifier = TransformerIntentClassifier(
        ensemble_strategy=EnsembleStrategy.ADAPTIVE,
        reasoning_depth=ReasoningDepth.COMPREHENSIVE
    )
    
    # Display system information
    system_info = classifier.get_system_info()
    logger.info(f"🚀 System: {system_info['system_name']} v{system_info['version']}")
    logger.info(f"🧠 Models loaded: {sum(system_info['components']['models'].values())}/3")
    
    # Process conversations with advanced analytics
    results = classifier.batch_classify(conversations)
    
    # Enhanced results with analytics
    enhanced_results = []
    for result in results:
        enhanced_result = {
            "conversation_id": result["conversation_id"],
            "predicted_intent": result["predicted_intent"],
            "confidence": result["confidence"],
            "rationale": result["rationale"]
        }
        
        # Add confidence level for easy filtering
        metadata = result.get("metadata", {})
        confidence_level = metadata.get("confidence_level", "unknown")
        enhanced_result["confidence_level"] = confidence_level
        
        # Add complexity assessment
        complexity = metadata.get("conversation_complexity", "unknown")
        enhanced_result["complexity"] = complexity
        
        enhanced_results.append(enhanced_result)
    
    # Save comprehensive results
    logger.info(f"💾 Saving results to {output_json} and {output_csv}")
    
    # Save detailed JSON with metadata
    detailed_results = {
        "system_info": system_info,
        "analysis_summary": _generate_analysis_summary(results),
        "predictions": enhanced_results
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Save clean CSV for spreadsheet analysis
    df = pd.DataFrame(enhanced_results)
    df.to_csv(output_csv, index=False)
    
    # Print comprehensive summary
    _print_classification_summary(enhanced_results)
    
    return enhanced_results

def _generate_analysis_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary analytics of classification results"""
    
    intents = [r["predicted_intent"] for r in results]
    confidences = [r["confidence"] for r in results]
    
    summary = {
        "total_conversations": len(results),
        "intent_distribution": {},
        "confidence_statistics": {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences)
        },
        "quality_metrics": {
            "high_confidence_rate": len([c for c in confidences if c >= 0.7]) / len(confidences),
            "low_confidence_rate": len([c for c in confidences if c < 0.5]) / len(confidences)
        }
    }
    
    # Calculate intent distribution
    from collections import Counter
    intent_counts = Counter(intents)
    summary["intent_distribution"] = dict(intent_counts)
    
    return summary

def _print_classification_summary(results: List[Dict[str, Any]]):
    """Print comprehensive classification summary"""
    
    logger.info("📈 Classification Summary:")
    
    # Intent distribution
    from collections import Counter
    intents = [r["predicted_intent"] for r in results]
    intent_counts = Counter(intents)
    
    for intent, count in intent_counts.items():
        percentage = (count / len(results)) * 100
        logger.info(f"  {intent}: {count} ({percentage:.1f}%)")
    
    # Confidence analysis
    confidences = [r["confidence"] for r in results]
    avg_confidence = sum(confidences) / len(confidences)
    high_conf_count = len([c for c in confidences if c >= 0.7])
    low_conf_count = len([c for c in confidences if c < 0.5])
    
    logger.info(f"📊 Confidence Analysis:")
    logger.info(f"  Average confidence: {avg_confidence:.3f}")
    logger.info(f"  High confidence (≥0.7): {high_conf_count}/{len(results)} ({high_conf_count/len(results)*100:.1f}%)")
    logger.info(f"  Low confidence (<0.5): {low_conf_count}/{len(results)} ({low_conf_count/len(results)*100:.1f}%)")
    
    # Complexity analysis
    if "complexity" in results[0]:
        complexities = [r["complexity"] for r in results]
        complexity_counts = Counter(complexities)
        logger.info(f"🔍 Complexity Distribution:")
        for complexity, count in complexity_counts.items():
            logger.info(f"  {complexity.title()}: {count}")


def main():
    """Enhanced main entry point with comprehensive options"""
    parser = argparse.ArgumentParser(
        description="🧠 Advanced Transformer-based Multi-Turn Intent Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python intent_classifier_transformer.py sample_conversations.json
  python intent_classifier_transformer.py data.json --output-json results.json --output-csv analysis.csv
  
System Features:
  • Multi-turn conversation analysis with context evolution tracking
  • Advanced ensemble of 3 transformer models (BART, Sentence-T, DialoGPT)
  • Intelligent reasoning engine with comprehensive explanations
  • Adaptive decision making based on conversation characteristics
  • Confidence calibration and quality assessment
        """
    )
    
    parser.add_argument("input_file", 
                       help="Path to input JSON file with conversations")
    parser.add_argument("--output-json", 
                       default="results/predictions.json", 
                       help="Output JSON file path (default: results/predictions.json)")
    parser.add_argument("--output-csv", 
                       default="results/predictions.csv", 
                       help="Output CSV file path (default: results/predictions.csv)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        logger.error(f"❌ Input file not found: {args.input_file}")
        return
    
    logger.info("🚀 Starting Advanced Intent Classification Analysis...")
    
    # Process conversations with advanced system
    try:
        process_conversations(args.input_file, args.output_json, args.output_csv)
        logger.info("✅ Analysis completed successfully!")
        logger.info(f"📄 Detailed results saved to: {args.output_json}")
        logger.info(f"📊 Summary table saved to: {args.output_csv}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
