"""
Intelligent Reasoning Engine - Advanced rationale generation and explanation
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class ReasoningDepth(Enum):
    """Levels of reasoning detail"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    EXPERT = "expert"

@dataclass
class ReasoningComponent:
    """Individual component of reasoning"""
    category: str
    evidence: str
    confidence_impact: float
    weight: float = 1.0

class IntelligentReasoningEngine:
    """Advanced reasoning and explanation generation system"""
    
    def __init__(self, reasoning_depth: ReasoningDepth = ReasoningDepth.COMPREHENSIVE):
        self.reasoning_depth = reasoning_depth
        
        # Intent-specific reasoning templates
        self.intent_reasoning_templates = {
            "Book Appointment": {
                "primary_indicators": [
                    "scheduling language", "time references", "availability inquiries",
                    "meeting requests", "visit planning", "appointment booking"
                ],
                "supporting_evidence": [
                    "calendar references", "location mentions", "timing constraints",
                    "availability questions", "booking confirmations"
                ],
                "context_clues": [
                    "when can", "available", "schedule", "meet", "visit", "appointment",
                    "book", "arrange", "time", "date", "calendar"
                ]
            },
            "Support Request": {
                "primary_indicators": [
                    "help requests", "problem descriptions", "technical issues",
                    "malfunction reports", "assistance needs", "troubleshooting"
                ],
                "supporting_evidence": [
                    "error descriptions", "functional problems", "assistance keywords",
                    "troubleshooting language", "support terminology"
                ],
                "context_clues": [
                    "help", "problem", "issue", "broken", "not working", "error",
                    "trouble", "fix", "support", "assist", "malfunction"
                ]
            },
            "Pricing Negotiation": {
                "primary_indicators": [
                    "cost discussions", "budget constraints", "price negotiations",
                    "financial planning", "payment terms", "discount requests"
                ],
                "supporting_evidence": [
                    "monetary values", "budget limitations", "cost comparisons",
                    "payment options", "financial concerns"
                ],
                "context_clues": [
                    "price", "cost", "budget", "expensive", "cheap", "discount",
                    "deal", "negotiate", "quote", "payment", "afford", "money"
                ]
            },
            "General Inquiry": {
                "primary_indicators": [
                    "information requests", "general questions", "basic inquiries",
                    "exploratory discussions", "fact-finding", "general interest"
                ],
                "supporting_evidence": [
                    "question patterns", "information seeking", "general curiosity",
                    "broad inquiries", "exploratory language"
                ],
                "context_clues": [
                    "information", "tell me", "explain", "how", "what", "where",
                    "general", "question", "inquire", "details", "about"
                ]
            },
            "Product Information": {
                "primary_indicators": [
                    "feature inquiries", "specification requests", "product details",
                    "technical information", "capability questions", "option exploration"
                ],
                "supporting_evidence": [
                    "technical terminology", "feature comparisons", "specification language",
                    "product-specific questions", "capability assessments"
                ],
                "context_clues": [
                    "features", "specifications", "specs", "details", "model",
                    "options", "available", "product", "version", "type", "capability"
                ]
            }
        }
        
        # Confidence impact factors
        self.confidence_factors = {
            "high_keyword_match": 0.3,
            "context_consistency": 0.25,
            "model_agreement": 0.2,
            "conversation_flow": 0.15,
            "semantic_similarity": 0.1
        }
        
        # Reasoning quality metrics
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "poor": 0.3
        }
    
    def generate_comprehensive_reasoning(self, 
                                       intent: str,
                                       confidence: float,
                                       conversation_text: str,
                                       conversation_metadata: Dict[str, Any],
                                       model_predictions: List[Any],
                                       ensemble_metadata: Dict[str, Any]) -> str:
        """Generate comprehensive reasoning explanation"""
        
        reasoning_components = []
        
        # 1. Intent Classification Reasoning
        intent_reasoning = self._analyze_intent_evidence(intent, conversation_text)
        reasoning_components.append(intent_reasoning)
        
        # 2. Confidence Analysis
        confidence_reasoning = self._analyze_confidence_factors(confidence, model_predictions, ensemble_metadata)
        reasoning_components.append(confidence_reasoning)
        
        # 3. Conversation Context Analysis
        if conversation_metadata:
            context_reasoning = self._analyze_conversation_context(conversation_metadata)
            reasoning_components.append(context_reasoning)
        
        # 4. Model Agreement Analysis
        if len(model_predictions) > 1:
            agreement_reasoning = self._analyze_model_agreement(model_predictions, ensemble_metadata)
            reasoning_components.append(agreement_reasoning)
        
        # 5. Multi-turn Evolution Analysis
        if conversation_metadata.get("total_turns", 1) > 1:
            evolution_reasoning = self._analyze_intent_evolution(conversation_metadata)
            reasoning_components.append(evolution_reasoning)
        
        # Combine reasoning based on depth setting
        return self._format_reasoning(reasoning_components, intent, confidence)
    
    def _analyze_intent_evidence(self, intent: str, conversation_text: str) -> ReasoningComponent:
        """Analyze evidence supporting the predicted intent"""
        
        if intent not in self.intent_reasoning_templates:
            return ReasoningComponent(
                category="Intent Evidence",
                evidence=f"Classified as {intent} based on general pattern matching",
                confidence_impact=0.3
            )
        
        template = self.intent_reasoning_templates[intent]
        text_lower = conversation_text.lower()
        
        # Find matching keywords
        matched_keywords = []
        for keyword in template["context_clues"]:
            if keyword in text_lower:
                matched_keywords.append(keyword)
        
        # Analyze evidence strength
        evidence_strength = len(matched_keywords) / len(template["context_clues"])
        
        # Generate evidence description
        if evidence_strength > 0.4:
            evidence_level = "Strong"
            impact = 0.8
        elif evidence_strength > 0.2:
            evidence_level = "Moderate"
            impact = 0.6
        else:
            evidence_level = "Weak"
            impact = 0.3
        
        evidence_text = f"{evidence_level} evidence for {intent}"
        if matched_keywords:
            evidence_text += f" - detected indicators: {', '.join(matched_keywords[:5])}"
        
        # Add primary indicator analysis
        primary_matches = 0
        for indicator in template["primary_indicators"]:
            if any(word in text_lower for word in indicator.split()):
                primary_matches += 1
        
        if primary_matches > 0:
            evidence_text += f". Found {primary_matches} primary intent indicators"
            impact += 0.1
        
        return ReasoningComponent(
            category="Intent Evidence",
            evidence=evidence_text,
            confidence_impact=min(impact, 1.0)
        )
    
    def _analyze_confidence_factors(self, confidence: float, model_predictions: List[Any], 
                                  ensemble_metadata: Dict[str, Any]) -> ReasoningComponent:
        """Analyze factors contributing to confidence level"""
        
        confidence_level = self._get_confidence_category(confidence)
        
        evidence_parts = [f"Overall confidence: {confidence:.2f} ({confidence_level})"]
        
        # Model agreement factor
        agreement = ensemble_metadata.get("agreement_level", 0.5)
        if agreement >= 0.8:
            evidence_parts.append("High model agreement supports confidence")
        elif agreement >= 0.6:
            evidence_parts.append("Moderate model agreement")
        else:
            evidence_parts.append("Low model agreement reduces confidence")
        
        # Ensemble strategy factor
        strategy = ensemble_metadata.get("ensemble_strategy", "unknown")
        strategy_confidence = {
            "confidence_based": "Selected highest confidence prediction",
            "weighted_average": "Combined multiple model scores",
            "majority_vote": "Based on model consensus",
            "hierarchical": "Applied reliability-based selection"
        }
        
        if strategy in strategy_confidence:
            evidence_parts.append(strategy_confidence[strategy])
        
        # Number of models factor
        num_models = ensemble_metadata.get("num_models", 1)
        if num_models > 2:
            evidence_parts.append(f"Decision supported by {num_models} independent models")
        
        return ReasoningComponent(
            category="Confidence Analysis",
            evidence=". ".join(evidence_parts),
            confidence_impact=confidence
        )
    
    def _analyze_conversation_context(self, conversation_metadata: Dict[str, Any]) -> ReasoningComponent:
        """Analyze conversation context factors"""
        
        evidence_parts = []
        impact = 0.5
        
        # Turn analysis
        total_turns = conversation_metadata.get("total_turns", 1)
        if total_turns == 1:
            evidence_parts.append("Single-turn conversation provides direct intent signal")
            impact += 0.1
        elif total_turns <= 3:
            evidence_parts.append(f"Short conversation ({total_turns} turns) with clear intent progression")
            impact += 0.05
        else:
            evidence_parts.append(f"Extended conversation ({total_turns} turns) provides rich context")
            impact += 0.15
        
        # Message balance
        customer_msgs = conversation_metadata.get("customer_messages", 1)
        agent_msgs = conversation_metadata.get("agent_messages", 0)
        
        if agent_msgs > 0:
            balance = agent_msgs / customer_msgs
            if balance > 0.7:
                evidence_parts.append("Balanced dialogue enhances context understanding")
                impact += 0.1
            else:
                evidence_parts.append("Customer-heavy conversation focuses intent clearly")
        
        # Features analysis
        features = conversation_metadata.get("contextual_features", {})
        if features.get("conversation_flow") == "balanced":
            evidence_parts.append("Balanced conversation flow supports accurate classification")
        
        return ReasoningComponent(
            category="Conversation Context",
            evidence=". ".join(evidence_parts) if evidence_parts else "Standard conversation context",
            confidence_impact=min(impact, 1.0)
        )
    
    def _analyze_model_agreement(self, model_predictions: List[Any], 
                               ensemble_metadata: Dict[str, Any]) -> ReasoningComponent:
        """Analyze agreement between different models"""
        
        agreement = ensemble_metadata.get("agreement_level", 0.5)
        num_models = ensemble_metadata.get("num_models", len(model_predictions))
        
        if agreement >= 0.9:
            agreement_desc = "Unanimous"
            impact = 0.9
        elif agreement >= 0.8:
            agreement_desc = "Strong"
            impact = 0.8
        elif agreement >= 0.6:
            agreement_desc = "Moderate" 
            impact = 0.6
        else:
            agreement_desc = "Weak"
            impact = 0.4
        
        evidence = f"{agreement_desc} agreement ({agreement:.1%}) across {num_models} models"
        
        # Add model diversity analysis
        if hasattr(model_predictions[0], 'method'):
            methods = list(set([p.method.split('(')[0] for p in model_predictions if hasattr(p, 'method')]))
            if len(methods) > 2:
                evidence += f". Diverse model types: {', '.join(methods)}"
                impact += 0.1
        
        return ReasoningComponent(
            category="Model Agreement",
            evidence=evidence,
            confidence_impact=impact
        )
    
    def _analyze_intent_evolution(self, conversation_metadata: Dict[str, Any]) -> ReasoningComponent:
        """Analyze how intent evolved through conversation"""
        
        evolution = conversation_metadata.get("intent_evolution", {})
        
        evidence_parts = []
        impact = 0.5
        
        stability = evolution.get("intent_stability", "stable")
        if stability == "stable":
            evidence_parts.append("Consistent intent throughout conversation")
            impact += 0.2
        elif stability == "evolving":
            evidence_parts.append("Intent evolved but remained focused")
            impact += 0.1
        elif stability == "escalating":
            evidence_parts.append("Intent escalated, requiring careful interpretation")
        
        # Escalation analysis
        if evolution.get("escalation_detected", False):
            evidence_parts.append("Escalation patterns detected")
            impact -= 0.1
        
        # Urgency analysis
        urgency = evolution.get("urgency_level", "normal")
        if urgency == "high":
            evidence_parts.append("High urgency indicators found")
            impact += 0.1
        
        # Frustration analysis
        frustration = evolution.get("frustration_indicators", [])
        if frustration:
            evidence_parts.append(f"Frustration indicators: {', '.join(frustration[:3])}")
        
        return ReasoningComponent(
            category="Intent Evolution",
            evidence=". ".join(evidence_parts) if evidence_parts else "Standard intent progression",
            confidence_impact=max(impact, 0.2)
        )
    
    def _get_confidence_category(self, confidence: float) -> str:
        """Get human-readable confidence category"""
        if confidence >= 0.9:
            return "very high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very low"
    
    def _format_reasoning(self, components: List[ReasoningComponent], 
                         intent: str, confidence: float) -> str:
        """Format reasoning components into final explanation"""
        
        if self.reasoning_depth == ReasoningDepth.BASIC:
            return self._format_basic_reasoning(intent, confidence)
        elif self.reasoning_depth == ReasoningDepth.DETAILED:
            return self._format_detailed_reasoning(components, intent, confidence)
        elif self.reasoning_depth == ReasoningDepth.COMPREHENSIVE:
            return self._format_comprehensive_reasoning(components, intent, confidence)
        else:  # EXPERT
            return self._format_expert_reasoning(components, intent, confidence)
    
    def _format_basic_reasoning(self, intent: str, confidence: float) -> str:
        """Generate basic reasoning explanation"""
        confidence_desc = self._get_confidence_category(confidence)
        return f"Classified as {intent} with {confidence_desc} confidence ({confidence:.2f})"
    
    def _format_detailed_reasoning(self, components: List[ReasoningComponent], 
                                 intent: str, confidence: float) -> str:
        """Generate detailed reasoning explanation"""
        parts = [f"Intent: {intent}"]
        
        # Add top 2 most impactful reasoning components
        sorted_components = sorted(components, key=lambda x: x.confidence_impact, reverse=True)
        for component in sorted_components[:2]:
            parts.append(component.evidence)
        
        confidence_desc = self._get_confidence_category(confidence)
        parts.append(f"Confidence: {confidence_desc} ({confidence:.2f})")
        
        return ". ".join(parts)
    
    def _format_comprehensive_reasoning(self, components: List[ReasoningComponent], 
                                      intent: str, confidence: float) -> str:
        """Generate comprehensive reasoning explanation"""
        reasoning_parts = []
        
        # Intent classification explanation
        reasoning_parts.append(f"Classified as '{intent}' based on comprehensive analysis")
        
        # Add all reasoning components
        for component in components:
            reasoning_parts.append(f"{component.category}: {component.evidence}")
        
        # Overall assessment
        confidence_desc = self._get_confidence_category(confidence)
        reasoning_parts.append(f"Final assessment: {confidence_desc} confidence ({confidence:.2f})")
        
        return ". ".join(reasoning_parts)
    
    def _format_expert_reasoning(self, components: List[ReasoningComponent], 
                               intent: str, confidence: float) -> str:
        """Generate expert-level reasoning explanation"""
        # Start with classification decision
        reasoning = f"Expert Analysis - Intent Classification: {intent}\n\n"
        
        # Detailed component analysis
        for i, component in enumerate(components, 1):
            impact_desc = "High" if component.confidence_impact > 0.7 else "Medium" if component.confidence_impact > 0.4 else "Low"
            reasoning += f"{i}. {component.category} (Impact: {impact_desc}): {component.evidence}\n"
        
        # Confidence breakdown
        reasoning += f"\nConfidence Analysis: {confidence:.3f}\n"
        reasoning += f"- Classification Quality: {self._assess_reasoning_quality(components)}\n"
        reasoning += f"- Evidence Strength: {self._calculate_evidence_strength(components):.2f}\n"
        reasoning += f"- Decision Reliability: {self._get_confidence_category(confidence)}\n"
        
        return reasoning
    
    def _assess_reasoning_quality(self, components: List[ReasoningComponent]) -> str:
        """Assess overall quality of reasoning"""
        avg_impact = sum(c.confidence_impact for c in components) / len(components) if components else 0
        
        for quality, threshold in self.quality_thresholds.items():
            if avg_impact >= threshold:
                return quality
        return "poor"
    
    def _calculate_evidence_strength(self, components: List[ReasoningComponent]) -> float:
        """Calculate overall evidence strength"""
        if not components:
            return 0.0
        
        weighted_sum = sum(c.confidence_impact * c.weight for c in components)
        total_weight = sum(c.weight for c in components)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
