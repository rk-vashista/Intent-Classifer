"""
Conversation Processing Module - Handles multi-turn conversation analysis
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SenderType(Enum):
    """Enumeration for conversation sender types"""
    CUSTOMER = "customer"
    AGENT = "agent"
    UNKNOWN = "unknown"

@dataclass
class Message:
    """Structured representation of a conversation message"""
    sender: SenderType
    text: str
    timestamp: Optional[str] = None
    message_id: Optional[str] = None
    
    def __post_init__(self):
        """Clean and validate message after initialization"""
        self.text = self._clean_text(self.text)
    
    def _clean_text(self, text: str) -> str:
        """Clean message text"""
        if not text:
            return ""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

@dataclass 
class ConversationTurn:
    """Represents a complete turn in conversation (customer + agent response)"""
    customer_message: Message
    agent_response: Optional[Message] = None
    turn_number: int = 0
    
    def get_turn_context(self) -> str:
        """Get formatted context for this turn"""
        context = f"Turn {self.turn_number}: Customer: {self.customer_message.text}"
        if self.agent_response:
            context += f" | Agent: {self.agent_response.text}"
        return context

class ConversationAnalyzer:
    """Advanced multi-turn conversation analysis"""
    
    def __init__(self, max_turns: int = 10, focus_on_recent: bool = True):
        self.max_turns = max_turns
        self.focus_on_recent = focus_on_recent
        self.intent_evolution_patterns = {
            "escalation": ["first", "then", "but", "however", "still", "actually"],
            "clarification": ["what I mean", "to clarify", "specifically", "in other words"],
            "urgency": ["urgent", "asap", "immediately", "quickly", "soon", "today"],
            "frustration": ["frustrated", "annoyed", "disappointed", "upset", "angry"]
        }
    
    def parse_conversation(self, messages: List[Dict[str, str]]) -> List[Message]:
        """Parse raw messages into structured Message objects"""
        parsed_messages = []
        
        for i, msg in enumerate(messages):
            # Handle different field names
            sender_raw = msg.get('sender', msg.get('role', 'unknown')).lower()
            text = msg.get('text', msg.get('message', ''))
            
            # Determine sender type
            if sender_raw in ['customer', 'user', 'client']:
                sender = SenderType.CUSTOMER
            elif sender_raw in ['agent', 'assistant', 'support', 'rep']:
                sender = SenderType.AGENT
            else:
                sender = SenderType.UNKNOWN
            
            message = Message(
                sender=sender,
                text=text,
                message_id=f"msg_{i:03d}"
            )
            
            if message.text:  # Only add non-empty messages
                parsed_messages.append(message)
        
        return parsed_messages
    
    def extract_conversation_turns(self, messages: List[Message]) -> List[ConversationTurn]:
        """Group messages into conversation turns"""
        turns = []
        current_customer_msg = None
        turn_number = 1
        
        for message in messages:
            if message.sender == SenderType.CUSTOMER:
                # If we have a previous customer message without agent response, create turn
                if current_customer_msg:
                    turns.append(ConversationTurn(
                        customer_message=current_customer_msg,
                        turn_number=turn_number
                    ))
                    turn_number += 1
                current_customer_msg = message
            
            elif message.sender == SenderType.AGENT and current_customer_msg:
                # Complete the turn with agent response
                turns.append(ConversationTurn(
                    customer_message=current_customer_msg,
                    agent_response=message,
                    turn_number=turn_number
                ))
                turn_number += 1
                current_customer_msg = None
        
        # Handle final customer message without agent response
        if current_customer_msg:
            turns.append(ConversationTurn(
                customer_message=current_customer_msg,
                turn_number=turn_number
            ))
        
        return turns
    
    def analyze_intent_evolution(self, turns: List[ConversationTurn]) -> Dict[str, Any]:
        """Analyze how customer intent evolves through conversation"""
        evolution_analysis = {
            "intent_stability": "stable",
            "escalation_detected": False,
            "clarification_requested": False,
            "urgency_level": "normal",
            "frustration_indicators": [],
            "key_turning_points": []
        }
        
        if len(turns) < 2:
            return evolution_analysis
        
        # Analyze each turn for patterns
        customer_texts = [turn.customer_message.text.lower() for turn in turns]
        combined_text = " ".join(customer_texts)
        
        # Check for escalation patterns
        for i, turn in enumerate(turns[1:], 1):
            text = turn.customer_message.text.lower()
            
            # Escalation detection
            if any(pattern in text for pattern in self.intent_evolution_patterns["escalation"]):
                evolution_analysis["escalation_detected"] = True
                evolution_analysis["key_turning_points"].append(f"Turn {i}: Escalation detected")
            
            # Clarification detection
            if any(pattern in text for pattern in self.intent_evolution_patterns["clarification"]):
                evolution_analysis["clarification_requested"] = True
                evolution_analysis["key_turning_points"].append(f"Turn {i}: Clarification requested")
            
            # Urgency detection
            if any(pattern in text for pattern in self.intent_evolution_patterns["urgency"]):
                evolution_analysis["urgency_level"] = "high"
            
            # Frustration detection
            frustration_words = [word for word in self.intent_evolution_patterns["frustration"] if word in text]
            if frustration_words:
                evolution_analysis["frustration_indicators"].extend(frustration_words)
        
        # Determine overall intent stability
        if len(evolution_analysis["key_turning_points"]) > 1:
            evolution_analysis["intent_stability"] = "evolving"
        elif evolution_analysis["escalation_detected"] or evolution_analysis["frustration_indicators"]:
            evolution_analysis["intent_stability"] = "escalating"
        
        return evolution_analysis
    
    def extract_contextual_features(self, turns: List[ConversationTurn]) -> Dict[str, Any]:
        """Extract features that help understand conversation context"""
        if not turns:
            return {}
        
        customer_messages = [turn.customer_message.text for turn in turns]
        agent_responses = [turn.agent_response.text for turn in turns if turn.agent_response]
        
        # Calculate conversation metrics
        features = {
            "total_turns": len(turns),
            "customer_message_count": len(customer_messages),
            "agent_response_count": len(agent_responses),
            "avg_customer_message_length": sum(len(msg.split()) for msg in customer_messages) / len(customer_messages),
            "conversation_flow": "balanced" if len(agent_responses) > len(customer_messages) * 0.7 else "customer_heavy",
            "recent_context_weight": 0.7,  # Weight given to recent messages
            "early_context_weight": 0.3    # Weight given to early messages
        }
        
        # Focus on recent turns if specified
        if self.focus_on_recent and len(turns) > 3:
            recent_turns = turns[-3:]  # Last 3 turns
            features["primary_context"] = " | ".join([
                turn.customer_message.text for turn in recent_turns
            ])
            features["supporting_context"] = " | ".join([
                turn.customer_message.text for turn in turns[:-3]
            ])
        else:
            features["primary_context"] = " | ".join(customer_messages)
            features["supporting_context"] = ""
        
        return features
    
    def generate_conversation_summary(self, turns: List[ConversationTurn]) -> str:
        """Generate a comprehensive conversation summary for classification"""
        if not turns:
            return ""
        
        # Get contextual features
        features = self.extract_contextual_features(turns)
        evolution = self.analyze_intent_evolution(turns)
        
        # Build weighted summary
        summary_parts = []
        
        # Primary context (recent messages) - highest weight
        if features.get("primary_context"):
            summary_parts.append(f"RECENT: {features['primary_context']}")
        
        # Supporting context (earlier messages) - lower weight
        if features.get("supporting_context"):
            summary_parts.append(f"EARLIER: {features['supporting_context']}")
        
        # Add evolution indicators for better classification
        if evolution["escalation_detected"]:
            summary_parts.append("ESCALATION: Customer showing escalation patterns")
        
        if evolution["urgency_level"] == "high":
            summary_parts.append("URGENT: Time-sensitive request detected")
        
        if evolution["frustration_indicators"]:
            summary_parts.append(f"FRUSTRATION: {', '.join(evolution['frustration_indicators'])}")
        
        return " | ".join(summary_parts)

class ConversationPreprocessor:
    """Main preprocessing class that orchestrates conversation analysis"""
    
    def __init__(self, max_turns: int = 10, enable_context_analysis: bool = True):
        self.max_turns = max_turns
        self.enable_context_analysis = enable_context_analysis
        self.analyzer = ConversationAnalyzer(max_turns, focus_on_recent=True)
    
    def preprocess_conversation(self, conversation: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Preprocess a conversation and return formatted text + metadata
        
        Returns:
            Tuple of (formatted_text, conversation_metadata)
        """
        if not conversation:
            return "", {"error": "Empty conversation"}
        
        try:
            # Parse messages
            messages = self.analyzer.parse_conversation(conversation)
            
            if not messages:
                return "", {"error": "No valid messages found"}
            
            # Extract conversation turns
            turns = self.analyzer.extract_conversation_turns(messages)
            
            # Limit turns if necessary
            if len(turns) > self.max_turns:
                turns = turns[-self.max_turns:]  # Keep most recent turns
            
            # Generate comprehensive summary
            formatted_text = self.analyzer.generate_conversation_summary(turns)
            
            # Collect metadata
            metadata = {
                "total_messages": len(messages),
                "total_turns": len(turns),
                "customer_messages": len([m for m in messages if m.sender == SenderType.CUSTOMER]),
                "agent_messages": len([m for m in messages if m.sender == SenderType.AGENT]),
                "contextual_features": self.analyzer.extract_contextual_features(turns) if self.enable_context_analysis else {},
                "intent_evolution": self.analyzer.analyze_intent_evolution(turns) if self.enable_context_analysis else {}
            }
            
            return formatted_text, metadata
            
        except Exception as e:
            logger.error(f"Error preprocessing conversation: {e}")
            # Fallback to simple concatenation
            customer_texts = []
            for msg in conversation:
                sender = msg.get('sender', msg.get('role', '')).lower()
                text = msg.get('text', msg.get('message', ''))
                if sender in ['customer', 'user'] and text:
                    customer_texts.append(text)
            
            return " | ".join(customer_texts), {"error": str(e), "fallback_used": True}
