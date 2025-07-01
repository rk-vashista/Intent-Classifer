🚪 Machine Test: Multi-Turn Intent Classification from WhatsApp-Style Conversations

📉 Objective
Build a system that takes a full WhatsApp-style multi-turn conversation between a user and a business, and classifies the final intent of the customer.

📁 Input
A JSON file containing multiple chat logs. Each conversation has a sequence of messages like:
{
  "conversation_id": "conv_001",
  "messages": [
    {"sender": "user", "text": "Hi, I'm looking for a 2BHK in Dubai"},
    {"sender": "agent", "text": "Great! Any specific area in mind?"},
    {"sender": "user", "text": "Preferably Marina or JVC"},
    {"sender": "agent", "text": "What's your budget?"},
    {"sender": "user", "text": "Max 120k. Can we do a site visit this week?"}
  ]
}

🎯 Output
A JSON or CSV file with intent classification for each conversation:
JSON Format:
{
  "conversation_id": "conv_001",
  "predicted_intent": "Book Appointment",
  "rationale": "The user requested a site visit after discussing budget and location."
}
CSV Format:
conversation_id,predicted_intent,rationale
conv_001,Book Appointment,"The user requested a site visit after discussing budget and location."

🍿 Intent Labels
The system must classify into one of these 5 final intents:
“Book Appointment”
“Product Inquiry”
“Pricing Negotiation”
“Support Request”
“Follow-Up”

🧠 What You Must Build
Preprocessor
Clean messages (e.g., lowercasing, emoji removal)
Optionally truncate old messages (last N messages)
Intent Classifier
Use prompt-based approach with Gemma or DistilBERT
Predictor
Reads input JSON
Predicts intent and rationale
Writes output in both JSON and CSV formats
Supports batch processing of thousands of conversations efficiently

⚙️ Requirements
Use only open-source models (e.g., Gemma via Hugging Face, DistilBERT, RoBERTa)
Language: Python
Frameworks: transformers, scikit-learn, pandas, etc.
No use of commercial APIs (OpenAI, Claude, etc.)

🗒️ Deliverables
Python script or notebook that:
Reads the input JSON
Preprocesses and classifies the intent
Writes predictions to both output JSON and CSV
Efficiently handles thousands of conversations
README.md that includes:
Setup and run instructions
Explanation of model choice
Sample predictions
Any limitations or edge cases
requirements.txt for all dependencies

📊 Evaluation Criteria
Category
Weight
Accuracy of predictions
30%
Understanding of multi-turn context
25%
Code clarity and modularity
20%
Creativity in model use
15%
Reasoning and rationale generation
10%


❓ Queries
Reach out to +91 95443 17670 - Amal Shyjo - Chief Product Officer @ Zeldin.ai
