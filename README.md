# Voice-Agentic-AI-Assistant
Agentic AI Assistant
Build an intelligent Voice Agentic AI Assistant using Python that can:
1. Accept voice input from users
2. Convert voice to text
3. Process the text using RegEx, pattern matching, and GenAI capabilities
4. Generate intelligent responses using Gemini AI or any open-source LLM
5. Convert AI-generated text response back to voice
6. Store all conversations/responses in CSV files for tracking and analytics
7. Functional Requirements:
Here are high level functional requirements expected from the candidates

1. Voice-to-Text
a. b. Capture voice input through microphone/audio file
Convert speech into text using Python libraries/APIs

2. Text Processing
a. Use extensive RegEx and pattern matching
b. Identify:
i. Greetings
ii. Dates
iii. Email IDs
iv. Phone numbers
v. Keywords/intents
vi. Commands
c. Use Python collections, sorting, filtering, searching, etc.

3. GenAI / LLM Integration
a. b. Integrate: with Google Gemini AI OR Open-source LLM (Mistral, Llama, Ollama, etc.)
Generate meaningful responses based on user queries
4. Text-to-Voice: Convert generated text response back to speech

5. File Storage
a. Store the following either in XLSX/CSV:
i. User voice transcript
ii. AI response
iii. Timestamp
iv. Intent/category
b. Use CSV file-based persistence

6. Basic Agentic Behavior
The system should:
a. Understand context
c. Perform conditional actions
d. Route requests based on detected intent/patterns
Example:
a. “Show today’s tasks”
b. “Save this note”
c. “Summarize this message”
