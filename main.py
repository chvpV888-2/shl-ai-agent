from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import json
import os
from dotenv import load_dotenv # <--- ADD THIS LINE

load_dotenv()

app = FastAPI()

# IMPORTANT: We now get the key from the server environment securely!
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set!")

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# Load catalog safely
try:
    with open("catalog.json", "r") as f:
        catalog_data = json.load(f)
    catalog_string = json.dumps(catalog_data)
except FileNotFoundError:
    catalog_string = "[]"

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    system_prompt = f"""
    You are an SHL assessment recommender agent. 
    Your ONLY job is to help users find the right assessment from the SHL catalog.
    
    CATALOG DATA:
    {catalog_string}

    RULES:
    1. If the user is vague (e.g., "I am hiring"), ASK questions to clarify (like role, skills, or seniority).
    2. If the user changes constraints mid-conversation (e.g., "Actually, add personality"), update your recommendations without starting over.
    3. If asked out-of-scope questions (legal advice, coding help, non-SHL tests), REFUSE politely.
    4. If asked to compare tests, use ONLY the catalog descriptions.
    5. Output response EXACTLY as JSON:
       - "reply": Natural language response.
       - "recommendations": Array of objects (name, url, test_type). Empty [] if gathering context or refusing.
       - "end_of_conversation": true ONLY if a final shortlist is provided, else false.
    """

    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        api_messages.append({"role": msg.role, "content": msg.content})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=api_messages,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        ai_output = json.loads(response.choices[0].message.content)
        
        return ChatResponse(
            reply=ai_output.get("reply", "I encountered an error processing your request."),
            recommendations=ai_output.get("recommendations", []),
            end_of_conversation=ai_output.get("end_of_conversation", False)
        )
    except Exception as e:
        print(f"Error: {e}")
        return ChatResponse(
            reply="I'm sorry, our AI service is temporarily unavailable.",
            recommendations=[],
            end_of_conversation=False
        )