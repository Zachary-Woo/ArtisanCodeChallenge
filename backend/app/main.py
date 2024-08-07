from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from .LCOpenAICall import basicLangchainOpenAICall
from .LCGetData import get_relevant_data
from .util import load_knowledge_base, find_best_match, get_answer_for_question

app = FastAPI()

API_KEY = os.getenv("ARTISAN_DEMO_API_KEY")
API_KEY_NAME = "ARTISAN_DEMO_API_KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Dependency to get the API key
def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# Pydantic model for request body
class Message(BaseModel):
    user: str
    content: str

# In-memory chat history
chat_history: List[Dict[str, Any]] = []

# Endpoint to handle chat messages with API key validation
@app.post("/chat", dependencies=[Depends(get_api_key)])
async def chat_endpoint(message: Message):
    global chat_history

    # Check if the message content is empty
    if not message.content.strip():
        return {"response": "Please type a question and I'll do my best to help you out!"}

    # Load the FAQ knowledge base & find the best match for the user's input
    faq_knowledge_base = load_knowledge_base("FAQ.json")
    best_match: str | None = find_best_match(message.content, [q["question"] for q in faq_knowledge_base["faq_list"]])

    # Generate response before adding the new message to chat_history
    if best_match:
        answer: str | None = get_answer_for_question(best_match, faq_knowledge_base)
        response = answer
    else:
        # Generate response using GPT when no match is found in FAQ
        response = generate_response(chat_history, message.content)
    
    # Add the user's message to chat history
    chat_history.append({"user": message.user, "content": message.content})
    
    # Add the assistant's response to chat history
    chat_history.append({"user": "assistant", "content": response})

    # Limit chat history to last 10 messages
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    return {"response": response}

# Function to generate a response using OpenAI API
def generate_response(history: List[Dict[str, Any]], user_message: str) -> str:
    # Format the chat history
    formatted_history = ""
    for msg in history:
        role = "User" if msg["user"] == "user" else "Assistant" # technically this is unnecessary as this line could be the cause of my bug
        formatted_history += f"{role}: {msg['content']}\n"

    # Get relevant data
    relevant_data = get_relevant_data(user_message, number_of_chunks=10, chunk_size=500)

    # Call the basicLangchainOpenAICall function
    response = basicLangchainOpenAICall(
        selected_model="gpt-3.5-turbo",
        relevant_data=relevant_data,
        user_message=user_message,
        chat_history=formatted_history
    )
    return response

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"Hello": "World"}