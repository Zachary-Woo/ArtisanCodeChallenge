from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from LCOpenAICall import basicLangchainOpenAICall
from LCGetData import get_relevant_data
import util

app = FastAPI()

# Pydantic model for request body
class Message(BaseModel):
    user: str
    content: str

# In-memory chat history
chat_history: List[Dict[str, Any]] = []

# Endpoint to handle chat messages
@app.post("/chat")
async def chat_endpoint(message: Message):
    global chat_history

    # Load the FAQ knowledge base & find the best match for the user's input
    faq_knowledge_base = util.load_knowledge_base("FAQ.json")
    best_match: str | None = util.find_best_match(message.content, [q["question"] for q in faq_knowledge_base["faq_list"]])

    # Generate response before adding the new message to chat_history
    if best_match:
        answer: str | None = util.get_answer_for_question(best_match, faq_knowledge_base)
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
        role = "User" if msg["user"] == "user" else "Assistant"
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