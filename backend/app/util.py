import json
import os
from difflib import get_close_matches

# Load the knowledge base from the JSON file
def load_knowledge_base(file_name: str) -> dict:
    # Get the directory of the current file (util.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to FAQ.json
    file_path = os.path.join(current_dir, file_name)

    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data

# Find the best match for the user's input
def find_best_match(user_question: str, queries: list[str]) -> str | None:
    matches: list = get_close_matches(user_question, queries, n=1, cutoff=0.9) # n is the number of matches to return. Cutoff is the minimum similarity ratio for a response (0.8 is 80%)
    return matches[0] if matches else None

# Get the answer to the user's question
def get_answer_for_question(question: str, knowledge_base: dict) -> str | None:
    for q in knowledge_base["faq_list"]:
        if q["question"] == question:
            return q["answer"]