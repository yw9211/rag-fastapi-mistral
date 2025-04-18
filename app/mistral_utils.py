import os
from mistralai import Mistral

# Initialize the Mistral client with your API key
client = Mistral(api_key="gwGmq0uYH2PWtj3ZnuNpDXeCgtqaXnOf")

# System prompt guiding the LLM to classify query intent
INTENT_PROMPT = (
    "You are an intent classifier. Given a user message, determine if it is an "
    "information-seeking query that should trigger a document search in a knowledge base. "
    "Only respond with 'YES' or 'NO'."
)

def is_search_query_llm(user_input: str) -> bool:
    """
    Uses Mistral LLM to determine whether the user input is an 
    information-seeking query that should trigger a knowledge base search.

    Returns True if the LLM responds with 'YES', otherwise False.
    """
    messages = [
        {"role": "system", "content": INTENT_PROMPT},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.complete(model="mistral-small-latest", messages=messages)
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")
