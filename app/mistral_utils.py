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
    Determines whether a given user input is an information-seeking query
    that should trigger a knowledge base search, using Mistral LLM.

    Args:
        user_input (str): The raw user query from the user.

    Returns:
        bool: True if the LLM determines a search is needed, False otherwise.
    """
    messages = [
        {"role": "system", "content": INTENT_PROMPT},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.complete(model="mistral-small-latest", messages=messages)
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")

# System prompt guiding the LLM to improve query for RAG
REWRITE_PROMPT = (
    "You are a retrieval assistant. Given a user input, rephrase it into a clear, standalone question "
    "optimized for semantic search over a business knowledge base. Do not add fluff."
)

def transform_query(query: str) -> str:
    """
    Rewrites a user query into a clear, standalone information-seeking question
    using a language model (Mistral) to optimize it for semantic retrieval
    in a knowledge base context.

    Args:
        query (str): The raw user input.

    Returns:
        str: A reformulated query, designed to improve retrieval performance.
    """
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": query}
    ]
    response = client.chat.complete(model="mistral-small-latest", messages=messages)
    return response.choices[0].message.content.strip()