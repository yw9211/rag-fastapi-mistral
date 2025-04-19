import os
from mistralai import Mistral

# Initialize the Mistral client with your API key
api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-latest"
client = Mistral(api_key=api_key)

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
        {"role": "user", "content": user_input},
    ]

    response = client.chat.complete(model=model, messages=messages)
    answer = response.choices[0].message.content.strip().lower()
    return answer.startswith("yes")

# System prompt guiding the LLM to improve query for RAG
REWRITE_PROMPT = (
    "You are a text cleaner. Your job is to rewrite user queries to improve grammar, clarity, and spelling "
    "while preserving the original meaning. Do not change the intent. "
    "Respond with the cleaned query only — no explanation, no formatting, no prefix."
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
    response = client.chat.complete(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def generate_response(query: str, context: str = "") -> str:
    """
    Generate a response from the Mistral model based on the given query and optional context.

    Args:
        query (str): The user's question.
        context (str): Contextual information from the knowledge base (optional).

    Returns:
        str: The model-generated answer.
    """
    if context:
        prompt = (
            f"You are a helpful assistant. Use the following context to answer the question. Be brief and polite. \n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    else:
        prompt = query

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.complete(model=model, messages=messages)
    return response.choices[0].message.content