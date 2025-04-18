# TODO: Transform the query to improve retrieval
def transform_query(user_query: str) -> str:
    # For now we just lowercase and strip — later can rephrase with LLM
    return user_query.lower().strip()
