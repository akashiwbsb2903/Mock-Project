"""
LangChain agent for source selection: decides whether to answer from Reddit or Journals based on user query intent.
"""
from typing import List

def select_source_agent(user_query: str) -> str:
    """
    Simple agentic logic: if the query is about research, prefer journals; if about opinions, trends, or community, prefer Reddit.
    Extend this logic for more nuanced decisions or use an LLM for classification.
    """
    q = user_query.lower()
    # Keywords for research/journals
    research_keywords = ["research", "study", "studies", "evidence", "journal", "clinical", "trial", "meta-analysis", "systematic review"]
    # Keywords for Reddit/community
    reddit_keywords = ["experience", "reddit", "community", "forum", "discussion", "opinions", "personal", "users", "thread", "post","trend","tredns","trending"]
    if any(word in q for word in research_keywords):
        return "journals"
    # If Reddit/community keywords are present, use both reddit and blogs
    if any(word in q for word in reddit_keywords):
        return "reddit_blogs"
    # Default: let both sources be considered
    return "all"

# Example usage in chatbot:
# source = select_source_agent(user_query)
# if source == "journals":
#     ... restrict context to journals ...
# elif source == "reddit_blogs":
#     ... restrict context to reddit and blogs ...
# else:
#     ... use all sources ...

