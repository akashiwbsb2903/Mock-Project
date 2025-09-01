from nutrition_insights.phase3.services.query_router import is_in_scope

if __name__ == "__main__":
    question = "What is the best time to take whey protein?"
    print(f"is_in_scope('{question}') = {is_in_scope(question)}")
