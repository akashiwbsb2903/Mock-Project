from nutrition_insights.phase3.services.query_router import is_in_scope

if __name__ == "__main__":
    questions = [
        "What is the best time to take whey protein?",
        "Tell me about protein shakes.",
        "Is creatine good for muscle gain?",
        "What is the capital of France?",
        "How much protein is in an egg?",
        "hi"
    ]
    for q in questions:
        print(f"is_in_scope('{q}') = {is_in_scope(q)}")
