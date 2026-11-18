
from app.services.router import SemanticRouter


def main():
    router = SemanticRouter()

    prompts = [
    "Implement a binary search tree in Python.",
    "Solve this complex mathematical proof.",
    "Explain how HTTP works.",
    "Debug this Python code.",
    "Analyze the logical implications of this statement.",
    "What is cloud computing?",

    # Ambiguous test cases
    "Explain the time complexity of Dijkstra's algorithm.",
    "Write a Python program to calculate statistical probability.",
    "Explain how neural networks make predictions.",
    "Design an algorithm to solve a scheduling problem.",
]

    for prompt in prompts:
        result = router.route(prompt)

        print(f"Category: {result['category']}")
        print(f"Top score: {result['top_score']:.4f}")
        print(f"Confidence margin: {result['margin']:.4f}")
        print(f"Scores: {result['scores']}")


if __name__ == "__main__":
    main()
