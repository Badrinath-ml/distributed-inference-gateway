
import torch

from app.services.embeddings import EmbeddingService


def main():
    embedding_service = EmbeddingService()

    prompts = [
    "Write Python code to implement a binary search algorithm.",
    "Debug this Python function and explain the error.",
    "Implement a graph traversal algorithm using BFS.",
    "Solve a complex logical reasoning puzzle step by step.",
    "Analyze this mathematical problem and derive its solution.",
    "Evaluate multiple possibilities and make a logical decision.",
    "Explain what an API is in simple terms.",
    "What is the difference between RAM and ROM?",
    "Give me a brief explanation of cloud computing.",
    ]

    embeddings = embedding_service.encode(prompts)

    print("\nEmbedding shape:", embeddings.shape)
    print("Device:", embeddings.device)

    print("\nCosine similarity comparisons:")

    similarities = embeddings @ embeddings.T

    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            print(
                f"\nPrompt {i + 1} vs Prompt {j + 1}: "
                f"{similarities[i, j].item():.4f}"
            )


if __name__ == "__main__":
    main()
