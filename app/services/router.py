
import torch
import torch.nn.functional as F

from app.services.embeddings import EmbeddingService


class SemanticRouter:
    def __init__(self):
        self.embedding_service = EmbeddingService()

        self.categories = {
            "coding": [
                "Write code to solve a programming problem.",
                "Debug a Python function.",
                "Implement an algorithm or data structure.",
                "Optimize code for performance.",
            ],
            "reasoning": [
                "Solve a complex logical reasoning problem.",
                "Analyze a mathematical problem step by step.",
                "Evaluate multiple possibilities and make a decision.",
                "Derive a solution using logical deduction.",
            ],
            "general": [
                "Explain a technical concept.",
                "Answer a general knowledge question.",
                "Provide a simple factual explanation.",
                "Explain how a technology works.",
            ],
        }

        self.prototypes = self._build_prototypes()

    def _build_prototypes(self):
        prototypes = {}

        for category, examples in self.categories.items():
            embeddings = self.embedding_service.encode(examples)

            prototype = embeddings.mean(dim=0)

            prototypes[category] = F.normalize(
                prototype.unsqueeze(0),
                p=2,
                dim=1,
            ).squeeze(0)

        return prototypes

   
    def route(self, prompt: str):
        query_embedding = self.embedding_service.encode([prompt])[0]

        scores = {
            category: torch.dot(query_embedding, prototype).item()
            for category, prototype in self.prototypes.items()
        }

        ranked_scores = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        selected_category, top_score = ranked_scores[0]
        second_score = ranked_scores[1][1]

        margin = top_score - second_score

        return {
            "category": selected_category,
            "scores": scores,
            "top_score": top_score,
            "margin": margin,
        }
