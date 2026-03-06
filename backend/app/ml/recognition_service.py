"""
Recognition Service
--------------------
Matches detected face embeddings against the stored reference embeddings.
Handles threshold filtering, scoring, and result packaging.
"""

import numpy as np
from typing import Optional


class RecognitionService:
    """
    Performs cosine similarity matching between a query embedding
    and all stored reference embeddings.
    """

    def __init__(self, threshold: float = 0.45):
        """
        Args:
            threshold: Minimum cosine similarity to accept a match.
                       buffalo_l recommended range: 0.35 – 0.55
                       Higher = stricter (fewer false positives)
                       Lower  = looser (fewer misses, more false positives)
        """
        self.threshold = threshold

    def match(
        self,
        face_embedding: np.ndarray,
        reference_embeddings: dict,   # {name: np.ndarray}
        top_k: int = 1,
    ) -> dict:
        """
        Match a single face embedding against all references.

        Returns:
        {
            "name": str,              # best match name or "Unknown"
            "score": float,           # cosine similarity of best match
            "matched": bool,          # True if score >= threshold
            "top_k": [                # top_k candidates (always returned)
                {"name": str, "score": float},
                ...
            ]
        }
        """
        if not reference_embeddings:
            return self._unknown(0.0, [])

        # L2-normalize query (defensive — engine already does this)
        q = face_embedding / (np.linalg.norm(face_embedding) + 1e-10)

        scores = {}
        for name, ref_emb in reference_embeddings.items():
            ref = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)
            scores[name] = float(np.dot(q, ref))

        # Sort descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_k_results = [{"name": n, "score": s} for n, s in ranked[:top_k]]

        best_name, best_score = ranked[0]

        # "it should return matches from stored embeddings only"
        # We NO LONGER return "Unknown" here. We always return the best candidate
        # but mark 'matched' as False if it's below threshold for UI warning.
        return {
            "name": best_name,
            "score": best_score,
            "matched": (best_score >= self.threshold),
            "top_k": top_k_results,
        }

    def _unknown(self, score: float, top_k: list) -> dict:
        return {
            "name": "Unknown",
            "score": score,
            "matched": False,
            "top_k": top_k,
        }

    def match_batch(
        self,
        face_embeddings: list[np.ndarray],
        reference_embeddings: dict,
        top_k: int = 1,
    ) -> list[dict]:
        """Match multiple face embeddings at once (e.g., all faces in one image)."""
        return [
            self.match(emb, reference_embeddings, top_k=top_k)
            for emb in face_embeddings
        ]