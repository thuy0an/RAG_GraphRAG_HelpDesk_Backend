import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_RERANK_PROMPT = """Trên thang điểm 0-10, đoạn văn sau có liên quan đến câu hỏi không?
Câu hỏi: {query}
Đoạn văn: {passage}
Chỉ trả về một số nguyên từ 0 đến 10, không giải thích thêm."""


class LLMReranker:
    """Re-ranks retrieved passages using LLM scoring."""

    def __init__(self, provider) -> None:
        self.provider = provider

    async def _score_one(self, query: str, passage: str) -> float:
        """Score a single passage against the query. Returns 0.0 on error."""
        try:
            prompt = _RERANK_PROMPT.format(query=query, passage=passage[:1000])
            response = await self.provider.ainvoke(prompt)
            text = response.content.strip() if hasattr(response, "content") else str(response).strip()
            match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(10.0, score))
        except Exception as e:
            log.warning(f"Re-ranking score failed for passage: {e}")
        return 0.0

    async def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], List[float], float]:
        """
        Re-rank docs by LLM relevance score.

        Returns:
            (reranked_docs, scores, elapsed_seconds)
            - reranked_docs: top_k docs sorted by score descending
            - scores: corresponding scores for reranked_docs
            - elapsed_seconds: time taken for re-ranking
        """
        if not docs:
            return [], [], 0.0

        start = time.perf_counter()
        passages = [doc.get("content", "") for doc in docs]

        # Score all passages in parallel
        scores = list(await asyncio.gather(
            *[self._score_one(query, p) for p in passages],
            return_exceptions=False,
        ))

        elapsed = round(time.perf_counter() - start, 3)

        # Sort by score descending, keep top_k
        scored_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_pairs = scored_pairs[:top_k]

        reranked_docs = [d for d, _ in top_pairs]
        final_scores = [s for _, s in top_pairs]

        log.info(f"Re-ranking completed: {len(docs)} → {len(reranked_docs)} docs in {elapsed}s")
        return reranked_docs, final_scores, elapsed
