import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

_CONFIDENCE_PROMPT = """Dựa trên context được cung cấp, câu trả lời này có độ tin cậy bao nhiêu?
Context: {context}
Câu hỏi: {query}
Câu trả lời: {answer}
Chỉ trả về một số thập phân từ 0.0 đến 1.0, không giải thích thêm."""


class ConfidenceScorer:
    """Scores the confidence of a generated answer using LLM self-evaluation."""

    def __init__(self, provider) -> None:
        self.provider = provider

    async def score(self, query: str, context: str, answer: str) -> Optional[float]:
        """
        Ask the LLM to self-evaluate the answer confidence.
        Returns a float in [0.0, 1.0] or None if scoring fails.
        """
        try:
            prompt = _CONFIDENCE_PROMPT.format(
                context=context[:2000],
                query=query[:500],
                answer=answer[:500],
            )
            response = await self.provider.ainvoke(prompt)
            text = response.content.strip() if hasattr(response, "content") else str(response).strip()
            # Match decimal like 0.85 or 1.0 or 0 or 1
            match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0*)?)\b", text)
            if match:
                score = float(match.group(1))
                return round(max(0.0, min(1.0, score)), 4)
        except Exception as e:
            log.warning(f"Confidence scoring failed: {e}")
        return None
