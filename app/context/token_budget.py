import logging
from typing import List, Dict
logger = logging.getLogger(__name__)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed - using approximate token counting")
class TokenBudgetManager:
    def __init__(self, max_tokens: int = 6000, model: str = "gpt-3.5-turbo"):
        self.max_tokens = max_tokens
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoder = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoder = None
    def count_tokens(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text) // 4
    def fit_to_budget(self, chunks: List[Dict], max_tokens: int = None) -> List[Dict]:
        budget = max_tokens or self.max_tokens
        result = []
        tokens_used = 0
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_tokens = self.count_tokens(text)
            if tokens_used + chunk_tokens <= budget:
                result.append(chunk)
                tokens_used += chunk_tokens
            else:
                remaining = budget - tokens_used
                if remaining > 50:
                    if self.encoder:
                        tokens = self.encoder.encode(text)
                        truncated = self.encoder.decode(tokens[:remaining])
                    else:
                        truncated = text[:remaining * 4]
                    result.append({**chunk, "text": truncated + "..."})
                    tokens_used += remaining
                break
        logger.info(f"Token budget: {tokens_used}/{budget} tokens used, {len(result)}/{len(chunks)} chunks kept")
        return result
