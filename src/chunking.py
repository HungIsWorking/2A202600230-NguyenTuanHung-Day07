from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Split into sentences using regex
        # Match sentence boundaries: '. ', '! ', '? ', or '.\n'
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks: list[str] = []
        current_chunk: list[str] = []
        sentence_count = 0
        
        for sentence in sentences:
            current_chunk.append(sentence)
            sentence_count += 1
            
            if sentence_count >= self.max_sentences_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                sentence_count = 0
        
        # Add remaining sentences
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: if current text fits in chunk_size, return it
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []
        
        # Base case: no more separators to try
        if not remaining_separators:
            # Force split at chunk_size boundary
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunk = current_text[i:i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks
        
        # Try the first separator
        separator = remaining_separators[0]
        rest = remaining_separators[1:]
        
        # Split by the separator
        if separator == "":
            # Empty separator means split character by character
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunk = current_text[i:i + self.chunk_size]
                if chunk:
                    chunks.append(chunk)
            return chunks
        
        parts = current_text.split(separator)
        
        # Try to merge parts while respecting chunk size
        good_chunks = []
        for i, part in enumerate(parts):
            # Reconstruct with separator if not the last part
            reconstructed = part if i == len(parts) - 1 else part + separator
            
            if len(reconstructed) <= self.chunk_size:
                good_chunks.append(reconstructed)
            else:
                # This part is too big, need to recursively split it
                sub_chunks = self._split(part, rest)
                good_chunks.extend(sub_chunks)
        
        # Merge consecutive small chunks if possible
        merged = []
        current = ""
        for chunk in good_chunks:
            if len(current) + len(chunk) <= self.chunk_size:
                current += chunk
            else:
                if current:
                    merged.append(current)
                current = chunk
        if current:
            merged.append(current)
        
        return merged


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # Compute dot product
    dot_product = _dot(vec_a, vec_b)
    
    # Compute magnitudes
    mag_a = math.sqrt(sum(x * x for x in vec_a)) or 1.0
    mag_b = math.sqrt(sum(x * x for x in vec_b)) or 1.0
    
    # Avoid division by zero
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Call each chunking strategy
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=0)
        fixed_chunks = fixed_chunker.chunk(text)
        
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        sentence_chunks = sentence_chunker.chunk(text)
        
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        recursive_chunks = recursive_chunker.chunk(text)
        
        # Compute statistics for each strategy
        result = {
            'fixed_size': {
                'count': len(fixed_chunks),
                'avg_length': sum(len(c) for c in fixed_chunks) / len(fixed_chunks) if fixed_chunks else 0,
                'chunks': fixed_chunks,
            },
            'by_sentences': {
                'count': len(sentence_chunks),
                'avg_length': sum(len(c) for c in sentence_chunks) / len(sentence_chunks) if sentence_chunks else 0,
                'chunks': sentence_chunks,
            },
            'recursive': {
                'count': len(recursive_chunks),
                'avg_length': sum(len(c) for c in recursive_chunks) / len(recursive_chunks) if recursive_chunks else 0,
                'chunks': recursive_chunks,
            },
        }
        
        return result
