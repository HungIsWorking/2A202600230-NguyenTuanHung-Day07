from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # Store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Retrieve chunks from the knowledge base
        retrieved_chunks = self.store.search(question, top_k=top_k)
        
        # Build a prompt with the retrieved chunks as context
        context = "".join([
            f"- {chunk['content']}\n"
            for chunk in retrieved_chunks
        ])
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Call the LLM to generate an answer
        answer = self.llm_fn(prompt)
        return answer
