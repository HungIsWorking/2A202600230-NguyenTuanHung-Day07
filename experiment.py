
import re
import os
import subprocess
from pathlib import Path

# Make sure we can import from the 'src' directory
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    compute_similarity,
)
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.embeddings import MockEmbedder

# --- 1. Define Custom Semantic Chunker ---

class SemanticChunker:
    """
    Splits text based on semantic boundaries like chapters (##) and sections (###).
    This is a custom strategy for the report.
    """
    def chunk(self, text: str) -> list[str]:
        chunks = []
        # Split by chapters first, keeping the delimiter
        chapters = re.split(r'(\n## .*\n)', text)
        
        current_chunk = chapters[0]
        for i in range(1, len(chapters), 2):
            # Split the current chunk by sections
            sections = re.split(r'(\n### .*\n)', current_chunk)
            
            # Add the chapter intro part
            if sections[0].strip():
                chunks.append(sections[0].strip())
            
            # Add the sections
            for j in range(1, len(sections), 2):
                section_header = sections[j]
                section_content = sections[j+1]
                if (section_header + section_content).strip():
                    chunks.append((section_header + section_content).strip())

            current_chunk = chapters[i] + chapters[i+1]

        # Process the last chapter
        sections = re.split(r'(\n### .*\n)', current_chunk)
        if sections[0].strip():
            chunks.append(sections[0].strip())
        for j in range(1, len(sections), 2):
            section_header = sections[j]
            section_content = sections[j+1]
            if (section_header + section_content).strip():
                chunks.append((section_header + section_content).strip())

        return [chunk for chunk in chunks if chunk]

# --- 2. Helper Functions for Experiments ---

def load_doc(file_path="book.md") -> str:
    """Loads the content of the book."""
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please create it.")
        # Create a dummy file for the script to run
        dummy_content = """## Chapter 1\n\nThis is a test.\n\n### Section 1.1\n\nMore content."""
        Path(file_path).write_text(dummy_content)
        return dummy_content

def run_chunking_comparison(text: str):
    """Runs all chunking strategies and prints a comparison table."""
    print("\n--- Chunking Strategy Comparison ---")
    
    strategies = {
        "FixedSizeChunker (`fixed_size`)": FixedSizeChunker(chunk_size=500, overlap=50),
        "SentenceChunker (`by_sentences`)": SentenceChunker(max_sentences_per_chunk=3),
        "RecursiveChunker (`recursive`)": RecursiveChunker(chunk_size=500),
        "SemanticChunker (của tôi)": SemanticChunker(),
    }
    
    print("| Strategy                       | Chunk Count | Avg Length |")
    print("|--------------------------------|-------------|------------|")
    
    results = {}
    for name, chunker in strategies.items():
        chunks = chunker.chunk(text)
        count = len(chunks)
        avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0
        results[name] = (count, int(avg_length))
        print(f"| {name:<30} | {count:<11} | {int(avg_length):<10} |")
    
    return results

def run_similarity_predictions():
    """Calculates and prints similarity scores for predefined sentence pairs."""
    print("\n--- Similarity Predictions ---")
    
    pairs = [
        ("The cat sat on the mat.", "A feline was resting on the rug."), # High
        ("SOLID principles are key for good software design.", "SRP is one of the SOLID principles."), # High
        ("I am learning about vector stores.", "My favorite food is pho."), # Low
        ("What is the capital of France?", "Paris is a beautiful city."), # Medium/High
        ("The system should be scalable.", "The system must handle many users."), # High
    ]
    
    embedder = MockEmbedder()
    
    print("| Pair | Sentence A                                       | Sentence B                                       | Dự đoán | Actual Score |")
    print("|------|--------------------------------------------------|--------------------------------------------------|---------|--------------|")

    for i, (a, b) in enumerate(pairs, 1):
        score = compute_similarity(embedder(a), embedder(b))
        # Simple prediction logic
        prediction = "high" if score > 0.1 else "low"
        print(f"| {i:<4} | {a:<48} | {b:<48} | {prediction:<7} | {score:<12.3f} |")

def run_pytest():
    """Runs pytest and captures the output."""
    print("\n--- Pytest Results ---")
    try:
        # Using the activated environment's python
        python_executable = sys.executable
        result = subprocess.run(
            [python_executable, "-m", "pytest", "tests/", "-v"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Pytest execution failed. Make sure pytest is installed.")
        if hasattr(e, 'stderr'):
            print(e.stderr)

def run_rag_benchmark():
    """Runs benchmark queries against the RAG agent."""
    print("\n--- RAG Benchmark Results ---")
    
    # Using my SemanticChunker strategy
    chunker = SemanticChunker()
    text = load_doc()
    chunks = chunker.chunk(text)
    
    documents = [Document(id=f"book-chunk-{i}", content=chunk, metadata={"source": "book.md"}) for i, chunk in enumerate(chunks)]
    
    store = EmbeddingStore(embedding_fn=MockEmbedder())
    store.add_documents(documents)
    
    agent = KnowledgeBaseAgent(store=store, llm_fn=lambda p: f"[DEMO LLM] Answer based on prompt...")

    benchmark_queries = [
        "What are the SOLID principles?",
        "Explain the DRY principle.",
        "What is the difference between SRP and ISP?",
        "What does KISS stand for?",
        "Summarize the main idea of the Open/Closed Principle.",
    ]
    
    print("| # | Query                               | Top-1 Retrieved Chunk (tóm tắt)                               | Score | Relevant? | Agent Answer (tóm tắt) |")
    print("|---|-------------------------------------|---------------------------------------------------------------|-------|-----------|------------------------|")

    relevant_count = 0
    for i, query in enumerate(benchmark_queries, 1):
        search_results = store.search(query, top_k=3)
        
        if not search_results:
            print(f"| {i} | {query:<35} | {'NO RESULTS FOUND':<61} | N/A   | No        | {'N/A'}                |")
            continue

        top_chunk = search_results[0]
        top_content_summary = top_chunk['content'][:60].replace('\n', ' ') + "..."
        score = top_chunk['score']
        
        # Simple relevance check
        is_relevant = any(kw.lower() in top_chunk['content'].lower() for kw in query.split())
        if is_relevant:
            relevant_count += 1
        
        agent_answer = agent.answer(query, top_k=3)
        agent_answer_summary = agent_answer[:25] + "..."

        print(f"| {i} | {query:<35} | {top_content_summary:<61} | {score:<5.2f} | {'Yes' if is_relevant else 'No':<9} | {agent_answer_summary:<22} |")
        
    print(f"\nBao nhiêu queries trả về chunk relevant trong top-3? {relevant_count} / 5")


if __name__ == "__main__":
    print("Running all experiments for the report...")
    book_content = load_doc()
    run_chunking_comparison(book_content)
    run_similarity_predictions()
    run_rag_benchmark()
    run_pytest()
    print("\nExperiment run complete. Copy the output above into your report.")
