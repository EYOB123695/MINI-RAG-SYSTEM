"""
Configuration settings for the Self-Reflective RAG system.
Centralizes all configuration parameters for easy modification.
"""

import os

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

# PDF URL for the Ethiopian Commercial Code
PDF_URL = "https://ethiodata.et/wp-content/uploads/2023/01/Commercial-Code-of-Ethiopia-English-Proclamation-No.-1243_2021.pdf"

# Local paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "commercial_code.pdf")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# ============================================================================
# TEXT PROCESSING CONFIGURATION
# ============================================================================

# Chunking parameters
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

# Sentence Transformer model for embeddings
# all-MiniLM-L6-v2: Fast, good quality, runs locally
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================================
# VECTOR STORE CONFIGURATION
# ============================================================================

# ChromaDB collection name
COLLECTION_NAME = "ethiopian_commercial_code"

# Number of relevant chunks to retrieve
TOP_K_RETRIEVAL = 4

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# Ollama model for generation and reflection
# Options: "llama2", "mistral", "llama3", "phi", etc.
# Make sure the model is pulled: ollama pull <model_name>
LLM_MODEL = "llama2"

# LLM temperature (0.0 = deterministic, 1.0 = creative)
LLM_TEMPERATURE = 0.7

# Maximum tokens for generation
MAX_TOKENS = 1024

# ============================================================================
# SELF-REFLECTION CONFIGURATION
# ============================================================================

# Maximum number of reflection iterations
MAX_REFLECTION_ITERATIONS = 2

# Reflection score threshold (0-10 scale)
# If score < threshold, trigger re-retrieval and refinement
REFLECTION_THRESHOLD = 7.0

# Reflection criteria weights (should sum to 1.0)
REFLECTION_WEIGHTS = {
    "relevance": 0.4,      # Does it answer the question?
    "grounding": 0.4,      # Is it based on retrieved context?
    "completeness": 0.2    # Is the answer complete?
}

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

# RAG generation prompt
RAG_PROMPT_TEMPLATE = """You are an expert on Ethiopian Commercial Law. Answer the following question based ONLY on the provided context from the Ethiopian Commercial Code.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- If the context doesn't contain enough information, say so
- Cite relevant sections or articles when possible
- Be concise but thorough

Answer:"""

# Reflection evaluation prompt
REFLECTION_PROMPT_TEMPLATE = """You are a quality evaluator for legal question-answering systems. Evaluate the following answer:

Question: {question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the answer on these criteria (rate each 0-10):

1. RELEVANCE: Does the answer directly address the question?
2. GROUNDING: Is the answer based on the provided context (not hallucinated)?
3. COMPLETENESS: Does the answer fully address all parts of the question?

Provide your evaluation in this exact format:
RELEVANCE: [score]
GROUNDING: [score]
COMPLETENESS: [score]
OVERALL: [average score]
REASONING: [brief explanation]

Evaluation:"""

# Query refinement prompt
QUERY_REFINEMENT_PROMPT = """The following question was asked, but the answer was not satisfactory.

Original Question: {original_query}

Issues identified:
{feedback}

Generate an improved, more specific version of the question that might retrieve better context. Only output the refined question, nothing else.

Refined Question:"""
