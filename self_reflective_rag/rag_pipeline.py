"""
RAG Pipeline module for the Self-Reflective RAG system.
Implements core retrieval and generation logic using Ollama LLM.
"""

from typing import List, Dict, Tuple
import ollama

from . import config
from .vector_store import VectorStore


class RAGPipeline:
    """
    Implements the Retrieval Augmented Generation pipeline.
    Combines vector search with LLM generation.
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Pre-initialized VectorStore instance
        """
        self.vector_store = vector_store
        self.llm_model = config.LLM_MODEL
        
        print(f"RAG Pipeline initialized with model: {self.llm_model}")
        
        # Verify Ollama is available and model is pulled
        try:
            # List available models
            models = ollama.list()
            available_models = [m['name'] for m in models.get('models', [])]
            
            if not any(self.llm_model in model for model in available_models):
                print(f"\n⚠️  WARNING: Model '{self.llm_model}' not found in Ollama.")
                print(f"Available models: {available_models}")
                print(f"\nTo pull the model, run: ollama pull {self.llm_model}")
                print("The system will attempt to use it anyway, but may fail.\n")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify Ollama status: {str(e)}")
            print("Make sure Ollama is installed and running.\n")
    
    def retrieve(self, query: str, k: int = config.TOP_K_RETRIEVAL) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            Tuple of (list of chunk texts, list of metadata)
        """
        return self.vector_store.search(query, k=k)
    
    def format_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Format the prompt for the LLM with query and retrieved context.
        
        Args:
            query: User query
            context_chunks: List of retrieved text chunks
            
        Returns:
            Formatted prompt string
        """
        # Combine context chunks with numbering for better readability
        context_str = "\n\n".join([
            f"[Context {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Use configured prompt template
        prompt = config.RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=query
        )
        
        return prompt
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[str],
        temperature: float = config.LLM_TEMPERATURE
    ) -> str:
        """
        Generate an answer using the LLM with retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            temperature: LLM temperature for generation
            
        Returns:
            Generated answer text
        """
        # Format the prompt
        prompt = self.format_prompt(query, context_chunks)
        
        try:
            # Call Ollama API for generation
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': config.MAX_TOKENS
                }
            )
            
            # Extract the generated text
            answer = response['response'].strip()
            return answer
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(f"❌ {error_msg}")
            return f"[Error: {error_msg}]"
    
    def query(
        self,
        question: str,
        k: int = config.TOP_K_RETRIEVAL,
        return_context: bool = False
    ) -> Dict[str, any]:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            return_context: If True, include retrieved context in response
            
        Returns:
            Dictionary with answer and optional metadata
        """
        # Step 1: Retrieve relevant context
        context_chunks, metadatas = self.retrieve(question, k=k)
        
        # Step 2: Generate answer
        answer = self.generate_answer(question, context_chunks)
        
        # Prepare response
        result = {
            'question': question,
            'answer': answer,
            'num_chunks_retrieved': len(context_chunks)
        }
        
        # Optionally include context
        if return_context:
            result['context'] = context_chunks
            result['metadata'] = metadatas
        
        return result


def call_llm(prompt: str, model: str = config.LLM_MODEL, temperature: float = 0.7) -> str:
    """
    Helper function to call Ollama LLM with a prompt.
    Used by other modules (e.g., reflection) for LLM calls.
    
    Args:
        prompt: Prompt text
        model: Ollama model name
        temperature: Generation temperature
        
    Returns:
        Generated response text
    """
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                'temperature': temperature,
                'num_predict': config.MAX_TOKENS
            }
        )
        return response['response'].strip()
    except Exception as e:
        print(f"❌ Error calling LLM: {str(e)}")
        return f"[LLM Error: {str(e)}]"


if __name__ == "__main__":
    # Test the RAG pipeline
    from .data_processing import process_document
    
    print("Testing RAG Pipeline...")
    print()
    
    # Setup: process document and build vector store
    chunks = process_document()
    
    vs = VectorStore()
    vs.build_vector_store(chunks)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(vs)
    
    # Test query
    print("\n" + "=" * 60)
    print("TESTING RAG QUERY")
    print("=" * 60)
    
    test_question = "What are the requirements for forming a business organization in Ethiopia?"
    
    result = rag.query(test_question, return_context=True)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nRetrieved {result['num_chunks_retrieved']} relevant chunks")
    print("\n" + "-" * 60)
    print("ANSWER:")
    print("-" * 60)
    print(result['answer'])
    print("-" * 60)
