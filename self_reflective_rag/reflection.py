"""
Self-Reflection module for the Self-Reflective RAG system.
Implements quality evaluation and iterative refinement of generated answers.
"""

import re
from typing import List, Dict, Tuple

from . import config
from .rag_pipeline import call_llm


class ReflectionEvaluator:
    """
    Evaluates the quality of generated answers and triggers refinement if needed.
    """
    
    def __init__(self):
        """Initialize the reflection evaluator."""
        self.model = config.LLM_MODEL
        self.threshold = config.REFLECTION_THRESHOLD
        self.max_iterations = config.MAX_REFLECTION_ITERATIONS
    
    def reflect_on_answer(
        self,
        question: str,
        answer: str,
        context: List[str]
    ) -> Dict[str, any]:
        """
        Evaluate the quality of a generated answer using the LLM as a judge.
        
        Args:
            question: Original user question
            answer: Generated answer to evaluate
            context: Retrieved context chunks used for generation
            
        Returns:
            Dictionary with evaluation scores and reasoning
        """
        # Format context for the reflection prompt
        context_str = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
        
        # Create reflection prompt
        reflection_prompt = config.REFLECTION_PROMPT_TEMPLATE.format(
            question=question,
            context=context_str,
            answer=answer
        )
        
        # Get LLM evaluation
        evaluation_text = call_llm(reflection_prompt, temperature=0.3)  # Lower temp for consistent evaluation
        
        # Parse the evaluation scores
        scores = self._parse_evaluation(evaluation_text)
        
        return scores
    
    def _parse_evaluation(self, evaluation_text: str) -> Dict[str, any]:
        """
        Parse LLM evaluation output to extract scores.
        
        Args:
            evaluation_text: Raw evaluation text from LLM
            
        Returns:
            Dictionary with parsed scores and reasoning
        """
        # Initialize default scores
        scores = {
            'relevance': 5.0,
            'grounding': 5.0,
            'completeness': 5.0,
            'overall': 5.0,
            'reasoning': 'Could not parse evaluation',
            'raw_evaluation': evaluation_text
        }
        
        try:
            # Extract scores using regex
            relevance_match = re.search(r'RELEVANCE:\s*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)
            grounding_match = re.search(r'GROUNDING:\s*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)
            completeness_match = re.search(r'COMPLETENESS:\s*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)
            overall_match = re.search(r'OVERALL:\s*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n\n|$)', evaluation_text, re.IGNORECASE | re.DOTALL)
            
            # Update scores if found
            if relevance_match:
                scores['relevance'] = float(relevance_match.group(1))
            if grounding_match:
                scores['grounding'] = float(grounding_match.group(1))
            if completeness_match:
                scores['completeness'] = float(completeness_match.group(1))
            if overall_match:
                scores['overall'] = float(overall_match.group(1))
            else:
                # Calculate overall as weighted average if not provided
                weights = config.REFLECTION_WEIGHTS
                scores['overall'] = (
                    scores['relevance'] * weights['relevance'] +
                    scores['grounding'] * weights['grounding'] +
                    scores['completeness'] * weights['completeness']
                ) / sum(weights.values())
            
            if reasoning_match:
                scores['reasoning'] = reasoning_match.group(1).strip()
            
        except Exception as e:
            print(f"⚠️  Warning: Error parsing evaluation: {str(e)}")
        
        return scores
    
    def refine_query(self, original_query: str, feedback: str) -> str:
        """
        Generate a refined query based on reflection feedback.
        
        Args:
            original_query: Original user query
            feedback: Feedback from reflection about what's missing
            
        Returns:
            Refined query string
        """
        # Create query refinement prompt
        refinement_prompt = config.QUERY_REFINEMENT_PROMPT.format(
            original_query=original_query,
            feedback=feedback
        )
        
        # Get refined query from LLM
        refined_query = call_llm(refinement_prompt, temperature=0.5)
        
        # Clean up the response (sometimes LLM adds extra text)
        refined_query = refined_query.strip()
        
        return refined_query


class SelfReflectiveRAG:
    """
    Self-reflective RAG pipeline that iteratively improves answers.
    """
    
    def __init__(self, rag_pipeline):
        """
        Initialize self-reflective RAG.
        
        Args:
            rag_pipeline: Standard RAGPipeline instance
        """
        self.rag = rag_pipeline
        self.evaluator = ReflectionEvaluator()
    
    def generate_with_reflection(
        self,
        question: str,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Generate answer with self-reflection and iterative refinement.
        
        This is the main method that implements the self-reflective loop:
        1. Retrieve context and generate initial answer
        2. Reflect on answer quality
        3. If quality is low, refine query and retry
        4. Repeat up to max_iterations
        
        Args:
            question: User question
            verbose: If True, print detailed progress
            
        Returns:
            Dictionary with final answer, reflection scores, and iteration history
        """
        if verbose:
            print("\n" + "=" * 60)
            print("SELF-REFLECTIVE RAG GENERATION")
            print("=" * 60)
        
        iteration_history = []
        current_query = question
        final_answer = None
        final_context = None
        final_scores = None
        
        # Iterative refinement loop
        for iteration in range(self.evaluator.max_iterations + 1):
            if verbose:
                print(f"\n📍 Iteration {iteration + 1}")
                print(f"Query: {current_query}")
            
            # Step 1: Retrieve and generate
            result = self.rag.query(current_query, return_context=True)
            answer = result['answer']
            context = result['context']
            
            if verbose:
                print(f"\n💡 Generated Answer:")
                print(f"{answer[:300]}{'...' if len(answer) > 300 else ''}")
            
            # Step 2: Reflect on the answer
            if verbose:
                print(f"\n🤔 Reflecting on answer quality...")
            
            scores = self.evaluator.reflect_on_answer(question, answer, context)
            
            if verbose:
                print(f"\n📊 Reflection Scores:")
                print(f"   Relevance: {scores['relevance']:.1f}/10")
                print(f"   Grounding: {scores['grounding']:.1f}/10")
                print(f"   Completeness: {scores['completeness']:.1f}/10")
                print(f"   Overall: {scores['overall']:.1f}/10")
                print(f"   Reasoning: {scores['reasoning']}")
            
            # Record iteration
            iteration_history.append({
                'iteration': iteration + 1,
                'query': current_query,
                'answer': answer,
                'scores': scores,
                'context_chunks': len(context)
            })
            
            # Step 3: Check if answer is good enough
            if scores['overall'] >= self.evaluator.threshold:
                if verbose:
                    print(f"\n✅ Answer quality meets threshold ({self.evaluator.threshold}). Stopping.")
                final_answer = answer
                final_context = context
                final_scores = scores
                break
            
            # Step 4: Refine if not at max iterations
            if iteration < self.evaluator.max_iterations:
                if verbose:
                    print(f"\n🔄 Score below threshold. Refining query...")
                
                # Generate refined query based on feedback
                refined_query = self.evaluator.refine_query(
                    question,
                    scores['reasoning']
                )
                
                if verbose:
                    print(f"Refined query: {refined_query}")
                
                current_query = refined_query
            else:
                # Max iterations reached
                if verbose:
                    print(f"\n⚠️  Max iterations reached. Using best available answer.")
                final_answer = answer
                final_context = context
                final_scores = scores
        
        # Prepare final result
        result = {
            'question': question,
            'final_answer': final_answer,
            'final_scores': final_scores,
            'iterations': len(iteration_history),
            'iteration_history': iteration_history,
            'improved': len(iteration_history) > 1,
            'context': final_context
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL RESULT")
            print("=" * 60)
            print(f"\nFinal Answer:\n{final_answer}")
            print(f"\nTotal Iterations: {result['iterations']}")
            print(f"Answer Improved: {'Yes' if result['improved'] else 'No'}")
            print("=" * 60)
        
        return result


if __name__ == "__main__":
    # Test the self-reflective RAG
    from .data_processing import process_document
    from .vector_store import VectorStore
    from .rag_pipeline import RAGPipeline
    
    print("Testing Self-Reflective RAG...")
    print()
    
    # Setup
    chunks = process_document()
    vs = VectorStore()
    vs.build_vector_store(chunks)
    rag = RAGPipeline(vs)
    
    # Create self-reflective RAG
    self_rag = SelfReflectiveRAG(rag)
    
    # Test with a question
    test_question = "What are the rules about commercial partnerships?"
    
    result = self_rag.generate_with_reflection(test_question, verbose=True)
