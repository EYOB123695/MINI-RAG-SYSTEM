"""
Main demo interface for the Self-Reflective RAG system.
Interactive CLI for querying the Ethiopian Commercial Code.
"""

import os
import sys
from typing import Optional

from self_reflective_rag.data_processing import process_document
from self_reflective_rag.vector_store import VectorStore
from self_reflective_rag.rag_pipeline import RAGPipeline
from self_reflective_rag.reflection import SelfReflectiveRAG
from self_reflective_rag import config


class RAGDemo:
    """Interactive demo interface for the Self-Reflective RAG system."""
    
    def __init__(self):
        """Initialize the demo system."""
        self.vector_store: Optional[VectorStore] = None
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.self_reflective_rag: Optional[SelfReflectiveRAG] = None
        self.initialized = False
    
    def initialize_system(self, force_rebuild: bool = False):
        """
        Initialize the RAG system: process documents and build vector store.
        
        Args:
            force_rebuild: If True, rebuild vector store even if it exists
        """
        print("\n" + "=" * 70)
        print(" " * 15 + "SELF-REFLECTIVE RAG SYSTEM")
        print(" " * 10 + "Ethiopian Commercial Code Question Answering")
        print("=" * 70)
        
        print("\n🚀 Initializing system...\n")
        
        # Step 1: Process the document
        print("📄 Step 1: Processing PDF document...")
        chunks = process_document(download_first=not os.path.exists(config.PDF_PATH))
        print(f"   ✓ Created {len(chunks)} text chunks\n")
        
        # Step 2: Build vector store
        print("🔍 Step 2: Building vector store...")
        self.vector_store = VectorStore()
        self.vector_store.build_vector_store(chunks, force_rebuild=force_rebuild)
        
        stats = self.vector_store.get_collection_stats()
        print(f"   ✓ Vector store ready with {stats['count']} documents\n")
        
        # Step 3: Initialize RAG pipeline
        print("🤖 Step 3: Initializing RAG pipeline...")
        self.rag_pipeline = RAGPipeline(self.vector_store)
        print("   ✓ RAG pipeline initialized\n")
        
        # Step 4: Initialize self-reflective RAG
        print("🧠 Step 4: Initializing self-reflection layer...")
        self.self_reflective_rag = SelfReflectiveRAG(self.rag_pipeline)
        print("   ✓ Self-reflection layer ready\n")
        
        self.initialized = True
        
        print("=" * 70)
        print("✅ System initialization complete!")
        print("=" * 70)
    
    def process_query(self, question: str, use_reflection: bool = True):
        """Process a user query and display results."""
        if not self.initialized:
            print("❌ System not initialized. Please initialize first.")
            return
        
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)
        
        if use_reflection:
            result = self.self_reflective_rag.generate_with_reflection(question, verbose=True)
        else:
            result = self.rag_pipeline.query(question, return_context=True)
            print(f"\n✓ Retrieved {result['num_chunks_retrieved']} chunks")
            print("\nANSWER:\n{result['answer']}")
    
    def run_interactive(self):
        """Run the interactive CLI interface."""
        if not self.initialized:
            self.initialize_system()
        
        print("\n💬 Ready! Type your question or 'exit' to quit.\n")
        
        while True:
            try:
                user_input = input("❓ Your question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                else:
                    self.process_query(user_input, use_reflection=True)
                    print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


def main():
    """Main entry point."""
    demo = RAGDemo()
    demo.run_interactive()


if __name__ == "__main__":
    main()
