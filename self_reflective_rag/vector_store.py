"""
Vector store module for the Self-Reflective RAG system.
Handles embedding generation and vector database operations using ChromaDB.
"""

import os
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from . import config


class VectorStore:
    """
    Manages embeddings and vector database operations for efficient retrieval.
    """
    
    def __init__(self, collection_name: str = config.COLLECTION_NAME):
        """
        Initialize the vector store with embedding model and ChromaDB client.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        print("Initializing Vector Store...")
        
        # Initialize embedding model (runs locally, no API needed)
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB client with persistent storage
        os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = None
        
        print("Vector Store initialized successfully")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings with progress bar
        embeddings = []
        batch_size = 32  # Process in batches for efficiency
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings.tolist())
        
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def build_vector_store(self, chunks: List[Dict[str, any]], force_rebuild: bool = False):
        """
        Build the vector store from document chunks.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            force_rebuild: If True, delete existing collection and rebuild
        """
        print("=" * 60)
        print("BUILDING VECTOR STORE")
        print("=" * 60)
        
        # Check if collection already exists
        existing_collections = [col.name for col in self.client.list_collections()]
        
        if self.collection_name in existing_collections:
            if force_rebuild:
                print(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"Collection '{self.collection_name}' already exists. Loading...")
                self.collection = self.client.get_collection(self.collection_name)
                print(f"Loaded collection with {self.collection.count()} documents")
                return
        
        # Create new collection
        print(f"Creating new collection: {self.collection_name}")
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Ethiopian Commercial Code embeddings"}
        )
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.create_embeddings(texts)
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{chunk['id']}" for chunk in chunks]
        metadatas = [
            {
                'start_char': chunk['start_char'],
                'end_char': chunk['end_char'],
                'size': chunk['size']
            }
            for chunk in chunks
        ]
        
        # Add to collection in batches (ChromaDB has batch size limits)
        print("Adding documents to vector store...")
        batch_size = 100
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to ChromaDB"):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        print("=" * 60)
        print(f"Vector store built successfully with {len(chunks)} documents")
        print("=" * 60)
    
    def retrieve_relevant_chunks(
        self,
        query: str,
        k: int = config.TOP_K_RETRIEVAL
    ) -> List[Dict[str, any]]:
        """
        Retrieve the top-k most relevant chunks for a query.
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
            
        Returns:
            List of dictionaries with chunk text, metadata, and relevance scores
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call build_vector_store first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )[0].tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        retrieved_chunks = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                chunk = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def search(self, query: str, k: int = config.TOP_K_RETRIEVAL) -> Tuple[List[str], List[Dict]]:
        """
        Search for relevant chunks and return texts and metadata separately.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            Tuple of (list of chunk texts, list of metadata dicts)
        """
        chunks = self.retrieve_relevant_chunks(query, k)
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        return texts, metadatas
    
    def get_collection_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if self.collection is None:
            return {"status": "not_initialized"}
        
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "status": "ready"
        }


if __name__ == "__main__":
    # Test the vector store
    from .data_processing import process_document
    
    print("Testing Vector Store...")
    print()
    
    # Process document
    chunks = process_document()
    
    # Build vector store
    vs = VectorStore()
    vs.build_vector_store(chunks)
    
    # Test retrieval
    print("\nTesting retrieval...")
    test_query = "What are the requirements for forming a business organization?"
    results = vs.retrieve_relevant_chunks(test_query, k=3)
    
    print(f"\nQuery: {test_query}")
    print(f"Retrieved {len(results)} chunks:\n")
    
    for i, chunk in enumerate(results, 1):
        print(f"Chunk {i} (ID: {chunk['id']}):")
        print(f"Distance: {chunk['distance']:.4f}")
        print(f"Text preview: {chunk['text'][:200]}...")
        print("-" * 60)
