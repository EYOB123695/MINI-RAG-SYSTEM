"""
Data processing module for the Self-Reflective RAG system.
Handles PDF downloading, text extraction, and document chunking.
"""

import os
import requests
from typing import List, Dict
import PyPDF2
import pdfplumber
from tqdm import tqdm

from . import config


def download_pdf(url: str = config.PDF_URL, save_path: str = config.PDF_PATH) -> str:
    """
    Download PDF from URL to local storage.
    
    Args:
        url: URL of the PDF to download
        save_path: Local path to save the PDF
        
    Returns:
        Path to the downloaded PDF
        
    Raises:
        Exception: If download fails
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(save_path):
        print(f"PDF already exists at {save_path}")
        return save_path
    
    print(f"Downloading PDF from {url}...")
    
    try:
        # Download with streaming to handle large files
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        # Write file with progress bar
        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"PDF downloaded successfully to {save_path}")
        return save_path
        
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")


def extract_text_from_pdf(pdf_path: str = config.PDF_PATH, use_pdfplumber: bool = True) -> str:
    """
    Extract text content from PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        use_pdfplumber: If True, use pdfplumber (better quality), else use PyPDF2
        
    Returns:
        Extracted text as a single string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If extraction fails
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    print(f"Extracting text from {pdf_path}...")
    
    try:
        if use_pdfplumber:
            # pdfplumber generally gives better text extraction quality
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            
            full_text = "\n\n".join(text_content)
        else:
            # Fallback to PyPDF2
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for i in tqdm(range(len(pdf_reader.pages)), desc="Extracting pages"):
                    page = pdf_reader.pages[i]
                    text_content.append(page.extract_text())
            
            full_text = "\n\n".join(text_content)
        
        print(f"Extracted {len(full_text)} characters from {len(text_content)} pages")
        return full_text
        
    except Exception as e:
        # If pdfplumber fails, try PyPDF2 as fallback
        if use_pdfplumber:
            print(f"pdfplumber failed, trying PyPDF2... Error: {str(e)}")
            return extract_text_from_pdf(pdf_path, use_pdfplumber=False)
        else:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP
) -> List[Dict[str, any]]:
    """
    Split text into overlapping chunks for better context preservation.
    
    Args:
        text: Full text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of dictionaries containing chunk text and metadata
    """
    print(f"Chunking text into {chunk_size}-character chunks with {overlap} overlap...")
    
    chunks = []
    start = 0
    chunk_id = 0
    
    # Simple character-based chunking with overlap
    while start < len(text):
        # Define chunk boundaries
        end = start + chunk_size
        
        # Get the chunk text
        chunk_text = text[start:end]
        
        # Try to break at sentence or paragraph boundaries for cleaner chunks
        if end < len(text):
            # Look for paragraph break first
            last_paragraph = chunk_text.rfind('\n\n')
            if last_paragraph > chunk_size * 0.5:  # Only break if it's not too early
                end = start + last_paragraph + 2
                chunk_text = text[start:end]
            else:
                # Look for sentence break
                last_period = chunk_text.rfind('. ')
                if last_period > chunk_size * 0.5:
                    end = start + last_period + 2
                    chunk_text = text[start:end]
        
        # Create chunk with metadata
        chunk = {
            'id': chunk_id,
            'text': chunk_text.strip(),
            'start_char': start,
            'end_char': end,
            'size': len(chunk_text)
        }
        
        chunks.append(chunk)
        
        # Move to next chunk with overlap
        start = end - overlap
        chunk_id += 1
    
    print(f"Created {len(chunks)} chunks")
    return chunks


def process_document(download_first: bool = True) -> List[Dict[str, any]]:
    """
    Complete pipeline: download, extract, and chunk the document.
    
    Args:
        download_first: If True, download the PDF before processing
        
    Returns:
        List of text chunks with metadata
    """
    print("=" * 60)
    print("DOCUMENT PROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Download PDF if needed
    if download_first or not os.path.exists(config.PDF_PATH):
        download_pdf()
    
    # Step 2: Extract text
    full_text = extract_text_from_pdf()
    
    # Step 3: Chunk text
    chunks = chunk_text(full_text)
    
    print("=" * 60)
    print(f"Processing complete! {len(chunks)} chunks ready for embedding.")
    print("=" * 60)
    
    return chunks


if __name__ == "__main__":
    # Test the data processing pipeline
    chunks = process_document()
    
    # Display sample chunk
    if chunks:
        print("\nSample chunk:")
        print("-" * 60)
        print(f"Chunk ID: {chunks[0]['id']}")
        print(f"Size: {chunks[0]['size']} characters")
        print(f"Text preview: {chunks[0]['text'][:200]}...")
