# Self-Reflective RAG System

A Retrieval Augmented Generation (RAG) system with **self-reflection** capabilities for querying the Ethiopian Commercial Code (Proclamation No. 1243/2021).

##  What is Self-Reflective RAG?

Traditional RAG systems retrieve relevant documents and generate answers, but they don't verify answer quality. **Self-Reflective RAG** adds an evaluation layer that:

1. **Generates** an initial answer from retrieved context
2. **Reflects** on answer quality (relevance, grounding, completeness)
3. **Refines** the query and re-generates if quality is below threshold
4. **Iterates** up to 2 times to improve the answer

This ensures higher quality, more accurate responses!

##  Architecture

```
User Query
    ↓
[Retrieval] → Get relevant chunks from vector DB
    ↓
[Generation] → Generate answer with LLM
    ↓
[Reflection] → Evaluate answer quality (0-10 score)
    ↓
Quality OK? → YES → Return final answer
    ↓ NO
[Refinement] → Refine query based on feedback
    ↓
Repeat (max 2 iterations)
```

##  Technology Stack

- **LangChain** - RAG orchestration
- **ChromaDB** - Vector database (persistent storage)
- **Sentence Transformers** - Local embeddings (`all-MiniLM-L6-v2`)
- **Ollama** - Local LLM inference (llama2/mistral)
- **PyPDF2/pdfplumber** - PDF processing

## Prerequisites

### 1. Install Python 3.8+

Make sure you have Python installed:
```bash
python --version
```

### 2. Install Ollama

Ollama is required for running the LLM locally (free, no API key needed).

**Windows:**
- Download from: https://ollama.ai/download
- Run the installer
- Verify installation: `ollama --version`

**After installing Ollama, pull the model:**
```bash
ollama pull llama2
```

This downloads the llama2 model (~4GB). You only need to do this once.

##  Setup Instructions

### Step 1: Install Dependencies

Open terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

This installs all required Python packages.

### Step 2: Run the System

Simply run:

```bash
python main.py
```

On first run, the system will:
1. Download the Ethiopian Commercial Code PDF (~10MB)
2. Extract and chunk the text
3. Generate embeddings for all chunks
4. Build the vector database

**This takes 5-10 minutes on first run.** Subsequent runs are instant!

##  Usage

Once initialized, you can ask questions about Ethiopian Commercial Law:

```
 Your question: What are the requirements for forming a business organization?
```

The system will:
- Show retrieval progress
- Display the generated answer
- Show reflection scores
- Refine if needed (you'll see iterations)

### Example Questions

- "What are the requirements for forming a business organization?"
- "Explain the rules about commercial partnerships"
- "What are the obligations of a shareholder?"
- "How is a company dissolved according to the commercial code?"

##  Testing if Everything Works

### Quick Test

After installation, run:

```bash
python main.py
```

If you see this, everything works:
```
============================================================
           SELF-REFLECTIVE RAG SYSTEM
      Ethiopian Commercial Code Question Answering
============================================================

 Initializing system...

 Step 1: Processing PDF document...
```

### Test Individual Components

**Test data processing:**
```bash
python -m self_reflective_rag.data_processing
```

**Test vector store:**
```bash
python -m self_reflective_rag.vector_store
```

**Test RAG pipeline:**
```bash
python -m self_reflective_rag.rag_pipeline
```

**Test self-reflection:**
```bash
python -m self_reflective_rag.reflection
```

## Project Structure

```
New folder/
├── self_reflective_rag/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── data_processing.py    # PDF download & chunking
│   ├── vector_store.py       # Embeddings & ChromaDB
│   ├── rag_pipeline.py       # Retrieval & generation
│   └── reflection.py         # Self-reflection logic
├── data/
│   └── commercial_code.pdf   # Downloaded PDF (auto-created)
├── vector_db/                # ChromaDB storage (auto-created)
├── requirements.txt
├── main.py                   # Interactive demo
└── README.md
```

##  Configuration

Edit `self_reflective_rag/config.py` to customize:

- **LLM Model**: Change `LLM_MODEL` (e.g., "mistral", "llama3")
- **Chunk Size**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP`
- **Retrieval**: Modify `TOP_K_RETRIEVAL` (how many chunks to retrieve)
- **Reflection**: Set `REFLECTION_THRESHOLD` and `MAX_REFLECTION_ITERATIONS`

##  Troubleshooting

### "Ollama not found" error

Make sure Ollama is installed and running:
```bash
ollama --version
ollama list  # Shows installed models
```

If llama2 is not listed:
```bash
ollama pull llama2
```

### "Model not found" error

Pull the model specified in config:
```bash
ollama pull llama2
```

### ChromaDB errors

Delete the vector database and rebuild:
```bash
rmdir /s vector_db  # Windows
python main.py      # Rebuild
```

### PDF download fails

Manually download the PDF from:
https://ethiodata.et/wp-content/uploads/2023/01/Commercial-Code-of-Ethiopia-English-Proclamation-No.-1243_2021.pdf

Save it to: `data/commercial_code.pdf`

##  How Self-Reflection Works

Each answer is evaluated on 3 criteria:

1. **Relevance** (40%): Does it address the question?
2. **Grounding** (40%): Is it based on retrieved context (not hallucinated)?
3. **Completeness** (20%): Does it fully answer the question?

If the overall score is below 7.0/10, the system:
- Analyzes what went wrong
- Refines the query to be more specific
- Re-retrieves context and re-generates
- Evaluates again

This happens up to 2 times, ensuring high-quality answers!

##  Notes

- **First run**: Takes 5-10 minutes to process the PDF and build the vector store
- **Subsequent runs**: Instant startup (uses cached vector DB)
- **Answer generation**: Each answer takes ~10-30 seconds depending on your hardware
- **Reflection iterations**: If reflection triggers, expect 20-60 seconds total

## 🎓 Example Session

```
Your question: What is a commercial business?

============================================================
SELF-REFLECTIVE RAG GENERATION
============================================================

 Iteration 1
Query: What is a commercial business?

Generated Answer:
According to the Ethiopian Commercial Code, a commercial business refers to...

Reflecting on answer quality...

Reflection Scores:
   Relevance: 9.0/10
   Grounding: 8.5/10
   Completeness: 8.0/10
   Overall: 8.5/10

Answer quality meets threshold (7.0). Stopping.

============================================================
FINAL RESULT
============================================================

Final Answer:
[High-quality answer based on the Commercial Code]

Total Iterations: 1
Answer Improved: No
============================================================
```

##  Contributing

Feel free to modify and extend this system! Some ideas:
- Add more data sources
- Implement web interface
- Add citation extraction
- Support multiple languages

##  License

This project uses the Ethiopian Commercial Code (Proclamation No. 1243/2021) as the knowledge base, which is a public legal document.

---

**Enjoy exploring Ethiopian Commercial Law with AI! 🇪🇹**
