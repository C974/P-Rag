# Palestine RAG System (P-RAG)

A comprehensive Retrieval-Augmented Generation (RAG) system specifically designed for Palestine-related knowledge base queries and benchmarking. This system combines document processing, semantic search, and language model generation with specialized focus on Palestinian historical, political, and cultural content.

## 🏛️ Project Overview

The Palestine RAG System consists of three main components:

1. **P-RAG.py** - Core RAG system with intelligent document processing and caching
2. **benchmark_rag.py** - Comprehensive benchmarking script with Bloom taxonomy evaluation
3. **pandascv.py** - Advanced analytics and visualization for benchmark results

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Benchmark System](#benchmark-system)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## ✨ Features

### Core RAG System (P-RAG.py)
- **Multi-format Document Processing**: Supports PDF, DOCX, TXT, MD, and JSON files
- **Intelligent Caching**: File-based caching with automatic change detection
- **Advanced Chunking**: Context-aware text segmentation with overlap
- **GPU Acceleration**: CUDA support for faster embedding generation
- **Batch Processing**: Efficient batch embedding creation with progress tracking
- **Interactive Chat Interface**: Real-time question-answering system

### Benchmarking System (benchmark_rag.py)
- **Bloom Taxonomy Integration**: Evaluates answers across 6 cognitive levels
- **Multiple Language Models**: Support for various open-source and commercial models
- **Comprehensive Metrics**: Accuracy, retrieval quality, and generation performance
- **Detailed Reporting**: JSON and HTML output with statistical analysis
- **Comparative Analysis**: Side-by-side model performance evaluation

### Analytics Dashboard (pandascv.py)
- **Data Visualization**: Heatmaps and performance charts
- **Cross-Model Comparison**: RAG vs Open Source model analysis
- **Statistical Insights**: Performance trends and accuracy distributions
- **Export Capabilities**: CSV and HTML report generation

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- Ollama (for local model inference)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd P-Rag
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Ollama (Optional)
If using local models via Ollama:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull deepseek-r1:8b
ollama pull phi4-mini-reasoning:latest
ollama pull qwen3:8b
ollama pull gemma3:4b-it-qat
```

### Step 4: Prepare Source Documents
```bash
mkdir Sources
# Add your PDF, DOCX, TXT, MD, or JSON files to the Sources folder
```

## 🚀 Quick Start

### Basic RAG Usage
```python
# Run the interactive Palestine RAG system
python P-RAG.py
```

The system will:
1. Load and process documents from the `Sources` folder
2. Create embeddings (cached for future use)
3. Start an interactive chat interface

Example interaction:
```
Palestine RAG Chatbot Ready!
You: What were the main causes of the Nakba?
Assistant: [Generated response based on your source documents]
```

### Running Benchmarks
```python
# Evaluate RAG system performance
python benchmark_rag.py
```

### Analyzing Results
```python
# Generate comprehensive analytics
python pandascv.py
```

## 🧩 Components

### P-RAG.py - Core RAG System

**Key Functions:**
- `load_documents_from_folder()` - Multi-format document processing
- `create_embeddings_batch()` - Efficient batch embedding generation
- `retrieve()` - Semantic similarity search
- `main()` - Interactive chat interface

**Supported File Types:**
- **PDF**: Automatic text extraction with error handling
- **DOCX**: Microsoft Word document processing
- **TXT/MD**: Plain text and Markdown files
- **JSON**: Structured data with intelligent text extraction

**Caching System:**
- Vector embeddings cached in `vector_db_cache.pkl`
- File metadata tracking in `source_metadata_cache.pkl`
- Automatic cache invalidation on file changes

### benchmark_rag.py - Evaluation Framework

**Evaluation Metrics:**
- **Accuracy**: Exact and semantic answer matching
- **Bloom Taxonomy**: Cognitive complexity assessment
- **Retrieval Quality**: Relevance and coverage metrics
- **Generation Performance**: Response time and coherence

**Supported Models:**
- Hugging Face Transformers
- Ollama local models
- Custom embedding models

**Output Formats:**
- Detailed JSON reports
- Statistical summaries
- Performance comparisons

### pandascv.py - Analytics & Visualization

**Features:**
- Performance heatmaps with seaborn styling
- Model type distinction (RAG vs Open Source)
- Statistical trend analysis
- Interactive HTML reports
- CSV data export

**Visualization Types:**
- Accuracy heatmaps by Bloom taxonomy level
- Model performance comparisons
- Distribution analysis
- Trend charts

## ⚙️ Configuration

### Model Configuration
```python
# Language Models (in P-RAG.py)
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
Deepseek = 'deepseek-r1:8b'
phi4 = 'phi4-mini-reasoning:latest'
qwen = 'qwen3:8b'

# Embedding Models
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast
embedding_model_name = 'BAAI/bge-large-en-v1.5'  # High quality
```

### Directory Structure
```
P-Rag/
├── Sources/                    # Source documents
├── detailed_Benchmark_Open Source/  # Open source results
├── detailed_Benchmark_RAG/     # RAG system results
├── P-RAG.py                   # Core RAG system
├── benchmark_rag.py           # Benchmarking script
├── pandascv.py                # Analytics dashboard
├── bloom_utils.py             # Bloom taxonomy utilities
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

### Cache Files
- `vector_db_cache.pkl` - Embedding vectors
- `source_metadata_cache.pkl` - File metadata
- `vector_db_cache_bge.pkl` - BGE model embeddings

## 💡 Usage Examples

### Example 1: Document Processing
```python
# Load specific documents
documents = load_specific_documents("./Sources", ["document1.pdf", "document2.docx"])

# Create embeddings
create_embeddings_batch(documents, batch_size=32)
```

### Example 2: Custom Queries
```python
# Retrieve relevant context
results = retrieve("What is the significance of olive trees in Palestinian culture?", top_n=5)

# Process results
for chunk, similarity, filename, metadata in results:
    print(f"Source: {filename}, Similarity: {similarity:.3f}")
    print(f"Content: {chunk[:200]}...")
```

### Example 3: Benchmark Evaluation
```python
# Initialize benchmarker
benchmarker = RAGBenchmarker()

# Load benchmark dataset
benchmark_data = load_benchmark("converted_benchmark_with_bloom.jsonl")

# Run evaluation
results = benchmarker.evaluate_benchmark(benchmark_data)
```

## 📊 Benchmark System

### Bloom Taxonomy Levels

The system evaluates responses across six cognitive levels:

1. **Remember** - Factual recall and basic information retrieval
2. **Understand** - Comprehension and explanation
3. **Apply** - Using knowledge in new situations
4. **Analyze** - Breaking down information and examining relationships
5. **Evaluate** - Making judgments and assessments
6. **Create** - Generating new ideas and solutions

### Evaluation Process

1. **Question Loading**: Import benchmark questions with Bloom classifications
2. **Context Retrieval**: Semantic search through document corpus
3. **Answer Generation**: LLM-based response generation
4. **Similarity Analysis**: Compare generated answers to gold standards
5. **Performance Metrics**: Calculate accuracy and quality scores
6. **Report Generation**: Create detailed analysis reports

### Sample Benchmark Output
```json
{
  "metadata": {
    "model": "deepseek-r1:8b",
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "total_questions": 50,
    "evaluation_date": "2025-07-05"
  },
  "analysis": {
    "overall_accuracy": 0.76,
    "correct_answers": 38,
    "bloom_level_stats": {
      "Remember": {"accuracy": 0.85, "total": 10},
      "Understand": {"accuracy": 0.78, "total": 12},
      "Apply": {"accuracy": 0.70, "total": 8}
    }
  }
}
```

## 📁 File Structure

```
P-Rag/
├── Core Files
│   ├── P-RAG.py              # Main RAG system
│   ├── benchmark_rag.py      # Evaluation framework
│   ├── pandascv.py           # Analytics dashboard
│   └── bloom_utils.py        # Bloom taxonomy utilities
├── Data Directories
│   ├── Sources/              # Source documents
│   ├── detailed_Benchmark_Open Source/
│   └── detailed_Benchmark_RAG/
├── Cache Files
│   ├── vector_db_cache.pkl
│   ├── source_metadata_cache.pkl
│   └── vector_db_cache_bge.pkl
├── Configuration
│   ├── requirements.txt      # Python dependencies
│   └── README.md            # Documentation
└── Output Files
    ├── *.json               # Benchmark results
    ├── *.csv                # Analytics data
    └── *.html               # Report visualizations
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
create_embeddings_batch(documents, batch_size=16)  # Instead of 32
```

**2. Ollama Connection Issues**
```bash
# Check Ollama status
ollama list
ollama serve  # If not running
```

**3. File Encoding Problems**
```python
# The system tries multiple encodings automatically:
encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
```

**4. Empty Vector Database**
- Ensure `Sources` folder contains supported file types
- Check file permissions and accessibility
- Verify files are not corrupted

**5. Slow Performance**
- Enable GPU acceleration: Install CUDA-compatible PyTorch
- Use smaller embedding models for faster processing
- Increase batch sizes if memory allows

### Performance Optimization

**Memory Usage:**
- Monitor GPU memory with `nvidia-smi`
- Adjust batch sizes based on available memory
- Use CPU fallback for large models

**Speed Improvements:**
- Cache embeddings for reuse
- Use quantized models when available
- Enable mixed precision training

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make your changes
5. Run tests and benchmarks
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for all functions
- Include error handling and logging

### Testing
```bash
# Run basic functionality tests
python P-RAG.py --test
python benchmark_rag.py --dry-run
```

## 📄 License

This project is part of the QCRI Summer Internship Program 2025. Please refer to your institution's guidelines for usage and distribution.

## 🙏 Acknowledgments

- Qatar Computing Research Institute (QCRI)
- Palestine research community
- Open-source embedding model providers
- Bloom taxonomy framework developers

---

**For support or questions, please contact the development team or create an issue in the repository.**

## 📚 Additional Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Ollama Model Library](https://ollama.ai/library)
- [Bloom's Taxonomy Reference](https://cft.vanderbilt.edu/guides-sub-pages/blooms-taxonomy/)
- [Palestine Academic Resources](https://palestine-studies.org/)