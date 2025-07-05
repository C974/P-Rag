import ollama
import os
import PyPDF2
from docx import Document
import re
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import sys
from tqdm import tqdm
import json  
from bloom_utils import bloom_levels, get_bloom_instruction


# Configuration
SOURCES_FOLDER = "./Sources"
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
Deepseek='deepseek-r1:8b'
phi4='phi4-mini-reasoning:latest'
qwen='qwen3:8b'
gemma='gemma3:4b-it-qat'
granite='granite3.3:latest'
llama='meta-llama/Llama-3.2-1B-Instruct'

# Use local embedding model with CUDA support
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast and efficient
embedding_model_name='BAAI/bge-large-en-v1.5'
# Alternative: 'all-mpnet-base-v2' for better quality

# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize embedding model with CUDA
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

# Vector database
VECTOR_DB = []
VECTOR_DB_FILE = "vector_db_cache.pkl"
METADATA_CACHE_FILE = "source_metadata_cache.pkl"

def get_source_files_metadata(folder_path: str) -> dict:
    """Get metadata of all source files for comparison"""
    if not os.path.exists(folder_path):
        return {}
    
    files_metadata = {}
    for filename in os.listdir(folder_path):
        if filename.startswith('~$'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
            
        file_ext = filename.lower().split('.')[-1]
        if file_ext in ['pdf', 'docx', 'txt', 'md', 'json']:  
            # Store filename, modification time, and file size
            stat = os.stat(file_path)
            files_metadata[filename] = {
                'mtime': stat.st_mtime,
                'size': stat.st_size
            }
    
    return files_metadata

def load_cached_metadata():
    """Load cached source files metadata"""
    if os.path.exists(METADATA_CACHE_FILE):
        try:
            with open(METADATA_CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cached metadata: {e}")
    return {}

def save_metadata_cache(metadata):
    """Save source files metadata to cache"""
    try:
        with open(METADATA_CACHE_FILE, 'wb') as f:
            pickle.dump(metadata, f)
    except Exception as e:
        print(f"Error saving metadata cache: {e}")

def find_new_or_modified_files(folder_path: str) -> List[str]:
    """Find new or modified files since last cache"""
    current_metadata = get_source_files_metadata(folder_path)
    cached_metadata = load_cached_metadata()
    
    new_or_modified_files = []
    
    for filename, current_info in current_metadata.items():
        if filename not in cached_metadata:
            # New file
            new_or_modified_files.append(filename)
            print(f"New file detected: {filename}")
        elif (cached_metadata[filename]['mtime'] != current_info['mtime'] or 
              cached_metadata[filename]['size'] != current_info['size']):
            # Modified file
            new_or_modified_files.append(filename)
            print(f"Modified file detected: {filename}")
    
    return new_or_modified_files

def load_cached_embeddings():
    """Load cached embeddings if available"""
    if os.path.exists(VECTOR_DB_FILE):
        try:
            with open(VECTOR_DB_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cached embeddings: {e}")
    return []

def save_embeddings_cache():
    """Save embeddings to cache"""
    try:
        with open(VECTOR_DB_FILE, 'wb') as f:
            pickle.dump(VECTOR_DB, f)
        print("Embeddings cached successfully")
    except Exception as e:
        print(f"Error saving embeddings cache: {e}")

def read_pdf(file_path: str) -> str:
    """Extract text from PDF file with better error handling"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Error reading page {page_num} from {file_path}: {e}")
                    continue
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""

def read_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def read_txt(file_path: str) -> str:
    """Extract text from TXT file with encoding detection"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    
    print(f"Could not decode {file_path} with any encoding")
    return ""

def read_json(file_path: str) -> str:
    """Extract text from JSON file with intelligent text extraction"""
    def extract_text_from_json(obj, path=""):
        """Recursively extract text from JSON object"""
        text_parts = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Extract key names as context
                if isinstance(value, (str, int, float)) and value:
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    text_parts.extend(extract_text_from_json(value, current_path))
                    
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                text_parts.extend(extract_text_from_json(item, current_path))
                
        elif isinstance(obj, (str, int, float)) and obj:
            text_parts.append(str(obj))
            
        return text_parts
    
    try:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    data = json.load(file)
                    
                # Extract all text from the JSON structure
                text_parts = extract_text_from_json(data)
                
                # Join with newlines for better readability
                return '\n'.join(text_parts)
                
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as e:
                print(f"Invalid JSON format in {file_path}: {e}")
                return ""
        
        print(f"Could not decode JSON file {file_path} with any encoding")
        return ""
        
    except Exception as e:
        print(f"Error reading JSON {file_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.,!?;:\-\'\"\u0600-\u06FF\u0750-\u077F]', ' ', text)  # Include Arabic characters
    
    # Remove very short lines (likely artifacts)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
    
    return '\n'.join(cleaned_lines)

def intelligent_chunk(text: str, chunk_size: int = 800, overlap: int = 150) -> List[Tuple[str, dict]]:
    """Create intelligent chunks optimized for embedding"""
    chunks = []
    
    if not text.strip():
        return chunks
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If paragraph is too long, split by sentences
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # If adding this sentence exceeds chunk size, save current chunk
                if current_size + len(sentence) > chunk_size and current_chunk:
                    chunks.append((current_chunk.strip(), {
                        'chunk_size': len(current_chunk),
                        'chunk_type': 'mixed'
                    }))
                    
                    # Start new chunk with overlap
                    if overlap > 0:
                        words = current_chunk.split()
                        overlap_text = ' '.join(words[-overlap//10:])
                        current_chunk = overlap_text + " " + sentence
                        current_size = len(current_chunk)
                    else:
                        current_chunk = sentence
                        current_size = len(sentence)
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_size += len(sentence)
        else:
            # If adding this paragraph exceeds chunk size, save current chunk
            if current_size + len(para) > chunk_size and current_chunk:
                chunks.append((current_chunk.strip(), {
                    'chunk_size': len(current_chunk),
                    'chunk_type': 'paragraph'
                }))
                
                # Start new chunk with overlap
                if overlap > 0:
                    words = current_chunk.split()
                    overlap_text = ' '.join(words[-overlap//10:])
                    current_chunk = overlap_text + " " + para
                    current_size = len(current_chunk)
                else:
                    current_chunk = para
                    current_size = len(para)
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_size += len(para)
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append((current_chunk.strip(), {
            'chunk_size': len(current_chunk),
            'chunk_type': 'final'
        }))
    
    return chunks

def load_specific_documents(folder_path: str, filenames: List[str]) -> List[Tuple[str, str, dict]]:
    """Load specific documents from folder and return chunks with metadata"""
    documents = []
    
    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):
            print(f"File not found: {filename}")
            continue
            
        print(f"Processing: {filename}")
        
        text = ""
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            text = read_pdf(file_path)
        elif file_ext == 'docx':
            text = read_docx(file_path)
        elif file_ext in ['txt', 'md']:
            text = read_txt(file_path)
        elif file_ext == 'json':  
            text = read_json(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        
        if not text.strip():
            print(f"No text extracted from: {filename}")
            continue
        
        text = clean_text(text)
        print(f"  -> Extracted {len(text)} characters")
        
        chunks = intelligent_chunk(text, chunk_size=800, overlap=150)
        
        for chunk_text, chunk_metadata in chunks:
            metadata = {
                'filename': filename,
                'file_type': file_ext,
                'file_size': os.path.getsize(file_path),
                **chunk_metadata
            }
            documents.append((chunk_text, filename, metadata))
        
        print(f"  -> Created {len(chunks)} chunks from {filename}")
    
    return documents

def load_documents_from_folder(folder_path: str) -> List[Tuple[str, str, dict]]:
    """Load all documents from folder and return chunks with metadata"""
    documents = []
    
    print(f"Looking for folder at: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist!")
        return documents
    
    files = os.listdir(folder_path)
    print(f"Found {len(files)} files in folder")
    
    for filename in files:
        if filename.startswith('~$'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):
            continue
            
        print(f"Processing: {filename}")
        
        text = ""
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            text = read_pdf(file_path)
        elif file_ext == 'docx':
            text = read_docx(file_path)
        elif file_ext in ['txt', 'md']:
            text = read_txt(file_path)
        elif file_ext == 'json':  # Added JSON support
            text = read_json(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        
        if not text.strip():
            print(f"No text extracted from: {filename}")
            continue
        
        text = clean_text(text)
        print(f"  -> Extracted {len(text)} characters")
        
        chunks = intelligent_chunk(text, chunk_size=800, overlap=150)
        
        for chunk_text, chunk_metadata in chunks:
            metadata = {
                'filename': filename,
                'file_type': file_ext,
                'file_size': os.path.getsize(file_path),
                **chunk_metadata
            }
            documents.append((chunk_text, filename, metadata))
        
        print(f"  -> Created {len(chunks)} chunks from {filename}")
    
    return documents

# ... (rest of the functions remain the same: create_embeddings_batch, cosine_similarity_batch, retrieve, safe_input, main)

def create_embeddings_batch(chunks_data: List[Tuple[str, str, dict]], batch_size: int = 32):
    """Create embeddings in batches for better GPU utilization with proper progress bar"""
    global VECTOR_DB
    
    texts = [chunk_text for chunk_text, _, _ in chunks_data]
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"Creating embeddings for {len(texts)} chunks in {total_batches} batches of {batch_size}")
    
    # Configure tqdm for VS Code terminal compatibility
    progress_bar = tqdm(
        range(0, len(texts), batch_size),
        desc="Creating embeddings",
        unit="batch",
        ncols=100,
        ascii=True,
        file=sys.stdout,
        leave=True
    )
    
    # Process in batches with progress bar
    for i in progress_bar:
        batch_texts = texts[i:i + batch_size]
        batch_data = chunks_data[i:i + batch_size]
        
        # Update progress bar description
        current_batch = i // batch_size + 1
        progress_bar.set_description(f"Processing batch {current_batch}/{total_batches}")
        
        try:
            # Create embeddings for the batch
            batch_embeddings = embedding_model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Add to vector database
            for j, embedding in enumerate(batch_embeddings):
                chunk_text, filename, metadata = batch_data[j]
                VECTOR_DB.append((chunk_text, embedding, filename, metadata))
                
        except Exception as e:
            print(f"\nError creating embeddings for batch {current_batch}: {e}")
            # Fall back to individual processing for this batch
            for chunk_text, filename, metadata in batch_data:
                try:
                    embedding = embedding_model.encode([chunk_text])[0]
                    VECTOR_DB.append((chunk_text, embedding, filename, metadata))
                except Exception as e2:
                    print(f"Error creating embedding for chunk from {filename}: {e2}")
    
    progress_bar.close()
    print(f"\nCompleted embedding creation for {len(VECTOR_DB)} chunks")

def cosine_similarity_batch(query_embedding, embeddings):
    """Vectorized cosine similarity calculation"""
    # Convert to numpy arrays
    query_embedding = np.array(query_embedding)
    embeddings = np.array(embeddings)
    
    # Calculate cosine similarity using vectorized operations
    dot_products = np.dot(embeddings, query_embedding)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    
    # Avoid division by zero
    similarities = np.where(norms != 0, dot_products / norms, 0.0)
    
    return similarities

def retrieve(query: str, top_n: int = 5) -> List[Tuple[str, float, str, dict]]:
    """Retrieve most relevant chunks using vectorized operations"""
    try:
        query_embedding = embedding_model.encode([query])[0]
    except Exception as e:
        print(f"Error creating query embedding: {e}")
        return []
    
    if not VECTOR_DB:
        return []
    
    # Extract embeddings and metadata
    embeddings = [item[1] for item in VECTOR_DB]
    
    # Calculate similarities using vectorized operations
    similarities = cosine_similarity_batch(query_embedding, embeddings)
    
    # Create results with metadata
    results = []
    for i, similarity in enumerate(similarities):
        chunk_text, _, filename, metadata = VECTOR_DB[i]
        results.append((chunk_text, float(similarity), filename, metadata))
    
    # Sort by similarity in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_n]

def safe_input(prompt: str) -> str:
    """Safe input function that handles terminal focus issues"""
    # Flush all outputs before asking for input
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        return input(prompt)
    except EOFError:
        # Handle case where input stream is not available
        print("\nInput stream not available. Please ensure terminal is focused.")
        return ""
    except KeyboardInterrupt:
        print("\nReceived interrupt signal.")
        raise

def main():
    global VECTOR_DB
    
    # Ensure proper terminal setup
    print("Palestine RAG System Starting...")
    print("=" * 50)
    
    # Check for new or modified files
    print("Checking for new or modified source files...")
    new_or_modified_files = find_new_or_modified_files(SOURCES_FOLDER)
    
    # Load cached embeddings
    print("Loading cached embeddings...")
    VECTOR_DB = load_cached_embeddings()
    
    if new_or_modified_files:
        print(f"Found {len(new_or_modified_files)} new or modified files:")
        for filename in new_or_modified_files:
            print(f"  - {filename}")
        
        # Remove old embeddings for modified files
        if VECTOR_DB:
            print("Removing old embeddings for modified files...")
            original_count = len(VECTOR_DB)
            VECTOR_DB = [item for item in VECTOR_DB if item[2] not in new_or_modified_files]
            removed_count = original_count - len(VECTOR_DB)
            if removed_count > 0:
                print(f"Removed {removed_count} old embeddings")
        
        # Load new/modified documents and create embeddings
        print("Loading new/modified documents...")
        new_documents = load_specific_documents(SOURCES_FOLDER, new_or_modified_files)
        
        if new_documents:
            print(f"Creating embeddings for {len(new_documents)} new chunks...")
            create_embeddings_batch(new_documents, batch_size=64)
            
            # Save updated cache
            save_embeddings_cache()
            
            # Update metadata cache
            current_metadata = get_source_files_metadata(SOURCES_FOLDER)
            save_metadata_cache(current_metadata)
            
            print(f"[OK] Added {len(new_documents)} new chunks to database")
        else:
            print("No valid content found in new/modified files")
    
    elif VECTOR_DB:
        print(f"[OK] No new files detected, using cached embeddings")
    else:
        # No cache and no files - load everything
        print("No cache found. Loading all documents...")
        dataset = load_documents_from_folder(SOURCES_FOLDER)
        
        if len(dataset) == 0:
            print("[ERROR] No documents found. Please add PDF, DOCX, TXT, or JSON files to the Sources folder.")
            return
        
        print(f'Loaded {len(dataset)} chunks from documents')
        create_embeddings_batch(dataset, batch_size=64)
        
        # Save cache
        save_embeddings_cache()
        current_metadata = get_source_files_metadata(SOURCES_FOLDER)
        save_metadata_cache(current_metadata)
    
    print(f'\n[OK] Vector database ready with {len(VECTOR_DB)} chunks')
    print(f'[OK] Using device: {device}')
    print("=" * 50)
    
    # Ensure terminal is ready for input
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Interactive chatbot
    print("\nPalestine RAG Chatbot Ready!")
    print("You can ask questions about Palestine based on your loaded documents.")
    print("Tips:")
    print("   - Make sure this terminal window is focused/clicked")
    print("   - Type 'exit', 'quit', or 'q' to stop")
    print("   - Press Ctrl+C to interrupt")
    
    while True:
        try:
            
            # Ask user to choose Bloom level
            print("\nSelect Bloom's level:")
            for key, val in bloom_levels.items():
                print(f"  {key}. {val}")

            selected_level = safe_input("Enter level number (1–6): ").strip()
            level_name = bloom_levels.get(selected_level, "Understand")
            if level_name in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
                citation_instruction = "Do not cite or reference specific documents or filenames."
            else:
                citation_instruction = "You may cite or reference specific documents or filenames if relevant."
            thinking_instruction = get_bloom_instruction(level_name)

            # Ask the actual question
            input_query = safe_input(f'\n[{level_name}] Ask your question about Palestine: ')

            
            if input_query.lower() in ['exit', 'quit', 'q', '']:
                print("Goodbye!")
                break
            
            if not input_query.strip():
                print("Please enter a question.")
                continue
            
            print(f"\nSearching for information about: '{input_query}'")
            
            # Retrieve relevant chunks
            retrieved_knowledge = retrieve(input_query, top_n=5)
            
            if not retrieved_knowledge:
                print("[NOTE] No relevant information was found in the documents. Proceeding without context.")
                context_chunks = []
                context_notice = "No supporting documents were retrieved. Please answer based on your own general understanding."
            else:
                context_chunks = [chunk for chunk, _, _, _ in retrieved_knowledge]
                context_notice = ""
            
            print('\nRetrieved knowledge:')
            for i, (chunk, similarity, filename, metadata) in enumerate(retrieved_knowledge, 1):
                print(f'{i}. (similarity: {similarity:.3f}) [{filename}]')
                print(f'   {chunk[:150]}...\n')
            
            # Create context for the LLM
            context_chunks = []
            for chunk, similarity, filename, metadata in retrieved_knowledge:
                context_chunks.append(f"[Source: {filename}] {chunk}")
            
            instruction_prompt = f'''You are a highly intelligent assistant with expertise in Palestine-related topics.
                                    {thinking_instruction}
                                    {context_notice}
                                    {citation_instruction}
                                    Do not cite or reference specific documents or filenames.
                                    Answer in your own words, using the context below only as guidance.

                                    Context:
                                    {chr(10).join(context_chunks)}
                                    '''
            
            print("Generating response...")
            
            try:
                stream = ollama.chat(
                    model=llama,
                    messages=[
                        {'role': 'system', 'content': instruction_prompt},
                        {'role': 'user', 'content': input_query},
                    ],
                    stream=True,
                )
                
                print('\nResponse:')
                print("-" * 50)
                for chunk in stream:
                    print(chunk['message']['content'], end='', flush=True)
                print("\n" + "-" * 50)
                
            except Exception as e:
                print(f"[ERROR] Error generating response: {e}")
                print("Make sure Ollama is running and the model is available.")
                
        except KeyboardInterrupt:
            print("\n\nExiting... Goodbye!")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            print("Please try again or restart the application.")

if __name__ == "__main__":
    main()