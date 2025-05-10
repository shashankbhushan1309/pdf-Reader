import os
import streamlit as st
import tempfile
from pypdf import PdfReader
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import tiktoken
from dotenv import load_dotenv
import uuid
import time
import pickle
import os.path

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini API with the developer's key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize encoding for token counting
encoding = tiktoken.get_encoding("cl100k_base")

# Set up page configuration
st.set_page_config(
    page_title="PDF Genius - Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2563EB;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #9CA3AF;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E5E7EB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #10B981;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #F59E0B;
    }
    .source-box {
        background-color: #E5E7EB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2563EB;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
        border-color: #3B82F6;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .document-item {
        padding: 0.5rem;
        border-bottom: 1px solid #9CA3AF;
    }
</style>
""", unsafe_allow_html=True)


class SimpleVectorStore:
    """Simple in-memory vector store without requiring ChromaDB."""
    
    def __init__(self, storage_dir="./vector_store"):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Try to load existing data
        self.load_data()
    
    def add_texts(self, texts, metadatas, ids=None):
        """Add texts to the vector store with metadata."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
        # Process in smaller batches to avoid rate limiting
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            # Generate embeddings for this batch
            batch_embeddings = []
            for text in batch_texts:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)
            
            # Add to our collections
            self.documents.extend(batch_texts)
            self.embeddings.extend(batch_embeddings)
            self.metadatas.extend(batch_metadatas)
            self.ids.extend(batch_ids)
            
            time.sleep(0.5)  # Pause to avoid API rate limits
        
        # Save updated data
        self.save_data()
        
        return ids
    
    def get_embedding(self, text):
        """Get embedding for a text using Gemini."""
        try:
            # Use Gemini's embedding API
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 768  # Standard embedding dimension
    
    def similarity_search(self, query, top_k=5, filter_docs=None):
        """Search for documents similar to the query text."""
        if not self.embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding])
        
        # Convert list of embeddings to numpy array for efficient computation
        embeddings_array = np.array(self.embeddings)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, embeddings_array)[0]
        
        # Filter by document if specified
        filtered_indices = list(range(len(similarities)))
        if filter_docs:
            filtered_indices = [
                i for i, metadata in enumerate(self.metadatas) 
                if metadata.get("source") in filter_docs
            ]
        
        # Get similarities only for filtered documents
        filtered_similarities = [(i, similarities[i]) for i in filtered_indices]
        
        # Sort by similarity (descending)
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
        top_results = filtered_similarities[:top_k]
        
        # Format results
        results = []
        for idx, similarity in top_results:
            results.append({
                "text": self.documents[idx],
                "document": self.metadatas[idx].get("source", "Unknown"),
                "page": self.metadatas[idx].get("page", "Unknown"),
                "chunk_id": self.ids[idx],
                "similarity": float(similarity)
            })
        
        return results
    
    def clear(self):
        """Clear all data."""
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
        # Remove saved data file
        data_file = os.path.join(self.storage_dir, "vector_data.pkl")
        if os.path.exists(data_file):
            os.remove(data_file)
    
    def save_data(self):
        """Save data to disk."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "metadatas": self.metadatas,
            "ids": self.ids
        }
        
        with open(os.path.join(self.storage_dir, "vector_data.pkl"), "wb") as f:
            pickle.dump(data, f)
    
    def load_data(self):
        """Load data from disk if available."""
        data_file = os.path.join(self.storage_dir, "vector_data.pkl")
        if os.path.exists(data_file):
            try:
                with open(data_file, "rb") as f:
                    data = pickle.load(f)
                
                self.documents = data.get("documents", [])
                self.embeddings = data.get("embeddings", [])
                self.metadatas = data.get("metadatas", [])
                self.ids = data.get("ids", [])
                
                print(f"Loaded {len(self.documents)} documents from storage")
            except Exception as e:
                print(f"Error loading vector data: {e}")


class DocumentProcessor:
    """Class to process PDF documents for question answering."""
    
    def __init__(self):
        self.documents = {}
        self.vector_store = SimpleVectorStore()
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    def extract_text_from_pdf(self, file, filename):
        """Extract text from a PDF file with page numbers."""
        reader = PdfReader(file)
        full_text = ""
        page_texts = []
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_texts.append({"page": i+1, "text": page_text})
                full_text += f"[Page {i+1}]: {page_text}\n\n"
        
        self.documents[filename] = {
            "full_text": full_text,
            "pages": page_texts,
            "page_count": len(reader.pages)
        }
        return self.documents[filename]
    
    def count_tokens(self, text):
        """Count the number of tokens in a text string."""
        return len(encoding.encode(text))
    
    def chunk_text(self, text, doc_id, chunk_size=None, chunk_overlap=None):
        """Chunk text into smaller pieces with intelligent boundaries."""
        if chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap:
            self.chunk_overlap = chunk_overlap
            
        # Extract page number pattern
        page_numbers = {}
        for match in re.finditer(r'\[Page (\d+)\]:', text):
            page_numbers[match.start()] = int(match.group(1))
        
        # Try to split on paragraph boundaries first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_position = 0
        current_page = 1
        
        for para in paragraphs:
            para_with_newlines = para + "\n\n"
            new_position = current_position + len(para_with_newlines)
            
            # Update current page if needed
            for pos, page in sorted(page_numbers.items()):
                if current_position <= pos < new_position:
                    current_page = page
                    break
            
            # If adding this paragraph exceeds chunk size and we already have content
            if self.count_tokens(current_chunk + para_with_newlines) > self.chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk, 
                    "doc_id": doc_id,
                    "page": current_page
                })
                
                # Start new chunk with overlap
                overlap_point = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_point:] + para_with_newlines
            else:
                current_chunk += para_with_newlines
                
            current_position = new_position
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append({
                "text": current_chunk, 
                "doc_id": doc_id,
                "page": current_page
            })
            
        # If paragraphs are too large, we might need to further split chunks
        final_chunks = []
        for chunk in chunks:
            if self.count_tokens(chunk["text"]) > self.chunk_size:
                # Split by sentences instead
                sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
                current_chunk = ""
                
                for sentence in sentences:
                    if self.count_tokens(current_chunk + sentence) > self.chunk_size and current_chunk:
                        final_chunks.append({
                            "text": current_chunk, 
                            "doc_id": chunk["doc_id"],
                            "page": chunk["page"]
                        })
                        # Start new chunk with overlap
                        overlap_point = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_point:] + sentence + " "
                    else:
                        current_chunk += sentence + " "
                
                if current_chunk:
                    final_chunks.append({
                        "text": current_chunk, 
                        "doc_id": chunk["doc_id"],
                        "page": chunk["page"]
                    })
            else:
                final_chunks.append(chunk)
                
        return final_chunks
    
    def process_documents(self, files):
        """Process multiple documents and store in vector database."""
        self.documents = {}
        stored_chunks = 0
        
        for file in files:
            filename = file.name
            doc_info = self.extract_text_from_pdf(file, filename)
            chunks = self.chunk_text(doc_info["full_text"], filename)
            
            # Prepare data for vector store
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [{"source": chunk["doc_id"], "page": chunk["page"]} for chunk in chunks]
            ids = [f"{filename}-{i}" for i in range(len(chunks))]
            
            # Add to vector store
            self.vector_store.add_texts(texts, metadatas, ids)
            stored_chunks += len(chunks)
            
        return {
            "document_count": len(self.documents),
            "chunk_count": stored_chunks
        }
    
    def clear_documents(self):
        """Clear all documents and vector store data."""
        self.documents = {}
        self.vector_store.clear()


class QASystem:
    """Class to handle question answering based on document context."""
    
    def __init__(self, processor):
        self.processor = processor
        
    def get_gemini_response(self, system_prompt, user_prompt):
        """Get response from Gemini API."""
        try:
            # Try with the latest API structure
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(
                [
                    {"role": "user", "parts": [system_prompt]},
                    {"role": "model", "parts": ["I understand. I'll follow these guidelines."]},
                    {"role": "user", "parts": [user_prompt]}
                ],
                generation_config={"temperature": 0.2, "max_output_tokens": 2048}
            )
            return response.text
        except Exception as e:
            print(f"Error with primary model: {e}")
            try:
                # Fallback to another model
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(
                    user_prompt,
                    generation_config={"temperature": 0.2}
                )
                return response.text
            except Exception as e2:
                print(f"Error with fallback model: {e2}")
                return "I encountered an error while processing your question. Please try again with a different question."
    
    def answer_question(self, question, top_k=5, filter_docs=None):
        """Answer a question based on the uploaded documents."""
        # Search for relevant chunks
        relevant_chunks = self.processor.vector_store.similarity_search(
            query=question, 
            top_k=top_k,
            filter_docs=filter_docs
        )
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "sources": []
            }
        
        # Prepare context from relevant chunks
        context = "\n\n".join([
            f"Document: {chunk['document']} (Page {chunk['page']})\nContent: {chunk['text']}" 
            for chunk in relevant_chunks
        ])
        
        # Generate system prompt
        system_prompt = """You are an expert document analysis assistant that answers questions based on provided document information. 
        Follow these rules strictly:
        1. Only use the information provided in the context to answer questions.
        2. If the answer isn't in the context, say "I couldn't find that information in the documents."
        3. Cite the specific document sources (filenames and page numbers) in your answer using footnote-style citations like [Document Name, Page X].
        4. Be concise but comprehensive in your answers.
        5. Format your answer using markdown for better readability.
        6. Use bullet points for lists and structured information where appropriate.
        7. If the answer spans multiple documents or pages, synthesize the information and cite all relevant sources.
        8. Do not invent or assume information not present in the provided context."""
        
        # Generate user prompt
        user_prompt = f"""
        CONTEXT INFORMATION:
        {context}
        
        QUESTION:
        {question}
        
        Please provide an answer based only on the information in the documents. Include citations to the relevant document sources.
        """
        
        answer = self.get_gemini_response(system_prompt, user_prompt)
        
        return {
            "answer": answer,
            "sources": relevant_chunks
        }


def display_welcome():
    """Display welcome message and instructions."""
    st.markdown('<h1 class="main-header">📚 PDF Genius - Document Q&A System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>Welcome to PDF Genius!</h3>
        <p>Upload your PDF documents and ask questions about their content. The system will use AI to find and provide answers based on the information in your documents.</p>
        
        <h4>How it works:</h4>
        <ol>
            <li>Upload one or more PDF documents</li>
            <li>Wait for the system to process your documents</li>
            <li>Ask questions about the content of your documents</li>
            <li>Get AI-powered answers with citations to the source material</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>Features:</h3>
        <ul>
            <li>📄 Multi-document support</li>
            <li>🔍 Semantic search</li>
            <li>📊 Source verification</li>
            <li>🧠 AI-powered answers</li>
            <li>📱 Mobile friendly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
        st.session_state.qa_system = QASystem(st.session_state.processor)
        st.session_state.documents_loaded = False
        st.session_state.document_names = []
        st.session_state.processing = False
        st.session_state.chat_history = []


def display_sidebar(processor):
    """Display and handle sidebar components."""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<h2>📁 Document Manager</h2>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload one or more PDF documents to ask questions about"
        )
        
        if uploaded_files:
            if not st.session_state.processing:
                col1, col2 = st.columns([1, 1])
                with col1:
                    process_button = st.button("📥 Process Documents", use_container_width=True)
                with col2:
                    clear_button = st.button("🗑️ Clear All", use_container_width=True)
                
                if process_button:
                    st.session_state.processing = True
                    with st.spinner("Processing documents..."):
                        try:
                            result = processor.process_documents(uploaded_files)
                            st.session_state.documents_loaded = True
                            st.session_state.document_names = list(processor.documents.keys())
                            st.session_state.processing = False
                            st.markdown(f"""
                            <div class="success-box">
                                ✅ Successfully processed {result['document_count']} documents into {result['chunk_count']} chunks.
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.session_state.processing = False
                            st.error(f"Error processing documents: {str(e)}")
                
                if clear_button:
                    st.session_state.documents_loaded = False
                    st.session_state.document_names = []
                    st.session_state.chat_history = []
                    processor.clear_documents()
                    st.experimental_rerun()
        
        # Display document list and settings when documents are loaded
        if st.session_state.documents_loaded:
            st.markdown('<h3>📄 Your Documents</h3>', unsafe_allow_html=True)
            
            for doc in st.session_state.document_names:
                doc_info = processor.documents.get(doc, {})
                page_count = doc_info.get("page_count", 0)
                st.markdown(f"""
                <div class="document-item">
                    <b>{doc}</b> ({page_count} pages)
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<h3>⚙️ Search Settings</h3>', unsafe_allow_html=True)
            
            top_k = st.slider(
                "Result chunks to consider", 
                min_value=1, 
                max_value=10, 
                value=5,
                help="Higher values find more information but may be slower"
            )
            st.session_state.top_k = top_k
            
            st.markdown('<h4>Filter by Document</h4>', unsafe_allow_html=True)
            doc_filters = {}
            for doc in st.session_state.document_names:
                doc_filters[doc] = st.checkbox(doc, value=True)
            
            st.session_state.selected_docs = [doc for doc, checked in doc_filters.items() if checked]
        
        st.markdown('</div>', unsafe_allow_html=True)


def display_qa_interface():
    """Display the question answering interface."""
    if st.session_state.documents_loaded:
        st.markdown('<h2 class="sub-header">💬 Ask Questions About Your Documents</h2>', unsafe_allow_html=True)
        
        question = st.text_input(
            "Enter your question",
            placeholder="What does the document say about...",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🔍 Ask", use_container_width=True)
        with col2:
            clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
        
        if clear_chat:
            st.session_state.chat_history = []
            st.experimental_rerun()
        
        if ask_button and question:
            with st.spinner("Finding answer..."):
                try:
                    filter_docs = st.session_state.selected_docs if hasattr(st.session_state, 'selected_docs') else None
                    result = st.session_state.qa_system.answer_question(
                        question, 
                        top_k=st.session_state.top_k if hasattr(st.session_state, 'top_k') else 5, 
                        filter_docs=filter_docs
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"]
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display chat history
        if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
            for i, entry in enumerate(st.session_state.chat_history):
                # Question
                st.markdown(f"""
                <div style="background-color: #F3F4F6; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                    <b>You:</b> {entry["question"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Answer
                st.markdown(f"""
                <div style="background-color: #EFF6FF; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                    <b>PDF Genius:</b>
                    <div>{entry["answer"]}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Sources
                with st.expander("📄 View Source Chunks"):
                    for j, source in enumerate(entry["sources"]):
                        similarity_percentage = round(source["similarity"] * 100)
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Source {j+1}: {source["document"]} (Page {source["page"]}) - {similarity_percentage}% relevance</b>
                            <pre style="white-space: pre-wrap; font-size: 0.85rem;">{source["text"][:500]}{'...' if len(source["text"]) > 500 else ''}</pre>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("Please upload and process PDF documents using the sidebar to begin asking questions.")


def main():
    # Initialize session state
    initialize_session_state()
    
    # Display welcome message
    display_welcome()
    
    # Display sidebar
    display_sidebar(st.session_state.processor)
    
    # Display QA interface
    display_qa_interface()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #E5E7EB; opacity: 0.7;">
        PDF Genius - Document Q&A System
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()