"""
AI Data Analyst - Streamlit Application

A RAG-based document chat application that allows users to:
- Upload documents (PDF, TXT, DOCX, CSV)
- Chat with document content using OpenAI models
- Get context-aware responses with source citations

Author: AI Data Analyst Team
"""

import os
import streamlit as st
from openai import OpenAI

# Import utility modules
from utils.document_processor import extract_text, get_file_metadata, SUPPORTED_EXTENSIONS
from utils.embeddings import chunk_text, create_embeddings, create_single_embedding, similarity_search
from utils.chat import validate_api_key, chat_response, chat_response_stream, format_sources_display


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize all session state variables."""
    
    # API Key state
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    if 'api_key_status' not in st.session_state:
        st.session_state.api_key_status = ""
    
    # Document state
    if 'document_text' not in st.session_state:
        st.session_state.document_text = None
    if 'document_metadata' not in st.session_state:
        st.session_state.document_metadata = None
    if 'document_chunks' not in st.session_state:
        st.session_state.document_chunks = []
    if 'chunk_embeddings' not in st.session_state:
        st.session_state.chunk_embeddings = []
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    
    # Chat state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = None


def clear_document_state():
    """Clear all document-related state."""
    st.session_state.document_text = None
    st.session_state.document_metadata = None
    st.session_state.document_chunks = []
    st.session_state.chunk_embeddings = []
    st.session_state.document_processed = False
    st.session_state.messages = []


# =============================================================================
# API KEY HANDLING
# =============================================================================
def handle_api_key():
    """Handle OpenAI API key input and validation."""
    
    st.sidebar.header("🔑 API Configuration")
    
    # Check environment variable first
    env_key = os.environ.get('OPENAI_API_KEY')
    
    if env_key and not st.session_state.api_key:
        # Use environment variable
        is_valid, status = validate_api_key(env_key)
        if is_valid:
            st.session_state.api_key = env_key
            st.session_state.api_key_valid = True
            st.session_state.api_key_status = status
            st.session_state.openai_client = OpenAI(api_key=env_key)
            st.sidebar.success("Using API key from environment")
            return
    
    # Show current status
    if st.session_state.api_key_valid:
        st.sidebar.success(st.session_state.api_key_status)
        if st.sidebar.button("🔄 Change API Key"):
            st.session_state.api_key = None
            st.session_state.api_key_valid = False
            st.session_state.api_key_status = ""
            st.session_state.openai_client = None
            st.rerun()
    else:
        # Show input field
        st.sidebar.warning("⚠️ OpenAI API key required")
        
        api_key_input = st.sidebar.text_input(
            "Enter your OpenAI API key:",
            type="password",
            placeholder="sk-...",
            help="Your API key is stored only in memory and never persisted to disk."
        )
        
        if st.sidebar.button("Validate Key", type="primary"):
            if api_key_input:
                with st.sidebar:
                    with st.spinner("Validating..."):
                        is_valid, status = validate_api_key(api_key_input)
                
                st.session_state.api_key_status = status
                
                if is_valid:
                    st.session_state.api_key = api_key_input
                    st.session_state.api_key_valid = True
                    st.session_state.openai_client = OpenAI(api_key=api_key_input)
                    st.sidebar.success(status)
                    st.rerun()
                else:
                    st.sidebar.error(status)
            else:
                st.sidebar.error("Please enter an API key")


# =============================================================================
# DOCUMENT UPLOAD & PROCESSING
# =============================================================================
def handle_document_upload():
    """Handle document upload and processing."""
    
    st.sidebar.header("📄 Document Upload")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload a document",
        type=['pdf', 'txt', 'docx', 'csv'],
        help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
    )
    
    # Document status display
    if st.session_state.document_processed and st.session_state.document_metadata:
        meta = st.session_state.document_metadata
        st.sidebar.success("✅ Document processed")
        
        with st.sidebar.expander("📋 Document Info", expanded=True):
            st.write(f"**File:** {meta['filename']}")
            st.write(f"**Size:** {meta['size_formatted']}")
            st.write(f"**Words:** {meta.get('word_count', 'N/A'):,}")
            st.write(f"**Chunks:** {len(st.session_state.document_chunks)}")
        
        # Clear button
        if st.sidebar.button("🗑️ Clear Document", type="secondary"):
            clear_document_state()
            st.rerun()
        
        return True
    
    # Process new upload
    if uploaded_file is not None:
        # Check if this is a new file
        current_filename = st.session_state.document_metadata.get('filename') if st.session_state.document_metadata else None
        
        if current_filename != uploaded_file.name:
            # New file uploaded - process it
            if not st.session_state.api_key_valid:
                st.sidebar.error("Please configure your API key first")
                return False
            
            with st.sidebar:
                with st.spinner("Processing document..."):
                    try:
                        # Extract text
                        text, metadata = extract_text(uploaded_file)
                        st.session_state.document_text = text
                        st.session_state.document_metadata = metadata
                        
                        # Chunk text
                        chunks = chunk_text(text, method='paragraph', chunk_size=600)
                        st.session_state.document_chunks = chunks
                        
                        # Create embeddings
                        embeddings = create_embeddings(
                            st.session_state.openai_client,
                            chunks
                        )
                        st.session_state.chunk_embeddings = embeddings
                        
                        # Mark as processed
                        st.session_state.document_processed = True
                        
                        # Clear previous chat
                        st.session_state.messages = []
                        
                        st.success("✅ Document processed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        return False
    else:
        st.sidebar.info("Upload a document to get started")
    
    return st.session_state.document_processed


# =============================================================================
# CHAT INTERFACE
# =============================================================================
def display_chat_interface():
    """Display the main chat interface."""
    
    # Header
    st.title("📊 AI Data Analyst")
    st.caption("Chat with your documents using AI-powered analysis")
    
    # Check prerequisites
    if not st.session_state.api_key_valid:
        st.info("👈 Please configure your OpenAI API key in the sidebar to get started.")
        return
    
    if not st.session_state.document_processed:
        st.info("👈 Please upload a document in the sidebar to start chatting.")
        
        # Show helpful instructions
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            1. **Enter your OpenAI API key** in the sidebar
            2. **Upload a document** (PDF, TXT, DOCX, or CSV)
            3. **Ask questions** about your document
            4. Get **AI-powered answers** based on the document content
            
            **Supported file types:**
            - 📕 PDF files
            - 📝 Text files (.txt)
            - 📘 Word documents (.docx)
            - 📊 CSV files
            
            **Tips:**
            - Ask specific questions for better answers
            - The AI will only answer based on document content
            - Use the "Sources" section to verify answers
            """)
        return
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📚 Sources used"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['section']}** (Relevance: {source['relevance']})")
                        st.text(source['preview'])
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            try:
                # Create embedding for the question
                with st.spinner("Searching document..."):
                    query_embedding = create_single_embedding(
                        st.session_state.openai_client,
                        prompt
                    )
                    
                    # Find relevant chunks
                    relevant_chunks = similarity_search(
                        query_embedding,
                        st.session_state.chunk_embeddings,
                        st.session_state.document_chunks,
                        top_k=3,
                        similarity_threshold=0.3
                    )
                
                # Format sources for display
                formatted_sources = format_sources_display(relevant_chunks)
                
                # Stream the response
                response = st.write_stream(
                    chat_response_stream(
                        st.session_state.openai_client,
                        prompt,
                        relevant_chunks,
                        st.session_state.messages[:-1]  # Exclude current user message
                    )
                )
                
                # Show sources after streaming completes
                if formatted_sources:
                    with st.expander("📚 Sources used"):
                        for source in formatted_sources:
                            st.markdown(f"**{source['section']}** (Relevance: {source['relevance']})")
                            st.text(source['preview'])
                            st.divider()
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": formatted_sources
                })
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {error_msg}",
                    "sources": []
                })


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Sidebar components
    handle_api_key()
    st.sidebar.divider()
    handle_document_upload()
    
    # Footer in sidebar
    st.sidebar.divider()
    st.sidebar.caption("Built with ❤️ using Streamlit & OpenAI")
    
    # Main content area
    display_chat_interface()


if __name__ == "__main__":
    main()
