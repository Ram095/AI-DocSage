# AI Data Analyst 📊

A Streamlit-based AI Data Analyst application that allows users to upload documents and chat with them using OpenAI models.

## Features

- **Document Support**: Upload and analyze PDF, TXT, DOCX, and CSV files
- **Secure API Key Handling**: API keys stored only in memory, never persisted
- **RAG-powered Chat**: Ask questions and get answers based on document content
- **Source Citations**: See which parts of the document were used to generate answers
- **Hallucination Prevention**: Strict prompting ensures AI only answers from document context

## Project Structure

```
AI-Data-Analyst/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── utils/
    ├── __init__.py
    ├── document_processor.py  # Document extraction utilities
    ├── embeddings.py          # Text chunking & embedding utilities
    └── chat.py               # Chat response utilities
```

## Installation

1. **Clone or download the project**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key** (optional):
   ```bash
   # Option 1: Environment variable
   export OPENAI_API_KEY='your-api-key-here'  # Linux/Mac
   set OPENAI_API_KEY=your-api-key-here       # Windows CMD
   $env:OPENAI_API_KEY='your-api-key-here'    # Windows PowerShell
   
   # Option 2: Enter in the app sidebar (more secure for shared environments)
   ```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

1. **Configure API Key**: Enter your OpenAI API key in the sidebar (or set via environment variable)
2. **Upload Document**: Click "Upload a document" and select a supported file
3. **Start Chatting**: Ask questions about your document in the chat interface
4. **View Sources**: Expand "Sources used" to see which document sections were referenced

## Supported File Types

| Format | Extension | Library Used |
|--------|-----------|--------------|
| PDF    | .pdf      | PyPDF2       |
| Text   | .txt      | Built-in     |
| Word   | .docx     | python-docx  |
| CSV    | .csv      | pandas       |

## Configuration

Key parameters can be adjusted in the utility files:

- **Chunk size**: `utils/embeddings.py` - `DEFAULT_CHUNK_SIZE` (default: 600 tokens)
- **Similarity threshold**: `utils/embeddings.py` - `similarity_threshold` in `similarity_search()` (default: 0.3)
- **Top-K results**: Passed to `similarity_search()` (default: 3)
- **Max file size**: `utils/document_processor.py` - `MAX_FILE_SIZE` (default: 10 MB)

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Upload    │────▶│   Extract    │────▶│    Chunk    │
│  Document   │     │    Text      │     │    Text     │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   OpenAI    │◀────│   Similar    │◀────│   Create    │
│    Chat     │     │   Search     │     │  Embeddings │
└─────────────┘     └──────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│  Response   │
│  + Sources  │
└─────────────┘
```

## Vector Database Integration

The codebase is designed to easily integrate with vector databases. See the comments in `utils/embeddings.py` for integration points.

Recommended options:
- **Pinecone**: Managed vector database
- **Chroma**: Local/embedded vector database
- **Weaviate**: Open-source vector search

## Security Notes

- API keys are stored only in `st.session_state` (memory)
- Keys are never logged or persisted to disk
- Document content is not stored permanently
- System prompts prevent prompt injection attacks

## Troubleshooting

**"Invalid API key"**: Ensure your OpenAI API key is correct and has available credits

**"No text extracted"**: The document may be image-based (scanned PDF). Try a text-based PDF

**"Rate limit exceeded"**: Wait a moment and try again, or upgrade your OpenAI plan

**Large files taking too long**: Reduce the file size or use a more powerful machine

## License

MIT License - Feel free to use and modify for your projects.
