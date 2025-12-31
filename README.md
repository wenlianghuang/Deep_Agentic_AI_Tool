# Deep Agentic AI Tool

A comprehensive deep research agent system with RAG (Retrieval-Augmented Generation) capabilities, built with LangGraph and featuring local MLX model support. This system provides intelligent research assistance with stock queries, web search, PDF knowledge base retrieval, and email generation capabilities.

## 🚀 Features

### Core Capabilities

- **🔍 Deep Research Agent**: Intelligent multi-step research planning and execution
  - Automatic task decomposition based on query type
  - Smart tool selection (stocks, web search, PDF knowledge base)
  - Real-time progress tracking and note-taking
  - Comprehensive final report generation

- **📊 Stock Information Query**: Real-time stock data retrieval
  - Company financial data and operational status
  - Market cap, P/E ratio, revenue growth
  - Business summaries and analysis

- **🌐 Web Search**: Internet search for latest news and information
  - Powered by Tavily Search API
  - Retrieves up-to-date information from the web

- **📚 PDF Knowledge Base**: RAG system for document retrieval
  - Vector-based semantic search
  - Currently includes "Tree of Thoughts" research paper
  - Extensible to other PDF documents

- **📧 Email Agent**: AI-powered email generation and sending
  - Automatic email draft generation from prompts
  - Gmail API integration (avoids spam classification)
  - Support for both Chinese and English emails
  - Editable drafts before sending

### Technical Highlights

- **Local MLX Models**: Privacy-preserving local inference using Qwen2.5-Coder-7B-Instruct-4bit
- **Groq API Fallback**: Automatic fallback to Groq API when local model is unavailable
- **LangGraph Orchestration**: Sophisticated agent workflow management
- **Gradio Web Interface**: Modern, user-friendly web UI with real-time updates
- **Modular Architecture**: Clean, extensible codebase structure

## 📋 Prerequisites

- Python >= 3.13
- macOS (for MLX support) or Linux
- Google Cloud account (for Gmail API - optional, only needed for email features)
- Tavily API key (for web search - optional, can be configured in `.env`)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Deep_Agentic_AI_Tool
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Set up environment variables** (create a `.env` file in the root directory):
   ```env
   # Optional: Groq API (for faster inference)
   GROQ_API_KEY=your_groq_api_key_here
   
   # Optional: Tavily API (for web search)
   TAVILY_API_KEY=your_tavily_api_key_here
   
   # Optional: Gmail API credentials
   GMAIL_CREDENTIALS_FILE=credentials.json
   GMAIL_TOKEN_FILE=token.json
   ```

4. **Prepare PDF data** (optional):
   - Place your PDF files in the `data/` directory
   - The system currently uses `data/Tree_of_Thoughts.pdf` by default
   - You can modify the path in `deep_agent_rag/config.py`

5. **Set up Gmail API** (optional, for email features):
   - Follow the instructions in `GMAIL_API_SETUP.md`
   - Download OAuth2 credentials and save as `credentials.json` in the root directory

## 🎯 Usage

### Starting the Application

Run the main application:

```bash
python Deep_Agent_Gradio_RAG_localLLM_main.py
```

The Gradio interface will be available at:
- Local: `http://localhost:7860`
- Network: `http://0.0.0.0:7860`

### Using the Deep Research Agent

1. Navigate to the **"🔍 Deep Research Agent"** tab
2. Enter your research question in the input box
3. Click **"🔍 開始研究"** (Start Research)
4. Watch real-time updates:
   - **Current Status**: Shows which agent node is executing
   - **Task List**: Displays the research plan and progress
   - **Research Notes**: Real-time notes from the research process
   - **Final Report**: Comprehensive research report (streamed sentence by sentence)

#### Example Queries

- **Academic/Theory Questions**:
  - "Explain Tree of Thoughts and deeply compare it with Chain of Thought"
  - "Analyze the advantages and disadvantages of the Tree of Thoughts method"

- **Stock-Related Questions**:
  - "Compare Microsoft (MSFT) and Google (GOOGL) in the AI field"
  - "Query Apple (AAPL) financial status and recent developments"

- **Combined Questions**:
  - "Compare Microsoft (MSFT) and Google (GOOGL) in AI, incorporating Tree of Thoughts methodology"

### Using the Email Tool

1. Navigate to the **"📧 Email Tool"** tab
2. Enter an email prompt (e.g., "Write a thank you letter")
3. Enter the recipient's email address
4. Click **"📝 生成郵件草稿"** (Generate Email Draft)
5. Review and edit the generated subject and body
6. Click **"📧 發送郵件"** (Send Email) to send

#### Example Email Prompts

- "Write a thank you letter for help with the project"
- "Invite someone to next week's product launch"
- "Ask about project progress and provide updates"

## 🏗️ Architecture

### Project Structure

```
Deep_Agentic_AI_Tool/
├── deep_agent_rag/          # Main package
│   ├── agents/              # Agent nodes
│   │   ├── planner.py      # Task planning node
│   │   ├── researcher.py   # Research execution node
│   │   ├── note_taker.py   # Note-taking node
│   │   ├── reporter.py     # Final report generation
│   │   ├── email_agent.py  # Email generation agent
│   │   └── state.py        # State definition
│   ├── graph/              # LangGraph orchestration
│   │   └── agent_graph.py  # Graph construction and routing
│   ├── models/             # Model wrappers
│   │   └── mlx_chat_model.py  # MLX model integration
│   ├── rag/                # RAG system
│   │   └── rag_system.py   # PDF loading and retrieval
│   ├── tools/              # Tool definitions
│   │   ├── agent_tools.py  # Stock, web search, PDF tools
│   │   └── email_tool.py   # Email sending tool
│   ├── ui/                 # User interface
│   │   └── gradio_interface.py  # Gradio web UI
│   ├── utils/              # Utilities
│   │   └── llm_utils.py    # LLM instance management
│   └── config.py           # Configuration
├── data/                   # Data directory
│   └── Tree_of_Thoughts.pdf
├── Deep_Agent_Gradio_RAG_localLLM_main.py  # Main entry point
├── main.py                 # Simple entry point
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

### Agent Workflow

The system uses a multi-agent workflow orchestrated by LangGraph:

1. **Planner Node**: Analyzes the query and decomposes it into research tasks
   - Detects query type (academic, stock-related, general)
   - Generates appropriate task list based on query type
   - Avoids unnecessary tool calls

2. **Research Agent Node**: Executes research tasks
   - Dynamically selects and calls appropriate tools
   - Can call multiple tools in sequence
   - Maintains conversation context

3. **Tool Node**: Executes tool calls
   - Stock query tool
   - Web search tool
   - PDF knowledge base tool

4. **Note-Taking Node**: Consolidates research findings
   - Summarizes tool results
   - Adds to research notes

5. **Final Report Node**: Generates comprehensive report
   - Synthesizes all research notes
   - Creates structured final report

### LLM Configuration

The system supports multiple LLM backends with automatic fallback:

1. **Primary**: Groq API (fast, requires API key)
   - Model: `llama-3.3-70b-versatile`
   - Automatically used if `GROQ_API_KEY` is set

2. **Fallback**: Local MLX Model (privacy-preserving, no API key needed)
   - Model: `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`
   - Automatically used when Groq API is unavailable or quota exhausted

The system automatically switches between backends based on availability.

## ⚙️ Configuration

Key configuration options in `deep_agent_rag/config.py`:

### MLX Model Settings
```python
MLX_MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
MLX_MAX_TOKENS = 2048
MLX_TEMPERATURE = 0.7
```

### RAG Settings
```python
PDF_PATH = "./data/Tree_of_Thoughts.pdf"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 3
```

### Agent Settings
```python
MAX_ITERATIONS = 5
MAX_RESEARCH_ITERATIONS = 20
```

### Email Settings
```python
EMAIL_SENDER = "wenliangmatt@gmail.com"  # Change to your email
GMAIL_CREDENTIALS_FILE = "credentials.json"
GMAIL_TOKEN_FILE = "token.json"
```

## 🔧 Development

### Adding New Tools

1. Create a new tool function in `deep_agent_rag/tools/agent_tools.py`:
   ```python
   @tool
   def my_new_tool(param: str) -> str:
       """Tool description for the LLM."""
       # Implementation
       return result
   ```

2. Add the tool to `get_tools_list()` in the same file

3. The agent will automatically discover and use the new tool

### Modifying Agent Logic

- **Planning logic**: Edit `deep_agent_rag/agents/planner.py`
- **Research logic**: Edit `deep_agent_rag/agents/researcher.py`
- **Report generation**: Edit `deep_agent_rag/agents/reporter.py`

### Customizing UI

Edit `deep_agent_rag/ui/gradio_interface.py` to modify the web interface.

## 📦 Dependencies

Key dependencies (see `pyproject.toml` for complete list):

- **LangChain**: Agent framework and tool integration
- **LangGraph**: Agent orchestration and workflow management
- **MLX/MLX-LM**: Local model inference (Apple Silicon optimized)
- **Gradio**: Web interface
- **ChromaDB**: Vector database for RAG
- **Tavily**: Web search API
- **yfinance**: Stock data retrieval
- **Google API Client**: Gmail API integration

## 🐛 Troubleshooting

### MLX Model Issues

- **Model not loading**: Ensure you have sufficient disk space and memory
- **Slow inference**: This is normal for local models. Consider using Groq API for faster results

### Groq API Issues

- **Quota exhausted**: The system automatically falls back to local MLX model
- **API errors**: Check your `GROQ_API_KEY` in `.env` file

### RAG System Issues

- **PDF not found**: Ensure the PDF file exists at the path specified in `config.py`
- **Embedding model errors**: The system will attempt to re-download the model if cache is corrupted

### Gmail API Issues

- **Authorization errors**: Delete `token.json` and re-authorize
- **Credentials not found**: Ensure `credentials.json` is in the project root
- See `GMAIL_API_SETUP.md` for detailed setup instructions

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]

## 🙏 Acknowledgments

- **LangChain & LangGraph**: For the excellent agent framework
- **MLX Team**: For efficient local model inference
- **Qwen Team**: For the Qwen2.5 model
- **Jina AI**: For the embedding model

---

**Note**: This system is designed to work primarily on macOS with Apple Silicon for optimal MLX performance. Linux support is available but may have different performance characteristics.

