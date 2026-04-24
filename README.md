# SmartDoc AI - Intelligent Document Q&A System (Backend)

Hệ thống RAG (Retrieval-Augmented Generation) chatbot hỗ trợ truy vấn tài liệu thông minh, xây dựng với FastAPI và LangChain. Hỗ trợ hai pipeline RAG: **PaCRAG** (Parent-Child chunking + Redis Vector Store) và **GraphRAG** (Knowledge Graph + Neo4j).

---

## Mục lục

- [Tech Stack](#tech-stack)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Chạy server](#chạy-server)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [API Endpoints](#api-endpoints)
- [Chạy tests](#chạy-tests)
- [Giấy phép](#giấy-phép)

---

## Tech Stack

| Thành phần | Công nghệ |
|---|---|
| Framework | FastAPI + Uvicorn |
| AI / LLM | LangChain + Ollama (Qwen2.5:7b) |
| Vector Store | Redis (RedisVL) |
| Graph Database | Neo4j |
| Conversation History | SQLite |
| Embedding | nomic-embed-text |
| Document Processing | PyMuPDF, Unstructured (OCR) |
| Language | Python 3.12 |

### 🚀 Key Features

- **Hybrid Search**: Vector Search + BM25 Full-Text Search với RRF Fusion
- **Parent-Child Chunking**: Parent (1024 tokens) cho context, Child (128 tokens) cho search
- **LLM Re-ranking**: Cải thiện retrieval accuracy
- **Streaming Response**: Real-time token streaming
- **Conversation History**: Context-aware responses

---

## Yêu cầu hệ thống

- **Python 3.12+**
- **[Ollama](https://ollama.ai)** đang chạy local
- **Redis** (port 6380) - Vector store
- **Neo4j** (port 7687) - Graph database

### Pull model Ollama

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text:latest
```

---

## Cài đặt

### 1. Clone và tạo virtual environment

```bash
git clone <repository-url>
cd AI_HelpDesk_Backend

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Cài đặt dependencies

```bash
pip install -r packages.sh
```

### 3. Cấu hình môi trường

Sao chép và chỉnh sửa file config:

```bash
cp config_env/config.yaml config_env/config.local.yaml
```

Chỉnh sửa các giá trị trong `config_env/config.yaml` (xem phần [Cấu hình](#cấu-hình)).

---

## Cấu hình

Chỉnh sửa `config_env/config.yaml`:

```yaml
# Redis Vector Store
redis:
  url: redis://localhost:6380

# LLM Provider
llm:
  provider: ollama
  ollama:
    host: http://localhost:11434
    model: qwen2.5:7b
    embed: nomic-embed-text:latest

# Neo4j Graph Database
neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: <your_password>

# Parent-Child Chunking
llm:
  splitter:
    PaC:
      parent_chunk_size: 1024      # Large context chunks
      parent_chunk_overlap: 50
      child_chunk_size: 128        # Search-optimized chunks
      child_chunk_overlap: 50

# GraphRAG
graph_rag:
  chunk_size: 800
  chunk_overlap: 100
  top_k: 20
  graph_depth: 3
```

---

## Chạy server

### Windows (PowerShell)

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

### Linux / macOS

```bash
export PYTHONPATH=src
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

### Dùng Granian (hiệu suất cao hơn)

```bash
PYTHONPATH=src granian --interface asgi --port 8080 src.main:app
```

Server chạy tại: `http://localhost:8080`  
API docs (Scalar): `http://localhost:8080/scalar`

---

## Cấu trúc thư mục

```
AI_HelpDesk_Backend/
├── src/
│   ├── main.py                          # Entry point
│   ├── Features/
│   │   ├── LangChainAPI/
│   │   │   ├── LangChainController.py   # REST API endpoints
│   │   │   ├── LangChainFacade.py       # PaCRAG + GraphRAG setup
│   │   │   ├── RAG/
│   │   │   │   ├── PaCRAG.py            # Parent-Child RAG
│   │   │   │   ├── GraphRAG.py          # Graph RAG
│   │   │   │   ├── Loader.py            # Document loader
│   │   │   │   ├── Process.py           # Chunking strategies
│   │   │   │   ├── Retriever.py         # Hybrid search
│   │   │   │   └── LLMReranker.py       # Re-ranking
│   │   │   └── persistence/
│   │   │       ├── RedisVSRepository.py # Redis operations
│   │   │       └── Neo4JStore.py        # Neo4j operations
│   │   └── RealTimeAPI/                 # WebSocket support
│   └── SharedKernel/                    # Base classes, config
├── config_env/
│   └── config.yaml                      # Main configuration
├── specs/data/                          # SQLite databases
├── evaluation/                          # Benchmark scripts
└── static/                              # File uploads
```

---

## API Endpoints

Base URL: `http://localhost:8080/api/v1`

### PaCRAG (Parent-Child RAG)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/langchain/load_document_pdf_PaC` | Upload PDF/DOCX |
| `POST` | `/langchain/retrieve_document` | Streaming chat |
| `DELETE` | `/langchain/delete_document` | Xóa document |
| `GET` | `/langchain/chat_history/{session_id}` | Lịch sử hội thoại |

### GraphRAG

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/langchain/build-graph` | Build knowledge graph |
| `POST` | `/langchain/graph/query` | Query GraphRAG |
| `DELETE` | `/langchain/graph/{source}` | Xóa graph |

### Compare (PaCRAG vs GraphRAG)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/langchain/compare/upload` | Upload vào cả 2 RAG |
| `POST` | `/langchain/compare/query` | Query song song |

---

## Chạy tests

```bash
python -m pytest src/tests/ -v
```

## Benchmark

Chạy evaluation trên dataset 12 câu hỏi:

```bash
PYTHONPATH=src python evaluation/scripts/run_benchmark.py
```

Kết quả lưu trong `evaluation/results/`

---

---

## Giấy phép

Dự án được phát hành theo giấy phép **GNU General Public License v3.0 (GPL-3.0)**.
