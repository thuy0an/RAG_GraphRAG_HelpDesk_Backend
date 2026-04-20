# SmartDoc AI - Intelligent Document Q&A System (Backend)

Hệ thống RAG (Retrieval-Augmented Generation) chatbot hỗ trợ truy vấn tài liệu thông minh, xây dựng với FastAPI và LangChain. Hỗ trợ hai pipeline RAG song song: **PaCRAG** (Parent-Child chunking + Redis) và **GraphRAG** (Lexical Graph + Neo4j + FAISS).

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
| Framework | FastAPI + Uvicorn / Granian |
| AI / LLM | LangChain, Ollama (Qwen2.5), Mistral API |
| Vector Store | Redis (PaCRAG) + FAISS (GraphRAG) |
| Graph DB | Neo4j |
| Relational DB | MySQL (aiomysql) |
| Conversation History | SQLite (aiosqlite) |
| Embedding | nomic-embed-text (Ollama) / codestral-embed (Mistral) |
| Language | Python 3.12 |

---

## Yêu cầu hệ thống

- Python 3.12+
- [Ollama](https://ollama.ai) đang chạy local
- Redis (port 6380 mặc định)
- Neo4j (port 7687 mặc định)
- MySQL (port 3306 mặc định)

### Pull model Ollama

```bash
ollama pull qwen2.5:3b
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

> Một số package cần thêm (PDF, DOCX processing):
> ```bash
> pip install pymupdf docx2txt faiss-cpu aiosqlite
> ```

### 3. Cấu hình môi trường

Sao chép và chỉnh sửa file config:

```bash
cp config_env/config.yaml config_env/config.local.yaml
```

Chỉnh sửa các giá trị trong `config_env/config.yaml` (xem phần [Cấu hình](#cấu-hình)).

---

## Cấu hình

Tất cả cấu hình nằm trong `config_env/config.yaml`.

### Các giá trị cần thay đổi

```yaml
database:
  mysql:
    url: mysql+aiomysql://root:<password>@localhost:3306/AI_HelpDesk

redis:
  url: redis://localhost:6380

llm:
  provider: ollama          # hoặc "mistral"
  ollama:
    host: http://localhost:11434
    model: qwen2.5:3b
    embed: nomic-embed-text:latest

neo4j:
  uri: bolt://localhost:7687
  user: neo4j
  password: <password>

conversational_rag:
  conversation_history_limit: 5   # 0 = tắt lịch sử hội thoại
```

### Chunk strategy (PaCRAG)

```yaml
llm:
  splitter:
    PaC:
      parent_chunk_size: 2048
      parent_chunk_overlap: 400
      child_chunk_size: 512
      child_chunk_overlap: 100
```

### GraphRAG

```yaml
graph_rag:
  chunk_size: 800
  chunk_overlap: 100
  top_k: 16
  graph_depth: 3
  faiss_index_dir: specs/data/graph_rag/faiss_index
  label_prefix: GR
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
│   ├── Domain/
│   │   ├── history_entities.py          # ConversationHistory schema
│   │   └── compare_entities.py          # CompareRun schema
│   ├── Features/
│   │   ├── LangChainAPI/
│   │   │   ├── LangChainController.py   # Tất cả RAG endpoints
│   │   │   ├── LangChainFacade.py       # Facade khởi tạo PaCRAG + GraphRAG
│   │   │   ├── LangChainDTO.py          # Request/Response models
│   │   │   ├── prompt.py                # Prompt templates + history formatting
│   │   │   ├── RAG/
│   │   │   │   ├── BaseRAG.py           # Abstract base class
│   │   │   │   ├── PaCRAG.py            # Parent-Child RAG (Redis)
│   │   │   │   ├── GraphRAG.py          # Graph RAG (Neo4j + FAISS)
│   │   │   │   ├── GraphRAGInternal.py  # Lexical graph pipeline
│   │   │   │   ├── Loader.py            # PDF / DOCX / TXT loader
│   │   │   │   ├── Process.py           # Chunking pipeline
│   │   │   │   ├── Retriever.py         # Hybrid retriever (BM25 + vector)
│   │   │   │   ├── LLMReranker.py       # LLM-based re-ranking
│   │   │   │   └── ConfidenceScorer.py  # Self-evaluation scoring
│   │   │   ├── persistence/
│   │   │   │   ├── MemoryRepository.py  # Conversation history (SQLite)
│   │   │   │   ├── CompareRepository.py # Compare runs (SQLite)
│   │   │   │   ├── RedisVSRepository.py # Redis vector store
│   │   │   │   └── Neo4JStore.py        # Neo4j graph store
│   │   └── SharedKernelAPI/
│   │       └── SharedKernelController.py  # Health check endpoints
│   └── SharedKernel/
│       ├── base/                        # FastAPI app, DI, Logger, Metrics
│       ├── config/                      # LLMConfig, VectorStoreConfig
│       ├── exception/                   # APIException
│       ├── persistence/                 # PersistenceManager, Neo4jManager, Redis
│       ├── threading/                   # ThreadPoolManager
│       └── utils/                       # yamlenv, Utils
├── config_env/
│   ├── config.yaml                      # Cấu hình chính
│   ├── redis.yaml                       # Redis connection pool
│   └── redis_index.yaml                 # Redis vector index schema
├── specs/
│   └── data/
│       ├── chat_history.db              # SQLite — lịch sử hội thoại
│       ├── compare_runs.db              # SQLite — compare runs
│       └── graph_rag/faiss_index/       # FAISS index files
├── static/                              # File upload storage
├── src/tests/                           # Unit tests
│   ├── test_prompt.py
│   ├── test_process.py
│   ├── test_memory_repository.py
│   └── test_compare_entities.py
├── packages.sh                          # Danh sách pip packages
├── script.sh                            # Dev helper script (fzf menu)
├── pytest.ini                           # Pytest config
├── conftest.py                          # Pytest sys.path setup
└── pyrightconfig.json                   # Pyright type checking
```

---

## API Endpoints

Base URL: `http://localhost:8080/api/v1`

### PaCRAG (Parent-Child RAG)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/langchain/load_document_pdf_PaC` | Upload PDF/DOCX vào Redis vector store |
| `POST` | `/langchain/retrieve_document` | Streaming chat với PaCRAG |
| `DELETE` | `/langchain/delete_document` | Xóa document khỏi vector store |
| `DELETE` | `/langchain/clear_vector_store` | Xóa toàn bộ (hoặc theo source) |
| `GET` | `/langchain/chat_history/{session_id}` | Lấy lịch sử hội thoại |
| `DELETE` | `/langchain/clear_history/{session_id}` | Xóa lịch sử hội thoại |

### GraphRAG

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/langchain/build-graph` | Upload file và build lexical graph |
| `POST` | `/langchain/graph/query` | Query GraphRAG |
| `POST` | `/langchain/graph/multi-hop-query` | Multi-hop reasoning query |
| `DELETE` | `/langchain/graph/{source}` | Xóa graph của một document |
| `DELETE` | `/langchain/graph` | Xóa toàn bộ Neo4j graph |
| `GET` | `/langchain/graph/history/{session_id}` | Lịch sử GraphRAG |
| `DELETE` | `/langchain/graph/history/{session_id}` | Xóa lịch sử GraphRAG |

### Compare (PaCRAG vs GraphRAG)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/langchain/compare/upload` | Upload file vào cả 2 RAG |
| `POST` | `/langchain/compare/query` | Query song song cả 2 RAG |
| `GET` | `/langchain/compare/history/{session_id}` | Lịch sử compare runs |
| `DELETE` | `/langchain/compare/history/{run_id}` | Xóa một compare run |

### Conversation Turn Management

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/langchain/begin_turn` | Tạo turn mới, trả về `turn_id` |
| `POST` | `/langchain/save_turn` | Lưu Q&A hoàn chỉnh vào history |

### Health Check

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/shared_kernel/db` | Kiểm tra kết nối MySQL |
| `GET` | `/shared_kernel/neo4j` | Kiểm tra kết nối Neo4j |
| `GET` | `/shared_kernel/llm` | Kiểm tra LLM |
| `GET` | `/shared_kernel/embedding` | Kiểm tra embedding model |

---

## Chạy tests

Project có 56 unit tests không cần LLM, Redis, hay Neo4j.

### test_prompt.py + test_process.py (dùng Anaconda / môi trường có langchain)

```bash
python -m pytest src/tests/test_prompt.py src/tests/test_process.py -v
```

### test_memory_repository.py + test_compare_entities.py (dùng venv có sqlmodel)

```bash
venv\Scripts\python.exe -m pytest src/tests/test_memory_repository.py src/tests/test_compare_entities.py -v
```

### Kết quả mong đợi

```
56 passed
```

> **Lưu ý:** Hai test suite cần hai môi trường khác nhau vì `langchain` và `sqlmodel` chưa được cài chung vào một venv. Để chạy tất cả từ một môi trường, cài đủ packages vào cùng một venv.

---

## Giấy phép

Dự án được phát hành theo giấy phép **GNU General Public License v3.0 (GPL-3.0)**.

- File giấy phép cấp dự án: [../LICENSE](../LICENSE)
- Bản sao trong backend: [LICENSE](LICENSE)
