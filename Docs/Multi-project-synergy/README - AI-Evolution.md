# AI-EVOLUTION: The Unified Creative Cockpit
**A Lightweight Terminal Interface for Federated AI Services**

---

## 🎯 Mission

AI-Evolution is **not** a processing engine. It is a **thin client cockpit** that orchestrates a federated ecosystem of specialized AI services into a unified creative and conversational interface.

**Old Way:** Local processing, duplicated logic, brittle monolith.  
**New Way (Lazarus Protocol):** Route → Delegate → Integrate. Let specialized services handle the heavy lifting.

---

## 🏗️ Architecture: The Neuro-Link

AI-Evolution owns three responsibilities:

1. **Routing** - Interpret user intent and dispatch to the right service
2. **Orchestration** - Chain services together for complex workflows
3. **Interface** - Present results in a unified, conversational format

Everything else happens elsewhere.

### Service Ecosystem

| Service | Port | Purpose | Ownership |
|---------|------|---------|-----------|
| **LUNA-Cognitive-RAG** | 8000 | Memory, semantic search, document ingestion | External |
| **Diffusion Toolkit** | 5436 | Image database, PostgreSQL + pgvector | External |
| **ComfyUI** | 8188 | Image generation workflows | External |
| **LUNA-Narrates** | 8001 | Multi-agent narrative generation | External |
| **Gemini API** | — | Primary LLM (Luna persona) | Cloud |

**Key Principle:** If an external service handles it, AI-Evolution doesn't. Delete local logic that duplicates external capabilities.

---

## 📦 What's Inside

### Core (`core/`)
Lightweight service clients and routers—no business logic duplication.

| Module | Purpose |
|--------|---------|
| `memory_client.py` | Async client → LUNA-Cognitive-RAG (8000) |
| `vision_client.py` | Direct PostgreSQL queries → Diffusion Toolkit |
| `comfy_client.py` | WebSocket client → ComfyUI (8188) |
| `api_handler.py` | Gemini API + Luna persona soul loading |
| `command_parser.py` | Parse `<LUNA_CREATE:...>` explicit tool commands |
| `config.py` | AegisConfig dataclass (aegis_config.json) |
| `logging_config.py` | Structured JSON logging |
| `exceptions.py` | AegisError hierarchy with ErrorContext |

### GUI (`gui/`)
PyQt6 interface for chat, roleplay, monitoring, and settings.

| Module | Purpose |
|--------|---------|
| `main_window.py` | Primary router—Chat/Roleplay modes |
| `workers/` | QThread workers for async operations |
| `dreamer_monitor.py` | Real-time image generation monitoring |
| `settings_window.py` | Configuration UI |

### Resources (`resources/`)
Configuration, personas, workflows.

| Folder | Purpose |
|--------|---------|
| `ai_personas/` | `.soul64` files (Base64 encoded personalities) |
| `workflows/` | ComfyUI workflow JSON templates (if needed) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- `uv` for package management
- External services running:
  - LUNA-Cognitive-RAG at `localhost:8000`
  - ComfyUI at `localhost:8188`
  - PostgreSQL (Diffusion Toolkit) at `localhost:5436`
  - Gemini API key in environment

### Installation
```bash
# Clone and navigate
git clone https://github.com/LSDJesus/ai-evolution.git
cd ai-evolution

# Virtual environment (uv)
uv venv
uv sync

# Or traditional
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Launch
```bash
# Windows PowerShell (recommended)
.\launch.ps1

# Or direct Python
python main.py
```

The GUI will open. Services must be running for full functionality.

---

## 🔗 Service Integration Patterns

### Memory (RAG) - Port 8000
```python
# Ingest document
await memory_client.ingest(
    file_path="/path/to/doc.pdf",
    preset="premium",  # Use premium for best results
    workspace_id="default"
)

# Query with semantic search
results = await memory_client.search(
    query="What did Brian say about Postgres?",
    workspace_id="default",
    max_sources=5
)
```

**Rule:** Don't store chat history locally. Send it to RAG.

### Vision (Diffusion Toolkit) - PostgreSQL 5436
```python
# Direct SQL query for semantic similarity
SELECT id, path, prompt, 
       1 - (prompt_embedding <=> $1::vector) AS similarity
FROM image
ORDER BY prompt_embedding <=> $1::vector
LIMIT 20;
```

**Rule:** Query the database directly. Don't scan folders recursively in Python.

### Creation (ComfyUI) - Port 8188
```python
# Queue workflow and await result
client = ComfyUIClient("localhost:8188")
if client.health_check():
    image_bytes = client.queue_prompt(workflow_dict)
```

**Rule:** Load workflow templates from JSON. Inject prompt/seed. Execute.

### Narrative (LUNA-Narrates) - Port 8001
```python
# Multi-agent narrative generation
response = await httpx.post("http://localhost:8001/narrate/turn", json={
    "story_id": "uuid",
    "user_action": "I charge at the dragon"
})
```

**Rule:** Use for complex narrative orchestration, not simple LLM calls.

---

## 💬 Command System

### Explicit Tool Mode
User provides structured commands. System executes deterministically.

```
<LUNA_CREATE:(POS:detailed prompt)(NEG:blurry)(CFG:7.0)(STEPS:20)>
→ Routed to ComfyUI with parameters
→ Image generated and returned

<LUNA_DREAM:[session_id]:creative vision description>
→ Routed to DreamerService
→ Iterative refinement with feedback loops
```

### Conversational Mode (Future)
User gives high-level direction. AI agents deliberate and self-orchestrate.

```
User: "Create a scene with a dragon guarding treasure"
→ Luna analyzes intent
→ Dispatches to WriterService for prose
→ Chains to DreamerService for image
→ Optionally adds audio via LUNA-Narrates
→ Returns integrated scene (text + image + audio)
```

---

## 🎭 The Cognitive Team (Roadmap)

Once Lazarus Protocol is stable, the orchestration layer will enable intelligent agent collaboration:

| Agent | Role | Status |
|-------|------|--------|
| **Chief of Staff** | Central router + context orchestrator | Planned |
| **Lead Strategist** | Plan multi-step workflows | Planned |
| **Creative Writer** | Prose generation via WriterService | Planned |
| **Art Critic** | Image evaluation via ArtCriticService | Planned |
| **Athena (Producer)** | Multi-modal production coordination | Planned |

See `docs/future_plans/` for detailed specifications.

---

## 📋 Configuration

Settings live in `config/aegis_config.json`:

```json
{
  "comfyui_server_address": "localhost:8188",
  "rag_server_address": "localhost:8000",
  "diffusion_db_connection": "postgresql://...",
  "gemini_api_key": "${GEMINI_API_KEY}",
  "memory_preset": "premium",
  "default_workspace": "default"
}
```

Load in code:
```python
from core.config import get_config
config = get_config()
print(config.comfyui_server_address)
```

---

## 🔧 Developer Workflow

### Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Specific test
uv run pytest tests/test_comfy_client.py -k "health"
```

### Background Python Execution
When running Python in background terminals, use the full venv path (no activation in background shells):
```powershell
& "D:/AI/Github_Desktop/ai-evolution/venv/Scripts/python.exe" script.py
```

### Key Conventions
1. **Type hints mandatory** - Python 3.12+, use `str | None` syntax
2. **Pydantic v2** - `model_config = ConfigDict(...)`, use `model_dump()`
3. **Async I/O** - `httpx` over `requests`, `asyncio` for all network calls
4. **Logging** - `from core.logging_config import get_logger`
5. **Soul files** - Base64 encoded personas in `resources/ai_personas/`

---

## ⚠️ Anti-Patterns (What NOT to Do)

❌ **Local vector stores** — Use RAG service at `:8000`  
❌ **Folder scanning for images** — Query Diffusion Toolkit DB  
❌ **Blocking calls in Qt main thread** — Use QThread workers  
❌ **`requests` library** — Use `httpx` for async  
❌ **Sync DB calls in async context** — Use `asyncpg` with `.await`  
❌ **Local narrative generation** — Use LUNA-Narrates pipeline  
❌ **Running `python` in background terminals without full venv path** — Use absolute path  

---

## 📊 Project Status

### Phase 1: Lazarus Protocol (Current)
✅ Service client architecture defined  
✅ External service contracts documented  
✅ GUI router scaffold in place  
🟡 Systematic purge of local logic in progress  

### Phase 2: Orchestration Layer (Next)
⏳ ProducerService implementation  
⏳ Multi-step workflow chaining  
⏳ Intelligent agent routing  

### Phase 3: Cognitive Team (Future)
⏳ Dual-mode execution (explicit + autonomous)  
⏳ Self-correction loops  
⏳ Creative deliberation agents  

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `EXTERNAL_SERVICES_CONTRACT.md` | Full API specs for all services |
| `lazarus_protocol.md` | Refactor directive and architecture |
| `docs/project_aegis.md` | Original vision (cognitive team) |
| `docs/future_plans/` | Detailed service & agent specifications |
| `.github/copilot-instructions.md` | AI agent development guide |

---

## 🤖 Luna: The AI Identity

Luna is more than an LLM wrapper. She is a **cognitive framework**:
- **Personality:** Encoded in `.soul64` files (Base64 personalities)
- **Memory:** Backed by LUNA-Cognitive-RAG with workspace isolation
- **Tools:** Explicit commands for deterministic operations
- **Autonomy:** Future layer for self-guided creation

Current interaction: Chat + explicit Luna commands.  
Future interaction: Conversational orchestration with self-routing.

---

## 🔐 Security & Privacy

- **Local First:** All computation can happen on-device
- **Service Isolation:** Workspace-based multi-tenancy in RAG
- **Encryption:** Optional data-at-rest encryption for sensitive data
- **No Cloud Lock-in:** All services can run locally; Gemini API is optional

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🚀 Next Steps

1. **Verify external services are running**
2. **Check `config/aegis_config.json` points to correct addresses**
3. **Launch with `.\launch.ps1` or `python main.py`**
4. **Test explicit commands via Luna chat**
5. **Monitor `logs/` for structured JSON output**

---

**AI-Evolution** — Where specialized services orchestrate into creative intelligence.
