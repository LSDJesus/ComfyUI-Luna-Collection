# LUNA-Narrates
### AI-Powered Interactive Storytelling as a Service

> Premium narrative AI that delivers coherent, immersive stories without the hallucinations, contradictions, or astronomical costs. Built for scale.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-orange.svg)](#license)

---

## 🎯 The Service

LUNA-Narrates is a **multi-agent narrative AI platform** designed for SaaS deployment. Instead of throwing one overworked LLM at storytelling (and watching it forget your character's name by turn 20), we use specialized AI agents working in concert—like a professional writers' room.

### The Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LUNA-Narrates SaaS                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │  PREPROCESSOR   │───▶│  LEAD STRATEGIST │───▶│ CREATIVE WRITER│ │
│  │  (Self-Hosted)  │    │  (Cloud - Haiku) │    │  (Self-Hosted) │ │
│  │  Context Curation│   │  Narrative Plans │    │  Prose Gen     │ │
│  └─────────────────┘    └──────────────────┘    └────────────────┘ │
│           │                                              │          │
│           ▼                                              ▼          │
│  ┌─────────────────┐                           ┌────────────────┐  │
│  │    DREAMER      │                           │   PostgreSQL   │  │
│  │  (Self-Hosted)  │                           │  (Self-Hosted) │  │
│  │  Async Enrichment│                          │  All Story Data│  │
│  └─────────────────┘                           └────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
           ┌──────────────┐              ┌──────────────────┐
           │ Cloud Models │              │ Leased Compute   │
           │ (Tier 3)     │              │ (Burst Capacity) │
           │ • Claude Haiku│             │ • Custom Finetunes│
           │ • Gemini Flash│             │ • Dedicated GPUs  │
           │   Lite        │             │ • Scalable Workers│
           └──────────────┘              └──────────────────┘
```

### What Each Agent Does

| Agent | Hosting | Purpose | Model Strategy |
|-------|---------|---------|----------------|
| **Preprocessor** | Self-hosted | Context curation, semantic search, 96% compression | Local LM Studio / llama.cpp |
| **Lead Strategist** | Cloud API | High-level narrative planning, story beats | Claude Haiku / Gemini Flash Lite |
| **Creative Writer** | Self-hosted | Vivid prose generation, entity resolution | Local finetuned models |
| **Dreamer** | Self-hosted | Async enrichment, image generation, asset prep | Multi-worker pool |

---

## 💰 Cost Structure

### Per-Turn Breakdown

Using **Tier 3 cloud models** (Haiku, Gemini Flash Lite) for strategic planning only:

```
Preprocessor:  $0.000   (self-hosted inference)
Strategist:    $0.002   (Claude Haiku: ~1,500 in + 300 out)
Writer:        $0.000   (self-hosted inference)
Dreamer:       $0.000   (self-hosted async)
──────────────────────────────────────────────────────────
Total:         ~$0.002/turn
```

### Monthly Projections

| Usage Tier | Turns/Month | Cloud Cost | Infrastructure |
|------------|-------------|------------|----------------|
| Starter | 10,000 | ~$20 | Shared hosting |
| Growth | 100,000 | ~$200 | Dedicated server |
| Scale | 1,000,000 | ~$2,000 | Multi-server + burst |

**Compare to competitors:**
- AI Dungeon: $30/month (limited turns, content restrictions)
- NovelAI: $25/month (single model, memory issues)
- Direct GPT-4 API: $500+/month at scale (no specialization)

---

## 🏗️ Deployment Architecture

### Self-Hosted Components (Your Infrastructure)

**Core Services:**
- PostgreSQL database (all story data, user accounts, cost tracking)
- Preprocessor Agent (context curation, semantic search)
- Creative Writer Agent (prose generation)
- Dreamer Agent (async enrichment, image generation)
- FastAPI application server
- ChromaDB (semantic asset library)

**Inference Stack:**
- LM Studio or custom llama.cpp wrapper for local inference
- ComfyUI for image generation (optional)
- Redis for job queues (production)

### Cloud API Usage (Pay-per-call)

**Tier 3 Models Only** - Maximum cost efficiency:
- **Anthropic**: Claude Haiku ($0.25/MTok in, $1.25/MTok out)
- **Google**: Gemini Flash Lite (free tier available, then $0.075/MTok)

*We explicitly avoid expensive models (Sonnet, Opus, GPT-4) for production. Haiku and Flash Lite provide sufficient quality for strategic planning when combined with specialized local agents.*

### Burst Capacity (Leased Compute)

For traffic spikes and heavy load periods:

**Cloud GPU Providers:**
- RunPod, Vast.ai, Lambda Labs
- Deploy custom finetuned models on leased hardware
- Scale workers up/down based on demand
- Use your own model weights (no vendor lock-in)

**Use Cases:**
- Holiday traffic spikes
- Marketing campaign launches
- New feature rollouts
- A/B testing at scale

---

## 🖥️ Infrastructure Investment Path

### Phase 1: Minimal Viable SaaS (~$5,000)

**Cloud-First Approach:**
- VPS for API server + PostgreSQL ($100-300/month)
- All inference via cloud APIs (Haiku, Flash Lite)
- ~$0.005/turn all-in cost
- Validate product-market fit

### Phase 2: Hybrid Optimization (~$15,000)

**Add Local Inference:**
- Dedicated server with RTX 4090 or A6000
- Self-host Preprocessor + Writer
- Cloud for Strategist only
- ~$0.002/turn, 60% cost reduction

### Phase 3: Full Self-Hosted (~$50,000)

**Maximum Margin, Maximum Control:**

With the right infrastructure investment, you can achieve near-zero marginal cost per turn while maintaining premium quality.

#### The $50,000 Build

**Target Specs:**
- 384GB unified VRAM (4x 96GB GPUs)
- 512GB system RAM (8-channel DDR5)
- 128 PCIe 5.0 lanes for full GPU bandwidth
- Capable of serving thousands of concurrent users

**Bill of Materials:**

| Component | Selection | Cost Est. |
|-----------|-----------|-----------|
| **CPU** | AMD Threadripper Pro 9985WX (64-core, Zen 5) | $7,500 |
| **Motherboard** | ASUS Pro WS WRX90E-SAGE SE | $1,200 |
| **RAM** | 512GB DDR5-6400 ECC RDIMM (8x 64GB) | $3,500 |
| **GPUs** | 4x NVIDIA RTX PRO 6000 Blackwell (96GB each) | $28,000 |
| **Storage** | 2x 4TB PCIe 5.0 NVMe (RAID 1) | $1,200 |
| **PSUs** | 2x Corsair AX1600i (1600W Titanium) | $1,000 |
| **Case** | Corsair Obsidian 1000D | $500 |
| **Cooling** | Noctua NH-U14S TR5-SP6 + case fans | $300 |
| **Misc** | Cables, PDU, network cards | $500 |
| **Total** | | **~$44,000** |

**Why This Configuration:**
- **64-core over 96-core**: LLM inference is memory-bandwidth bound, not compute bound. The 9985WX fully saturates 8-channel memory while running at higher clocks.
- **300W "Max-Q" GPUs**: The 600W variants require liquid cooling or server chassis. Air-cooled blower cards stack safely.
- **384GB VRAM total**: Run multiple 70B+ models simultaneously, or one massive 405B model with room to spare.
- **8x RAM sticks**: Mandatory for full memory bandwidth (800GB/s+). Essential for AI inference speed.

#### Supporting Infrastructure Prerequisites

The $50k build assumes you have or will invest in:

- **100A 240V subpanel** - ~1.2kW idle, 2.5kW peak draw
- **Enterprise UPS** (e.g., Eaton 9355 30kVA) - Protects against power events
- **Dedicated cooling** (mini-split HVAC) - 2.5kW continuous heat output
- **2Gbps fiber** or equivalent - No network bottlenecks
- **Enterprise networking** - Proper switching, VLANs, firewall

If you already have datacenter-equivalent infrastructure in place, you're positioned for immediate deployment.

#### ROI Calculation

**Assumptions:**
- 1,000,000 turns/month at scale
- $0.002/turn with hybrid cloud ($2,000/month cloud costs)
- $0.0001/turn fully self-hosted (~$100/month electricity)

**Break-even:** ~2 years at 1M turns/month
**After break-even:** $23,000/year savings vs. hybrid approach

---

## 🔧 Technical Stack

### Production Inference

**Development Only:** LM Studio (excellent UX, model management)

**Production Options:**
- **vLLM** - High-throughput serving, tensor parallelism across GPUs
- **TensorRT-LLM** - NVIDIA-optimized, maximum performance on RTX/datacenter GPUs
- **llama.cpp server** - Lightweight, customizable, good for edge deployment
- **Triton Inference Server** - Enterprise-grade, multi-model serving
- **Text Generation Inference (TGI)** - HuggingFace's production server

### Database

**PostgreSQL 16+** with:
- pgvector extension (semantic search)
- Connection pooling (PgBouncer or built-in)
- Streaming replication for HA
- Point-in-time recovery

### Caching & Queues

- **Redis** - Session cache, job queues, rate limiting
- **ChromaDB** - Semantic asset library (warm tier)
- **Babel compression** - 90% storage reduction for cold tier

### API Layer

- **FastAPI** - Async Python, OpenAPI docs, SSE streaming
- **Uvicorn** - ASGI server with auto-reload
- **Nginx** - Reverse proxy, SSL termination, load balancing

---

## 📊 Scaling Strategy

### Horizontal Scaling

```
                    ┌─────────────┐
                    │   Nginx     │
                    │   (LB)      │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  API Pod 1  │ │  API Pod 2  │ │  API Pod 3  │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
    ┌──────────────────────────────────────────────┐
    │              Inference Workers               │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │
    │  │ GPU 1   │ │ GPU 2   │ │ GPU 3   │ ...    │
    │  │Preproc. │ │ Writer  │ │ Dreamer │        │
    │  └─────────┘ └─────────┘ └─────────┘        │
    └──────────────────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ PostgreSQL  │
                    │  Primary    │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ┌─────────────┐          ┌─────────────┐
       │  Replica 1  │          │  Replica 2  │
       │  (Read)     │          │  (Read)     │
       └─────────────┘          └─────────────┘
```

### Burst Handling

1. **Queue depth monitoring** - Scale workers when queue exceeds threshold
2. **Leased GPU activation** - Spin up RunPod/Vast.ai instances on demand
3. **Request prioritization** - Premium users get dedicated capacity
4. **Graceful degradation** - Fall back to faster/smaller models under load

---

## 🔐 Multi-Tenancy

### User Isolation

- Schema-per-tenant or row-level security
- Separate ChromaDB collections per user
- Rate limiting per API key
- Cost tracking per account

### Subscription Tiers

| Tier | Turns/Month | Features | Price Point |
|------|-------------|----------|-------------|
| Free | 100 | Basic generation, no images | $0 |
| Creator | 5,000 | Full features, standard priority | $9.99 |
| Author | 25,000 | Priority queue, custom worlds | $29.99 |
| Studio | Unlimited | Dedicated capacity, API access | $99.99+ |

---

## 📈 Metrics & Monitoring

### Key Performance Indicators

- **Turns per second** (throughput)
- **P50/P95/P99 latency** (user experience)
- **Cost per turn** (unit economics)
- **GPU utilization** (efficiency)
- **Queue depth** (capacity planning)
- **Error rate** (reliability)

### Observability Stack

- **Prometheus** - Metrics collection
- **Grafana** - Dashboards and alerting
- **Loki** - Log aggregation
- **Jaeger** - Distributed tracing

---

## 🚀 Getting Started (Development)

### Local Setup

```bash
# Clone repository
git clone https://github.com/LSDJesus/luna-narrates.git
cd luna-narrates

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: Add ANTHROPIC_API_KEY, DATABASE_URL

# Start PostgreSQL (Docker)
docker-compose -f _docker/docker-compose.yml up -d

# Apply migrations
Get-Content _docker\init_schema.sql | docker exec -i luna-narrates-db psql -U luna_dev -d luna_narrates

# Start development server
.\start_server.ps1
```

### Local Inference Setup (LM Studio)

1. Download [LM Studio](https://lmstudio.ai/)
2. Load recommended models:
   - Preprocessor: `gemma-2-9b-it` or similar 8-12B model
   - Writer: `mistral-nemo-instruct-2407` or finetuned variant
3. Start local server at `http://localhost:1234/v1/`
4. Configure in `.env`: `LMSTUDIO_HOST=http://localhost:1234/v1`

### API Documentation

Visit `http://localhost:8001/docs` for interactive Swagger UI.

---

## 🛣️ Roadmap

### Shipped ✅
- Multi-agent pipeline (Preprocessor → Strategist → Writer → Dreamer)
- PostgreSQL story persistence with arc summaries
- Cost tracking per agent/story
- ChromaDB semantic search
- ComfyUI image generation integration
- Basic authentication system

### In Progress 🔄
- WebUI (React/TypeScript)
- Model sync manager (multi-provider discovery)
- Advanced caching layer
- Subscription/billing integration

### Planned 📋
- Kubernetes deployment manifests
- Multi-region support
- Fine-tuning pipeline for custom models
- Mobile apps (iOS/Android)
- Third-party API (for other apps to use LUNA)

---

## 📄 License

**Commercial License** for SaaS deployment.

AGPL-3.0 available for self-hosted, non-commercial use.

Contact for enterprise licensing and support agreements.

---

## 👨‍💻 Author

**Brian Emmons**  
Building the future of AI-powered interactive storytelling.

---

*LUNA-Narrates — Premium narrative AI at scale.*
