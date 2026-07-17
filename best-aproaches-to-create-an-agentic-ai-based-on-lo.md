# Best Approaches to Create an Agentic AI Based on Locally Hosted Models with Web-Based Chat UI

**Research Date:** July 17, 2026  
**Sources:** 7 research agents, 50+ GitHub repositories, official documentation

---

## Executive Summary

Building a locally hosted agentic AI with a web-based chat interface is well-supported by a mature open-source ecosystem as of mid-2026. The dominant pattern combines **Ollama** (for easy local model serving) or **llama.cpp / vLLM** (for production), an agentic framework such as **LangGraph** or **CrewAI** (for orchestrating multi-step tool use), and a web UI layer such as **Open WebUI** (plug-and-play) or a custom stack built with **FastAPI + Next.js / Vercel AI SDK** (for full control). The **Model Context Protocol (MCP)** has emerged as the universal standard for connecting tools to agents, and **Qwen3** and **Llama 3.x** families are the top-performing open models for agentic tool calling. For most teams, the fastest path to a production-quality local agent is: Ollama + Open WebUI for rapid prototyping, graduating to LangGraph + FastAPI + custom React UI for custom agent workflows.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Layer 1: Local LLM Hosting](#2-layer-1-local-llm-hosting)
3. [Layer 2: Recommended Models](#3-layer-2-recommended-models)
4. [Layer 3: Agentic Frameworks](#4-layer-3-agentic-frameworks)
5. [Layer 4: Web-Based Chat UI Options](#5-layer-4-web-based-chat-ui-options)
6. [Agentic Architecture Patterns](#6-agentic-architecture-patterns)
7. [Tool Use & Function Calling](#7-tool-use--function-calling)
8. [MCP: Model Context Protocol](#8-mcp-model-context-protocol)
9. [RAG: Retrieval-Augmented Generation](#9-rag-retrieval-augmented-generation)
10. [Memory Systems](#10-memory-systems)
11. [Streaming Architecture](#11-streaming-architecture)
12. [Complete Stack Recommendations](#12-complete-stack-recommendations)
13. [Production Deployment](#13-production-deployment)
14. [Key Repositories Summary](#14-key-repositories-summary)
15. [Confidence Assessment](#15-confidence-assessment)

---

## 1. System Architecture Overview

An agentic AI system with a web chat UI involves four layers: the model serving layer, the agentic orchestration layer, the backend API layer, and the frontend UI layer.

```mermaid
graph TD
    subgraph "Web Frontend"
        UI[React / SvelteKit / Next.js<br/>Chat UI]
        WS[WebSocket / SSE<br/>Streaming]
    end

    subgraph "Backend API Layer"
        API[FastAPI / Express<br/>Chat Endpoint]
        AUTH[Auth / Session<br/>JWT / OAuth]
    end

    subgraph "Agentic Orchestration"
        AGENT[Agent Framework<br/>LangGraph / CrewAI / smolagents]
        TOOLS[Tool Registry<br/>Web Search, Code, DB, RAG]
        MEMORY[Memory<br/>Vector DB + SQLite]
    end

    subgraph "Model Context Protocol"
        MCP[MCP Servers<br/>Filesystem, Git, DB, Web]
    end

    subgraph "LLM Serving Layer"
        OLLAMA[Ollama<br/>localhost:11434]
        LCPP[llama.cpp server<br/>localhost:8080]
        VLLM[vLLM<br/>localhost:8000]
    end

    subgraph "Models"
        M1[Qwen3:14b / Qwen3:8b]
        M2[Llama3.1:8b / Llama3.3:70b]
        M3[Hermes3:8b]
    end

    UI <-->|HTTP + SSE / WS| WS
    WS <--> API
    API --> AUTH
    API --> AGENT
    AGENT --> TOOLS
    AGENT --> MEMORY
    TOOLS --> MCP
    AGENT -->|OpenAI-compat API| OLLAMA
    AGENT -->|OpenAI-compat API| LCPP
    AGENT -->|OpenAI-compat API| VLLM
    OLLAMA --> M1
    OLLAMA --> M2
    OLLAMA --> M3
    LCPP --> M1
    VLLM --> M2
```

---

## 2. Layer 1: Local LLM Hosting

All major local LLM servers expose an **OpenAI-compatible API** (`/v1/chat/completions`), enabling drop-in replacement with frameworks designed for OpenAI.

### 2.1 Comparison Table

| Solution | Stars | Tech | API Port | Tool Calling | Best For |
|---|---|---|---|---|---|
| **Ollama** | ~100K | Go + llama.cpp | 11434 | ✅ Full (JSON + streaming) | Easiest setup, development, most integrations |
| **llama.cpp server** | ~80K | C++ | 8080 | ✅ Very mature (grammar-constrained) | Maximum performance, CPU support, fine-grained control |
| **vLLM** | ~50K | Python + PyTorch | 8000 | ✅ Production-grade (10+ model parsers) | High-throughput multi-user production serving |
| **LM Studio** | Closed | Electron + llama.cpp/MLX | 1234 | ✅ Via OpenAI-compat | Desktop users, Apple Silicon (MLX backend) |
| **LocalAI** | ~27K | Go | 8080 | ✅ OpenAI-compat | Multi-modal all-in-one local API |
| **Jan.ai** | Active | Node/Tauri + llama.cpp | 1337 | ✅ MCP integration | Privacy-first desktop app |
| **KoboldCpp** | Active | Python + llama.cpp | 5001 | ✅ + MCP | Creative writing, roleplay, multimodal single-binary |

### 2.2 Ollama (Recommended Default)

Ollama is the easiest path and has the largest ecosystem integration[^1]:

```bash
# Install (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run a model
ollama pull qwen3:14b
ollama serve  # starts API at http://localhost:11434

# Use with any OpenAI SDK — just change base_url
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(
    model="qwen3:14b",
    messages=[{"role": "user", "content": "Hello!"}],
    tools=[...]  # tool definitions work natively
)
```

**Tool calling with Ollama native API:**[^1]
```python
import ollama

response = ollama.chat(
    model='qwen3:14b',
    messages=[{'role': 'user', 'content': 'What is the weather in Tokyo?'}],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get current weather for a city',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string', 'description': 'City name'},
                },
                'required': ['city']
            }
        }
    }]
)
# response['message']['tool_calls'] contains the structured call
```

### 2.3 llama.cpp Server (Performance Option)

The underlying inference engine behind Ollama, exposed directly for fine-grained control[^2]:

```bash
# Start server with tool support
llama-server -m qwen3-14b-q4_k_m.gguf --port 8080 --jinja

# Multiple concurrent users
llama-server -m model.gguf -c 16384 -np 4  # 4 parallel slots

# Structured JSON output via grammar
llama-server -m model.gguf --grammar-file grammars/json.gbnf
```

**Model-specific tool call parsers:**[^2]

| Model Family | Format |
|---|---|
| Llama 3.1, 3.2, 3.3 | Native Llama JSON + `<function>` tags |
| Qwen2.5 / Hermes 3 | Hermes 2 Pro XML |
| Mistral Nemo | Mistral JSON |
| DeepSeek R1 | DeepSeek R1 (WIP) |

### 2.4 vLLM (Production Multi-User)

Best choice when serving many concurrent users[^3]:

```bash
# Install
pip install vllm

# Serve with tool calling enabled
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --port 8000

# Or Qwen3
vllm serve Qwen/Qwen3-14B \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
```

Key advantage: **PagedAttention** eliminates KV cache fragmentation, enabling 10–100× more concurrent requests than llama.cpp for multi-user workloads.[^3]

---

## 3. Layer 2: Recommended Models

### 3.1 Model Selection by Hardware

| GPU VRAM | Recommended Model | Ollama Tag | Quantization | Context Budget |
|---|---|---|---|---|
| **8GB** (RTX 3080/4070) | Qwen3 8B or Hermes3 8B | `qwen3:8b` | Q4_K_M | Up to 32K |
| **12GB** (RTX 3080 Ti) | Qwen3 8B or Mistral-Nemo 12B | `mistral-nemo:12b` | Q8_0 (8B) / Q4_K_M (12B) | Up to 64K |
| **16GB** (RTX 4080/3090) | Qwen3 14B | `qwen3:14b` | Q4_K_M | Up to 64K |
| **24GB** (RTX 4090) | Qwen3 14B Q8_0 or Qwen3 32B Q4_K_M | `qwen3:32b` | Q8_0 / Q4_K_M | Up to 128K |
| **48GB** (2×24GB / A6000) | Qwen3 32B Q8_0 or Llama 3.3 70B Q4_K_M | `llama3.3:70b` | Q8_0 | Up to 128K |
| **80GB+** (A100/H100) | Llama 3.3 70B Q8_0 or Qwen3 235B | `qwen3:235b` | Q8_0 | Full 128K |

### 3.2 Tier 1: Best for Agentic Use[^4]

| Model | Ollama Tag | Context | Tool Calling | Notes |
|---|---|---|---|---|
| **Qwen3:14b** | `qwen3:14b` | 128K | ⭐⭐⭐⭐⭐ Native FC + thinking | **Top choice 16GB GPU** |
| **Qwen3:8b** | `qwen3:8b` | 128K | ⭐⭐⭐⭐⭐ Native FC + thinking | **Top choice 8GB GPU** |
| **Hermes3:8b** | `hermes3:8b` | 8K | ⭐⭐⭐⭐⭐ Purpose-built tool calling | Best tool-call specialist |
| **Llama3.1:8b** | `llama3.1:8b` | 128K | ⭐⭐⭐⭐ Native FC | Most framework support |
| **Llama3.3:70b** | `llama3.3:70b` | 128K | ⭐⭐⭐⭐⭐ Native FC | Best quality, multi-GPU |
| **Qwen2.5:14b** | `qwen2.5:14b` | 128K | ⭐⭐⭐⭐⭐ Native FC | Battle-tested stable |

### 3.3 Key Model Notes

**Qwen3 (Alibaba, 2025)** — Overall top recommendation[^4]:
- Unique hybrid **thinking mode** (`/think` for deep reasoning, `/no_think` for fast dispatch)
- Leading performance on Berkeley Function-Calling Leaderboard among open models
- Supports 100+ languages
- MoE variants: `qwen3:30b` (activates 3B params), `qwen3:235b` (activates 22B params)

**Hermes3 (Nous Research)** — Tool-calling specialist[^4]:
> *"More powerful and reliable function calling and structured output capabilities... advanced agentic capabilities."*
- Based on Llama 3.1 fine-tuned specifically for tool use
- Highly reliable JSON/XML schema adherence

**Quantization rule for agentic use:**[^4]
- **Q4_K_M** = Best size/quality for most agent tasks; allows larger context
- **Q8_0** = Better at precise JSON/tool call formatting; use if VRAM permits
- **Never use Q2_K or Q3_K for agents** — too much degradation in instruction following

---

## 4. Layer 3: Agentic Frameworks

### 4.1 Framework Comparison

| Framework | Stars | Pattern | Local LLM | Best For |
|---|---|---|---|---|
| **LangGraph** | 37.5K | Stateful graph | ✅ Any OpenAI-compat | Production complex workflows, persistence |
| **LangChain** | 142K | Chains + LCEL | ✅ Any OpenAI-compat | Broad ecosystem, RAG, integrations |
| **CrewAI** | 55.7K | Role-based crews | ✅ via LiteLLM | Multi-agent delegation, role personas |
| **smolagents** | 28.4K | ReAct + CodeAgent | ✅ via LiteLLM/Transformers | Python-code agents, simplicity |
| **LlamaIndex** | 50.9K | Document/Workflow | ✅ Ollama native | RAG-first agentic, document processing |
| **Agno** | 41.2K | Structured agents | ✅ OpenAI-compat | Fast, lightweight, team support |
| **AutoGen** | 59.8K | Multi-agent chat | ✅ Ollama native | ⚠️ **Maintenance mode** — use MAF instead |
| **MS Agent Framework** | 12.2K | Enterprise | ✅ LiteLLM | Enterprise, .NET + Python, AutoGen successor |

### 4.2 LangGraph (Recommended for Production)

LangGraph models agents as **stateful directed graphs** — the most powerful approach for complex, stateful, long-running agents[^5]:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama

# 1. Define shared state with type-safe reducers
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # accumulates messages
    user_id: str                               # last-value semantics

# 2. Configure local LLM
llm = ChatOllama(model="qwen3:14b", temperature=0.0)

# 3. Build the agent graph
def call_model(state: AgentState) -> dict:
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # LLM decides: call tool or end
builder.add_edge("tools", "agent")  # loop back after tool use

# 4. Compile with persistence checkpointing
with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    app = builder.compile(checkpointer=saver)

# 5. Run with thread ID for conversation persistence
config = {"configurable": {"thread_id": "user-42-session-1"}}
result = app.invoke({"messages": [("user", "What files are in my project?")]}, config=config)
```

**Key LangGraph features:**[^5]
- **Cycles**: Agents loop back (tool call → LLM → tool call → ...) until done
- **Checkpointing**: Full conversation state persisted (SQLite dev, PostgreSQL prod)
- **Human-in-the-loop**: `interrupt()` to pause and await human approval
- **Subgraphs**: Agents as nodes in parent graphs (multi-agent hierarchies)
- **`astream_events`**: Stream every intermediate step (tokens, tool calls, thoughts) to the frontend
- **RetryPolicy / TimeoutPolicy**: Per-node resilience configuration

### 4.3 CrewAI (Recommended for Role-Based Multi-Agent)

CrewAI excels at multi-agent systems where different "experts" collaborate[^6]:

```python
from crewai import Agent, Task, Crew, Process, LLM

researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="Expert analyst known for insightful reports.",
    tools=[search_tool, scrape_tool],
    llm=LLM(model="ollama/qwen3:14b"),  # local model via LiteLLM
    memory=True,  # enables RAG memory
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content",
    llm=LLM(model="ollama/llama3.1:8b"),
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=True,
    embedder={"provider": "ollama", "config": {"model": "nomic-embed-text"}},  # local embeddings!
)

result = crew.kickoff(inputs={"topic": "AI agents 2026"})
```

**Local LLM support via LiteLLM:**[^6]
```python
# Ollama
llm = LLM(model="ollama/qwen3:14b", base_url="http://localhost:11434")

# llama.cpp server
llm = LLM(model="openai/local-model", base_url="http://localhost:8080/v1", api_key="none")

# vLLM
llm = LLM(model="openai/qwen3-14b", base_url="http://localhost:8000/v1", api_key="none")
```

### 4.4 smolagents (HuggingFace — Recommended for Simplicity)

smolagents is HuggingFace's lightweight agent framework with a key insight: **writing actions as Python code is 30% more efficient than JSON tool calls**[^7]:

```python
from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel, WebSearchTool

# Primary local path via LiteLLMModel
model = LiteLLMModel(
    model_id="ollama_chat/qwen3:14b",
    api_base="http://localhost:11434",
    api_key="ollama",
    num_ctx=32768,  # CRITICAL: Ollama default 2048 is too small for agents
    temperature=0.0,
)

# CodeAgent: actions as Python code (more powerful)
agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    additional_authorized_imports=['requests', 'bs4'],
    max_steps=20,
    planning_interval=5,  # re-plan every 5 steps
)
agent.run("Research the latest advances in local AI agents and write a summary")
```

**CodeAgent vs ToolCallingAgent:**[^7]

| Scenario | Use |
|---|---|
| Multi-step reasoning, complex planning | **CodeAgent** |
| Dynamic tool combinations, loops | **CodeAgent** |
| Simple API calls, web search | **ToolCallingAgent** |
| Safety-critical / untrusted environment | **ToolCallingAgent** |

**Multi-agent with managed_agents:**[^7]
```python
# Sub-agents become callable tools for the manager
web_agent = ToolCallingAgent(
    tools=[WebSearchTool()], model=model,
    name="web_search_agent",  # REQUIRED
    description="Runs web searches. Give it your query.",  # REQUIRED
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],  # injected as callable tools
)
```

---

## 5. Layer 4: Web-Based Chat UI Options

### 5.1 Option A: Plug-and-Play (No Custom Code)

#### Open WebUI (⭐ Recommended Out-of-the-Box)[^8]

**145K+ stars** — The most feature-complete, actively maintained web UI for local AI[^8]:

```bash
# Minimal setup: Ollama + Open WebUI
docker compose -f docker-compose.yaml up
```

```yaml
# docker-compose.yaml (from open-webui/open-webui)
services:
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama:/root/.ollama
    restart: unless-stopped

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    depends_on: [ollama]
    ports:
      - "3000:8080"
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    volumes:
      - open-webui:/app/backend/data
    restart: unless-stopped

volumes:
  ollama: {}
  open-webui: {}
```

**Tech stack**: SvelteKit v5 + TypeScript (frontend) + FastAPI + SQLAlchemy (backend)[^8]

**Built-in agentic features**:
- **Plugin system**: Tools, Filters, Actions, Pipes, Skills
- **MCP support** via MCPO bridge
- **RAG**: 9 vector DBs (ChromaDB, Qdrant, PGVector, Milvus, etc.), BM25+vector hybrid
- **Web search**: 20+ providers (SearXNG, Brave, Tavily, etc.)
- **Persistent memory** across conversations
- **20+ LLM provider support** (Ollama, LM Studio, vLLM, OpenAI, Anthropic, etc. simultaneously)
- RBAC, LDAP/SSO, SCIM 2.0, horizontal scaling with Redis

#### LibreChat (Full-Featured Alternative)[^8]

**40.8K stars** — Feature-rich with strong multi-provider support[^8]:
- **Agents marketplace** with sharing
- **MCP client** built-in (v0.7.6+)
- **Subagents** with isolated context windows
- **Code Interpreter** (Python, Node.js, Go, C++, Java, Rust, Fortran)
- **RAG** with file search
- MongoDB + Redis backend; Docker Compose deployment

#### AnythingLLM (No-Code Agent Builder)[^8]

**~45K stars** — Strong agentic features with no-code approach[^8]:
- No-code AI Agent builder with flow editor
- MCP-compatible (works with MCP servers)
- Multi-vector DB support (LanceDB, Chroma, Qdrant, Milvus, Pinecone, etc.)
- Scheduled tasks (cron with full agent capabilities)
- Available as Docker container or Electron desktop app

### 5.2 Option B: Visual Flow Builders

#### Flowise[^8]

**54.7K stars** — Drag-and-drop visual agent builder:
- LangChain-based component nodes (Ollama, LM Studio nodes available)
- **Embeddable chat widget** for websites
- REST API for every chatflow (webhook-style invocation)
- Node.js/React + TypeScript stack

```bash
npm install -g flowise && npx flowise start
# Available at http://localhost:3000
```

#### Langflow[^8]

**152K stars** — Visual pipeline builder with MCP server export:
- Python/FastAPI backend + React/ReactFlow canvas
- Deploy flows as REST API endpoints
- **Export flows as MCP servers** (consumed by Claude Desktop etc.)
- `uv pip install langflow && python -m langflow run`

### 5.3 Option C: Custom UI Development

For full control, build a custom stack:

```mermaid
graph LR
    FE["Frontend<br/>Next.js + Vercel AI SDK<br/>(useChat + streamText)"]
    BE["Backend<br/>FastAPI + LangGraph<br/>SSE Streaming"]
    OL["Ollama<br/>qwen3:14b"]
    DB["Vector DB<br/>ChromaDB / Qdrant"]
    
    FE <-->|SSE Stream| BE
    BE --> OL
    BE --> DB
```

**Frontend: Next.js + Vercel AI SDK**[^9]:

```bash
npm install ai @ai-sdk/react ollama-ai-provider-v2
```

```typescript
// app/api/chat/route.ts
import { streamText, tool } from 'ai';
import { createOllama } from 'ollama-ai-provider-v2';
import { z } from 'zod';

const ollama = createOllama({ baseURL: 'http://localhost:11434/api' });

export async function POST(req: Request) {
  const { messages } = await req.json();
  
  const result = streamText({
    model: ollama('qwen3:14b'),
    messages,
    tools: {
      getWeather: tool({
        description: 'Get weather for a location',
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }) => ({ temp: 22, condition: 'sunny' }),
      }),
    },
    stopWhen: isStepCount(10),  // Max 10 agent steps
  });
  
  return createUIMessageStreamResponse({ stream: result.stream });
}
```

```tsx
// app/page.tsx
'use client';
import { useChat } from '@ai-sdk/react';

export default function ChatPage() {
  const { messages, sendMessage, status, stop } = useChat({
    transport: new DefaultChatTransport({ url: '/api/chat' }),
  });

  return (
    <div>
      {messages.map(message => (
        <div key={message.id}>
          {message.parts.map((part, i) => {
            if (part.type === 'text') return <p key={i}>{part.text}</p>;
            if (part.type.startsWith('tool-')) {
              // Shows tool calls with streaming states
              if (part.state === 'input-streaming') return <div>Calling tool...</div>;
              if (part.state === 'output-available') return <div>{JSON.stringify(part.output)}</div>;
            }
          })}
        </div>
      ))}
      {status === 'streaming' && <button onClick={stop}>Stop</button>}
    </div>
  );
}
```

**Tool call states in Vercel AI SDK v5**[^9]: `input-streaming` → `input-available` → `output-available` | `output-error` | `approval-requested`

---

## 6. Agentic Architecture Patterns

### 6.1 ReAct Pattern (Primary Recommended Pattern)

The dominant agent loop: **Thought → Action → Observation → Thought...**[^10]

```
┌─────────────────────────────────────────────────┐
│  Input: Task / User Question                     │
└──────────────────────────┬──────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  THOUGHT     │  ← LLM Reasoning (chain-of-thought)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   ACTION     │  ← Tool Call (JSON / Python code)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ OBSERVATION  │  ← Tool Result appended to context
                    └──────┬──────┘
                           │
              ┌────────────▼──────────────┐
              │  Contains "Final Answer"?  │
              └─────┬───────────┬─────────┘
                   NO          YES
                    │           └─→ Return Answer
                    └──→ Loop back to THOUGHT
```

### 6.2 Plan-and-Execute Pattern

Decouple planning from execution to reduce context pollution in long tasks[^10]:

```python
class PlanExecuteState(TypedDict):
    task: str
    plan: List[str]
    current_step: int
    results: List[str]

# Node 1: Planner generates steps
def planner(state):
    plan = planner_llm.invoke(state["task"])
    return {"plan": plan, "current_step": 0}

# Node 2: Executor handles ONE step at a time
def executor(state):
    step = state["plan"][state["current_step"]]
    result = executor_agent.invoke(step)
    return {"results": state["results"] + [result], "current_step": state["current_step"] + 1}

# Optional Node 3: Replanner adjusts based on new info
def replanner(state):
    if state["current_step"] >= len(state["plan"]):
        return {"final_answer": synthesize(state["results"])}
    # Re-evaluate plan based on learned info...
```

### 6.3 Multi-Agent Architectures

**Supervisor / Orchestrator Pattern:**[^10]

```
         ┌────────────────────────────┐
         │       ORCHESTRATOR         │
         │   Routes tasks to          │
         │   sub-agents as tools      │
         └──┬─────────┬─────────┬────┘
            │         │         │
     ┌──────▼──┐ ┌────▼─────┐ ┌▼──────────┐
     │ Search  │ │   Code   │ │  Writing  │
     │  Agent  │ │  Agent   │ │   Agent   │
     └─────────┘ └──────────┘ └───────────┘
```

**LangGraph Supervisor Implementation:**[^5]
```python
from langgraph.prebuilt import create_react_agent

researcher = create_react_agent(llm, [search_tool], name="researcher")
coder = create_react_agent(llm, [code_tool], name="coder")

# Supervisor routes between agents
def supervisor(state: MessagesState):
    response = supervisor_llm.invoke(state["messages"])
    return {"next": response.next_agent}

builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)
builder.add_conditional_edges("supervisor", lambda s: s["next"])
```

---

## 7. Tool Use & Function Calling

### 7.1 Universal JSON Schema Tool Definition

All major frameworks and local LLMs use this format[^10]:

```json
{
  "type": "function",
  "function": {
    "name": "search_database",
    "description": "Search the internal knowledge database for relevant documents",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query"
        },
        "max_results": {
          "type": "integer",
          "description": "Maximum results to return",
          "default": 5
        }
      },
      "required": ["query"]
    }
  }
}
```

### 7.2 LangChain Tool Definition

```python
from langchain_core.tools import tool
from pydantic import BaseModel

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Tools are passed to LLM as JSON schema, results returned as ToolMessage
llm_with_tools = llm.bind_tools([search_web])
```

### 7.3 Error Handling and Retry Pattern[^10]

```python
MAX_RETRIES = 3

async def execute_with_retry(tool_call, context):
    for attempt in range(MAX_RETRIES):
        try:
            result = await tool_registry[tool_call.name](**tool_call.args)
            return result
        except ToolExecutionError as e:
            if attempt == MAX_RETRIES - 1:
                # Inject error back to agent for self-correction
                return f"ERROR: Tool '{tool_call.name}' failed: {str(e)}. Please try a different approach."
            await asyncio.sleep(2 ** attempt)  # exponential backoff
        except ValidationError as e:
            return f"ARGUMENT_ERROR: {e}. Check required parameters."
```

---

## 8. MCP: Model Context Protocol

MCP is Anthropic's open protocol (2024) that standardizes how LLM applications connect to external tools and data. It functions as "USB-C for AI integrations."[^11]

### 8.1 Architecture

```
┌─────────────────────────────────────────────────────┐
│               MCP HOST                              │
│  (Open WebUI, LibreChat, LangGraph app, custom app) │
│                                                     │
│  ┌──────────┐   ┌──────────┐  ┌──────────┐         │
│  │  MCP     │   │  MCP     │  │  MCP     │         │
│  │ Client 1 │   │ Client 2 │  │ Client 3 │         │
│  └────┬─────┘   └────┬─────┘  └────┬─────┘         │
└───────┼──────────────┼─────────────┼────────────────┘
        │ stdio        │ HTTP        │ SSE
  ┌─────▼──────┐  ┌────▼─────┐ ┌────▼──────┐
  │ Filesystem │  │   Git    │ │ Database  │
  │ MCP Server │  │ MCP Svr  │ │ MCP Svr   │
  └────────────┘  └──────────┘ └───────────┘
```

### 8.2 Reference MCP Servers[^11]

| Server | Install | Description |
|---|---|---|
| `filesystem` | `npx @modelcontextprotocol/server-filesystem /path` | File read/write with ACL |
| `git` | `uvx mcp-server-git --repository /path` | Git operations |
| `memory` | `npx @modelcontextprotocol/server-memory` | Knowledge graph persistent memory |
| `fetch` | Built-in | Web content fetching |
| `postgres` | `npx @modelcontextprotocol/server-postgres postgresql://...` | Read-only DB + schema |
| `sequentialthinking` | `npx @modelcontextprotocol/server-sequentialthinking` | Dynamic thought sequences |
| `puppeteer` | `npx @modelcontextprotocol/server-puppeteer` | Browser automation |

### 8.3 LangGraph + MCP Integration[^11]

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

model = ChatOllama(model="qwen3:14b")

client = MultiServerMCPClient({
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"],
        "transport": "stdio",
    },
    "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "transport": "stdio",
    },
})

tools = await client.get_tools()
agent = create_react_agent(model, tools)
response = await agent.ainvoke({"messages": "What files are in my docs folder?"})
```

### 8.4 Building Custom MCP Servers (Python)[^11]

```python
# server.py — A complete MCP server in ~15 lines
from mcp.server import MCPServer

mcp = MCPServer("MyKnowledgeBase")

@mcp.tool()
def search_documents(query: str, max_results: int = 5) -> list[dict]:
    """Search documents by semantic similarity."""
    # Your RAG/vector DB implementation
    return [{"id": 1, "content": "...", "score": 0.95}]

@mcp.resource("docs://{doc_id}")
def get_document(doc_id: str) -> str:
    """Retrieve a specific document by ID."""
    return f"Content of document {doc_id}"

# Run as stdio server (for local use)
# mcp dev server.py
# python server.py

# Run as HTTP server (for remote/multi-client)
if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
```

### 8.5 MCPO: MCP → OpenAPI Bridge for Open WebUI[^11]

```bash
# Single MCP server → OpenAPI
uvx mcpo --port 8000 --api-key "secret" -- uvx mcp-server-time --local-timezone=Europe/Berlin

# Multi-server via config
mcpo --config config.json --hot-reload

# Add to Open WebUI: Settings → Tools → URL: http://localhost:8000
```

### 8.6 MCP Support Matrix

| Framework | MCP Support | Notes |
|---|---|---|
| **LangChain / LangGraph** | ✅ Official | `langchain-mcp-adapters` |
| **Open WebUI** | ✅ Via MCPO | Auto-discovers OpenAPI tools |
| **LibreChat** | ✅ Built-in | Config file, all transports |
| **smolagents** | ✅ `MCPClient` | stdio + HTTP |
| **Claude Desktop** | ✅ Native (reference) | JSON config |

---

## 9. RAG: Retrieval-Augmented Generation

### 9.1 Vector Database Selection

| DB | Deployment | Best For | Stars |
|---|---|---|---|
| **ChromaDB** | Embedded or server | Easiest dev path, prototyping | Active |
| **Qdrant** | Docker (Rust-based) | Production self-hosted, filtering | Active |
| **pgvector** | PostgreSQL extension | Existing Postgres, simplicity | Active |
| **FAISS** | Pure Python library | Lightweight, in-process | Meta |
| **LanceDB** | Serverless columnar | Multimodal, no server needed | Active |
| **Milvus** | Docker + etcd + MinIO | Enterprise billion-scale | Active |
| **Weaviate** | Docker | Built-in reranking, BM25+vector hybrid | Active |

### 9.2 ChromaDB (Easiest Start)[^12]

```python
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Persistent local DB — no server needed
client = chromadb.PersistentClient(path="./chroma_db")

# Use local Ollama embedding model
ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434/api/embeddings"
)

collection = client.get_or_create_collection("my_docs", embedding_function=ef)

# Add documents
collection.add(
    documents=["Document content here..."],
    metadatas=[{"source": "manual.pdf", "page": 1}],
    ids=["doc_001"]
)

# Query
results = collection.query(
    query_texts=["How do I configure the server?"],
    n_results=5,
    where={"source": "manual.pdf"}
)
```

### 9.3 Qdrant (Best Production Self-Hosted)[^12]

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
```

### 9.4 RAG as a Tool for Agents[^10]

```python
def create_rag_tool(collection):
    def knowledge_base_search(query: str, n_results: int = 5) -> str:
        """Search the knowledge base for relevant information."""
        results = collection.query(query_texts=[query], n_results=n_results)
        if not results['documents'][0]:
            return "No relevant information found."
        
        output = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            output.append(f"[Source: {meta.get('source', 'unknown')}]\n{doc}")
        return "\n\n---\n\n".join(output)
    
    return knowledge_base_search
```

### 9.5 Local Embedding Models

| Model | Dims | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` (sentence-transformers) | 384 | ⚡ Fast | Good general |
| `nomic-embed-text` (via Ollama) | 768 | Medium | Better quality, easy to use |
| `mxbai-embed-large` (via Ollama) | 1024 | Medium | Best open-source quality |
| `bge-m3` | 1024 | Medium | Best multilingual |
| `snowflake-arctic-embed-m` | 768 | Fast | Good for code/technical |

---

## 10. Memory Systems

### 10.1 Three-Tier Architecture[^10]

```
TIER 1: In-Context (Short-term)
  • Current conversation messages
  • Tool call history this session
  • Limited by context window (8K–128K tokens)
  • Implementation: Message history in LangGraph state

TIER 2: Episodic (Medium-term, Cross-session)
  • Summaries of past sessions
  • Key decisions/facts extracted and stored
  • Retrieved via semantic similarity
  • Implementation: Vector DB + timestamp metadata

TIER 3: Semantic / Long-term (Persistent)
  • Structured facts (user preferences, entities)
  • Procedural knowledge (how-tos, workflows)
  • Implementation: Vector DB + KV store (Redis/SQLite)
```

### 10.2 LangGraph Memory with Checkpointing[^5]

```python
from langgraph.checkpoint.sqlite import SqliteSaver   # dev
from langgraph.checkpoint.postgres import PostgresSaver  # production

# SQLite for dev
with SqliteSaver.from_conn_string("checkpoints.db") as saver:
    app = graph.compile(checkpointer=saver)

# Each thread_id = one conversation = independently resumable
config = {"configurable": {"thread_id": "user-123-session-456"}}
result = app.invoke({"messages": [user_message]}, config=config)

# Full conversation replay from any checkpoint:
history = list(app.get_state_history(config))
```

### 10.3 Episodic Memory Extraction[^10]

```python
MEMORY_EXTRACTION_PROMPT = """
Summarize key facts from this conversation to remember:
- User preferences stated
- Tasks completed
- Important entities mentioned
- Decisions made

Output as JSON: {facts: [], preferences: [], entities: [], decisions: []}
"""

async def extract_and_store_memory(conversation, user_id, vector_store):
    summary = await llm.ainvoke(MEMORY_EXTRACTION_PROMPT.format(conversation=conversation))
    parsed = json.loads(summary)
    
    for fact in parsed["facts"]:
        await vector_store.add_texts(
            [fact],
            metadatas=[{"user_id": user_id, "type": "fact", "timestamp": now()}]
        )
```

---

## 11. Streaming Architecture

### 11.1 SSE vs WebSocket Decision

| Feature | SSE | WebSocket |
|---|---|---|
| **Direction** | Server → Client only | Bidirectional |
| **Protocol** | HTTP | WS upgrade |
| **Reconnect** | Automatic | Manual |
| **Proxying** | Works with standard HTTP proxies | Requires special config |
| **Best for** | LLM token streaming | Real-time bidirectional (voice, collaborative) |
| **Recommendation** | ✅ Use for LLM chat streaming | Use for live collaboration features |

**Most production AI chat UIs use SSE** for streaming LLM responses (Open WebUI, LibreChat, AnythingLLM, Vercel AI SDK default).

### 11.2 FastAPI SSE Streaming Implementation[^9]

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Handle streaming chat requests with full agent step visibility."""
    
    async def stream_generator():
        try:
            async for event in agent_graph.astream_events(
                {"messages": [HumanMessage(content=request.message)]},
                version="v2"
            ):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                
                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name']})}\n\n"
                
                elif kind == "on_tool_end":
                    yield f"data: {json.dumps({'type': 'tool_end', 'output': str(event['data'].get('output', ''))})}\n\n"
                
                elif kind == "on_chain_end" and event["name"] == "LangGraph":
                    yield "data: [DONE]\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # CRITICAL: disables nginx buffering
        }
    )
```

---

## 12. Complete Stack Recommendations

### 12.1 Path A: Rapid Prototyping (< 30 minutes)

**Use when**: Evaluating capabilities, demos, personal use

```bash
docker compose up -d

# Then: Open http://localhost:3000
# Pull a model: Settings → Models → Pull qwen3:14b
```

```yaml
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    volumes: [ollama:/root/.ollama]
    restart: unless-stopped
    # For GPU: add deploy.resources.reservations below

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    depends_on: [ollama]
    ports: ["3000:8080"]
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    volumes: [open-webui:/app/backend/data]
    restart: unless-stopped

volumes:
  ollama: {}
  open-webui: {}
```

Add MCP tools via MCPO in a few lines, configure RAG in the UI settings.

### 12.2 Path B: Production AI Platform

**Use when**: Multi-user deployment, monitoring, custom agent workflows

```yaml
# Full production stack
services:
  ollama:
    image: ollama/ollama
    restart: unless-stopped
    ports: ["11434:11434"]
    volumes: [ollama-data:/root/.ollama]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]   # GPU passthrough

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    restart: unless-stopped
    depends_on: [ollama, postgres, redis, qdrant]
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
      DATABASE_URL: postgresql://user:pass@postgres:5432/openwebui
      REDIS_URL: redis://redis:6379
      VECTOR_DB: qdrant
      QDRANT_URI: http://qdrant:6333

  postgres:
    image: postgres:16
    restart: unless-stopped
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes: [postgres-data:/var/lib/postgresql/data]

  redis:
    image: redis:7
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant
    restart: unless-stopped
    ports: ["6333:6333"]
    volumes: [qdrant-data:/qdrant/storage]

  nginx:
    image: nginx:stable
    depends_on: [open-webui]
    ports: ["80:80", "443:443"]
    volumes: [./nginx/default.conf:/etc/nginx/conf.d/default.conf]

  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes: [./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml]

  grafana:
    image: grafana/grafana
    ports: ["3001:3000"]
    volumes: [grafana-data:/var/lib/grafana]

volumes:
  ollama-data: {}
  postgres-data: {}
  qdrant-data: {}
  grafana-data: {}
```

### 12.3 Path C: Custom Agentic Application

**Use when**: Custom workflows, domain-specific agents, embedding in existing products

**Recommended tech stack:**

```
Backend:  FastAPI (Python) + LangGraph + LangChain
LLM:      Ollama (dev) / vLLM (production)
Models:   Qwen3:14b (general) / Hermes3:8b (tool-calling)
Vector DB: ChromaDB (dev) / Qdrant (production)
Memory:   LangGraph SQLite checkpointer (dev) / PostgreSQL (production)
Tools:    Custom Python functions + MCP servers
Frontend: Next.js + Vercel AI SDK (useChat + streamText)
Auth:     JWT + FastAPI-Users
Infra:    Docker Compose + Nginx
```

**Project structure:**
```
my-agent-app/
├── backend/
│   ├── main.py               # FastAPI app, SSE endpoints
│   ├── agent/
│   │   ├── graph.py          # LangGraph StateGraph definition
│   │   ├── state.py          # AgentState TypedDict
│   │   ├── nodes.py          # Graph nodes (llm, tools, memory)
│   │   └── tools/
│   │       ├── search.py     # Web search tool
│   │       ├── rag.py        # Knowledge base search tool
│   │       └── calculator.py # Math tool
│   ├── memory/
│   │   ├── vector_store.py   # ChromaDB/Qdrant setup
│   │   └── episodic.py       # Memory extraction
│   └── mcp/
│       └── server.py         # Custom MCP server (optional)
├── frontend/
│   ├── app/
│   │   ├── api/chat/route.ts # Next.js API route (streamText)
│   │   └── page.tsx          # Chat UI (useChat hook)
│   └── components/
│       ├── ChatMessage.tsx
│       └── ToolCallDisplay.tsx
├── docker-compose.yml
└── nginx/default.conf
```

---

## 13. Production Deployment

### 13.1 GPU Passthrough in Docker[^9]

```yaml
# Prerequisite: sudo apt install nvidia-container-toolkit && sudo systemctl restart docker
services:
  ollama:
    image: ollama/ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all          # or device_ids: ['0'] for specific GPU
              capabilities: [gpu] # MANDATORY — omitting causes error
    environment:
      NVIDIA_VISIBLE_DEVICES: all
```

### 13.2 Nginx Configuration for SSE + WebSocket[^9]

```nginx
http {
  # WebSocket upgrade mapping
  map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
  }

  server {
    listen 80;

    # SSE endpoint — disable buffering
    location /api/chat/stream {
      proxy_pass http://backend:8000;
      proxy_buffering off;          # CRITICAL for SSE
      proxy_cache off;
      proxy_read_timeout 600s;      # Long timeout for LLM inference
      proxy_send_timeout 600s;
      add_header 'X-Accel-Buffering' 'no';
    }

    # WebSocket endpoint
    location /ws/ {
      proxy_pass http://backend:8000;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection $connection_upgrade;
      proxy_read_timeout 3600s;
    }

    # Standard API + SPA
    location / {
      proxy_pass http://frontend:3000;
      proxy_set_header Host $host;
    }
  }
}
```

### 13.3 Authentication Pattern

For multi-user custom agents, use JWT with FastAPI:

```python
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTStrategy

# Token-based auth for WebSocket/SSE streams
async def get_current_user_ws(
    websocket: WebSocket,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    user = await decode_jwt_token(token, db)
    if not user:
        await websocket.close(code=4001)
    return user
```

---

## 14. Key Repositories Summary

| Repository | Purpose | Stars | License |
|---|---|---|---|
| [ollama/ollama](https://github.com/ollama/ollama) | Local LLM server (Go + llama.cpp) | ~100K | MIT |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Core inference engine | ~80K | MIT |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Production LLM serving | ~50K | Apache 2.0 |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Stateful agent graphs | 37.5K | MIT |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | Agent framework + integrations | 142K | MIT |
| [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | Role-based multi-agent | 55.7K | MIT |
| [huggingface/smolagents](https://github.com/huggingface/smolagents) | Lightweight ReAct agents | 28.4K | Apache 2.0 |
| [open-webui/open-webui](https://github.com/open-webui/open-webui) | Feature-complete chat UI | 145K | Custom |
| [danny-avila/LibreChat](https://github.com/danny-avila/LibreChat) | Multi-provider chat UI | 40.8K | MIT |
| [FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise) | Visual agent builder | 54.7K | Apache 2.0 |
| [langflow-ai/langflow](https://github.com/langflow-ai/langflow) | Visual pipeline + MCP server | 152K | MIT |
| [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm) | No-code agent builder | ~45K | MIT |
| [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) | Reference MCP servers | Active | MIT |
| [open-webui/mcpo](https://github.com/open-webui/mcpo) | MCP → OpenAPI bridge | Active | MIT |
| [chroma-core/chroma](https://github.com/chroma-core/chroma) | Embedded vector DB | Active | Apache 2.0 |
| [qdrant/qdrant](https://github.com/qdrant/qdrant) | Production vector DB | Active | Apache 2.0 |
| [vercel/ai](https://github.com/vercel/ai) | Vercel AI SDK (useChat, streamText) | Active | Apache 2.0 |

---

## 15. Decision Guide: Choosing Your Stack

```mermaid
graph TD
    Start([I want to build a local agentic AI<br/>with web chat UI]) --> Q1{How much<br/>custom logic<br/>do I need?}
    
    Q1 -->|None – use it as-is| A1[Open WebUI + Ollama<br/>Docker Compose<br/>Add MCP tools for extra capabilities]
    
    Q1 -->|Some – visual builder| A2[Flowise or Langflow<br/>+ Ollama<br/>Visual drag-and-drop agents]
    
    Q1 -->|Full control| Q2{Team language<br/>preference?}
    
    Q2 -->|Python| Q3{Agent complexity?}
    Q2 -->|TypeScript| A3[Vercel AI SDK + Next.js<br/>+ Node.js backend<br/>+ Ollama]
    
    Q3 -->|Simple to Medium| A4[smolagents + Ollama<br/>+ FastAPI<br/>+ Vercel AI SDK frontend]
    
    Q3 -->|Complex / Production| A5[LangGraph + LangChain<br/>+ Ollama / vLLM<br/>+ FastAPI SSE<br/>+ Next.js / SvelteKit]
    
    Q3 -->|Multi-Agent Collaboration| A6[CrewAI + Ollama<br/>via LiteLLM<br/>+ FastAPI<br/>+ Vercel AI SDK]
```

---

## Confidence Assessment

| Area | Confidence | Notes |
|---|---|---|
| Ollama API and tool calling | ✅ High | Official docs + live GitHub data |
| llama.cpp tool calling formats | ✅ High | Official docs fetched |
| vLLM tool call parsers | ✅ High | Official docs, verified July 2026 |
| Model benchmarks (BFCL V4) | ✅ High | Gorilla Berkeley Leaderboard sourced |
| LangGraph StateGraph API | ✅ High | Source code + README fetched |
| CrewAI LiteLLM integration | ✅ High | README + source |
| smolagents CodeAgent | ✅ High | HuggingFace docs + source |
| Open WebUI features | ✅ High | Live GitHub + package.json |
| MCP spec (2025-11-25) | ✅ High | Official spec fetched |
| MCPO architecture | ✅ High | Official README |
| Vercel AI SDK v5 | ✅ High | Live docs |
| Docker GPU passthrough | ✅ High | Official docs verified |
| AutoGen status | ✅ High | **MAINTENANCE MODE** confirmed from README |
| Model VRAM estimates | ⚠️ Medium | Derived from formula; actual may vary ±10% |
| Custom stack project structures | ⚠️ Medium | Based on patterns from discovered repos; not a single authoritative source |
| "Best" stack choices | ℹ️ Opinion | Based on community adoption + benchmarks |

---

## Footnotes

[^1]: [ollama/ollama](https://github.com/ollama/ollama) — README, API docs, `ollama.com/blog/tool-support`, `docs.ollama.com/api`

[^2]: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — `tools/server/README.md`, `docs/function-calling.md`

[^3]: [vllm-project/vllm](https://github.com/vllm-project/vllm) — README, `docs.vllm.ai/en/latest/features/tool_calling/`

[^4]: [ollama/ollama model library](https://ollama.com/library) — `ollama.com/library/qwen3`, `ollama.com/library/hermes3`, `ollama.com/library/llama3.1`, gorilla.cs.berkeley.edu/leaderboard.html

[^5]: [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — README, `libs/langgraph/langgraph/graph/state.py`, checkpoint libraries

[^6]: [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) — README, docs on local LLM integration via LiteLLM

[^7]: [huggingface/smolagents](https://github.com/huggingface/smolagents) — README v1.26.0, `huggingface.co/docs/smolagents/guided_tour`

[^8]: [open-webui/open-webui](https://github.com/open-webui/open-webui) — `backend/open_webui/main.py`, `package.json`; [danny-avila/LibreChat](https://github.com/danny-avila/LibreChat); [Mintplex-Labs/anything-llm](https://github.com/Mintplex-Labs/anything-llm); [FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise); [langflow-ai/langflow](https://github.com/langflow-ai/langflow)

[^9]: [vercel/ai](https://github.com/vercel/ai) — README, `ai-sdk.dev/docs`; `ssgrummons/rag-with-milvus-langchain-streamlit:backend/src/app.py`; `LangGraph-GUI/LangGraph-GUI:docker-compose.yml`; nginx docs

[^10]: [react paper arXiv:2210.03629](https://arxiv.org/abs/2210.03629); `huggingface.co/blog/open-source-llms-as-agents`; `chroma-core/chroma` README; `meta-llama/llama-models:models/llama3_1/prompt_format.md`

[^11]: [modelcontextprotocol/specification](https://github.com/modelcontextprotocol/specification) — `docs/specification/2025-11-25/basic/transports.mdx`; [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers); [open-webui/mcpo](https://github.com/open-webui/mcpo) — README; [langchain-ai/langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)

[^12]: [chroma-core/chroma](https://github.com/chroma-core/chroma) — README, `docs.trychroma.com`; [qdrant/qdrant](https://github.com/qdrant/qdrant) — docs
