<div align="center">
  
# 🎙️ LiveKit RAG Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LiveKit](https://img.shields.io/badge/LiveKit-00E5FF?style=for-the-badge&logo=livekit&logoColor=black)](https://livekit.io/)
[![Groq](https://img.shields.io/badge/Groq-f3f4f6?style=for-the-badge&logo=groq&logoColor=f56565)](https://groq.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

A multi-user, role-aware Retrieval-Augmented Generation (RAG) assistant built with **FastAPI** and **LiveKit**. This application powers real-time voice calls and group chat, equipped with an intelligent suggestion engine designed specifically for project managers interacting with stakeholders.

</div>

---

## ✨ Features

- 🗣️ **Real-time Voice Pipeline**: Multi-party audio calls and low-latency transcriptions powered by **LiveKit** and **Deepgram**.
- 🧠 **Context-Aware RAG**: Retrieval-Augmented Generation using local `sentence-transformers` embeddings (BAAI/bge-small-en-v1.5) and **Groq** LLMs.
- 💬 **Live Chat & Websockets**: Real-time, group-scoped chat rooms enabling seamless collaboration.
- 🗂️ **Smart Knowledge Base**: Upload documents (PDFs, TXT, MD) and instantly parse them for RAG indexing and search.
- 🤖 **Suggestion Engine**: Automatically classifies stakeholder questions vs statements, drafts intelligent responses based on project context, and pushes them securely to Project Managers.
- 🔒 **Role-Based Access Control**: Strict access controls managing permissions for Developers, Project Managers, and Stakeholders.

---

## 🛠️ Tech Stack

### **Backend Core**
- [FastAPI](https://fastapi.tiangolo.com/) - High-performance web framework for APIs and WebSockets.
- [SQLite](https://www.sqlite.org/) - Lightweight database for groups, messages, transcripts, and users.

### **AI & Voice**
- [LiveKit Agents](https://github.com/livekit/agents) - Real-time audio processing and streaming.
- [Deepgram](https://deepgram.com/) - STT (Speech-to-Text) provider via LiveKit plugins.
- [Groq](https://groq.com/) - Lightning-fast LLM inference for intent classification and RAG generation.
- [Sentence Transformers](https://sbert.net/) - High-quality local CPU text embeddings.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system. 

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and configure the following variables:

```ini
# LiveKit Configuration
LIVEKIT_URL=wss://<your-livekit-project>.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# AI Models (Groq & Embeddings)
GROQ_API_KEY=your_groq_api_key
GROQ_TEXT_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Server configuration
SERVER_BASE_URL=http://localhost:8000
CALL_ASSIST_DEBUG=true
```

### 4. Running the Application

This architecture is split into a central API server and a dedicated LiveKit AI agent worker.

**Start the FastAPI Server:**
```bash
uvicorn server:app --reload
```
*The server handles auth, group chats, document parsing, and serving the frontend.*

**Start the Call-Assist Agent:**
In a separate terminal (with the virtual environment activated), start the LiveKit worker:
```bash
python agent.py dev
```
*The agent will connect to your LiveKit cloud instance and handle real-time audio transcription and processing.*

---

## 📂 Project Structure

```text
├── agent.py               # LiveKit worker logic (Deepgram STT & WebRTC)
├── server.py              # Core FastAPI server (Chat, WebSockets, DB interaction)
├── rag.py                 # Retrieval-Augmented Generation & Intent Detection
├── db.py                  # SQLite database models and queries
├── auth.py                # JWT authentication and password hashing
├── seed.py                # Database seeder for default users/groups
├── requirements.txt       # Python dependencies
├── frontend/              # Static frontend assets (HTML/JS/CSS)
├── attachments/           # Uploaded chat attachments (auto-generated)
├── kb_documents/          # Root directory for Knowledge Base docs (auto-generated)
└── model_cache/           # Cached embedding models (auto-generated)
```

---

## 📝 Usage Details

- **Groups & Isolation:** RAG context is strictly isolated per group. A user querying data in Project A will never receive context or document embeddings from Project B.
- **Intent Classification:** The system uses Groq to identify if an utterance is a `QUESTION` or a `STATEMENT`. RAG suggestions are *only* generated for questions to reduce noise for Project Managers.
- **Transcripts:** Call transcripts are processed incrementally and embedded for future retrieval. They are designed to act as an invisible context layer and are never directly rendered in the chat UI.
- **Open Source Models** The Models used are hosted by Groq and Deepgram.