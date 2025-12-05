# Astronix-Local-RAG-Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2-orange?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Edge%20Compatible-lightgrey?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green?style=for-the-badge)

**A low-latency, fully offline Voice-to-Voice RAG pipeline designed for edge hardware. Built for Technex 2025.**

---

## 📖 Overview

**Astronix** is an intelligent, voice-enabled AI agent capable of answering domain-specific queries without an internet connection. Unlike standard cloud-based chatbots, Astronix runs entirely locally, leveraging a split-compute architecture to optimize performance on consumer hardware (e.g., NVIDIA RTX 3050).

It utilizes **Retrieval Augmented Generation (RAG)** to ground its responses in a custom dataset (College Event Manuals), eliminating hallucinations while maintaining a conversational persona.

## ⚙️ System Architecture

This pipeline is optimized for constrained VRAM environments (4GB) by strategically offloading tasks:

| Component | Technology | Compute Target | Description |
| :--- | :--- | :--- | :--- |
| **Input (ASR)** | **Faster-Whisper** (`base.en`) | **CPU** | Quantized (Int8) inference for sub-second speech-to-text. |
| **Memory (RAG)** | **ChromaDB** + **FastEmbed** | **CPU** | Vector storage and semantic search using BAAI embeddings. |
| **Reasoning (LLM)** | **Llama 3.2 1B** (via Ollama) | **GPU** | Lightweight instruct model for context synthesis and response generation. |
| **Output (TTS)** | **pyttsx3** | **CPU** | Offline text-to-speech engine with low-latency audio stream. |

---

## 🚀 Key Features

* **🔒 Privacy-First:** Zero data leaves the local machine. Works in air-gapped environments.
* **⚡ Edge Optimized:** Runs smoothly on 12th Gen Intel i7 + RTX 3050 (4GB VRAM) by balancing CPU/GPU loads.
* **🧠 Context-Aware:** Uses RAG to answer specific questions about schedules, fees, and guests using a custom knowledge base.
* **🗣️ Push-to-Talk Interface:** Interactive "Walkie-Talkie" style loop for controlled audio input.
* **🛡️ Social Filtering Layer:** Pre-processing logic to handle greetings and casual chit-chat without querying the vector database.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10 or higher
* [Ollama](https://ollama.com/) installed and running.
* [FFmpeg](https://ffmpeg.org/) installed and added to System PATH.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/Astronix-Local-RAG-Pipeline.git](https://github.com/yourusername/Astronix-Local-RAG-Pipeline.git)
cd Astronix-Local-RAG-pipeline
```


### 2. Set up Virtual Environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
(Note: Ensure requirements.txt includes: ollama, faster-whisper, langchain-community, chromadb, sounddevice, numpy, pyttsx3, keyboard)

### 4. Pull Local Models

```bash 
ollama pull llama3.2:1b
```

# 💻 Usage

## **Step 1: Ingest Data (Build Memory)**
If this is your first run or if you have updated `technex_data.txt`:

\`\`\`bash
python build_memory.py
\`\`\`

This script chunks your text data, creates embeddings, and stores them in the local \`./technex_db\` vector store.

---

## **Step 2: Run the Pipeline**
\`\`\`bash
python main.py
\`\`\`

---

## **Step 3: Interaction**
- Wait for the system to say **"✅ SYSTEM ONLINE!"**
- Press and **HOLD** the 'S' key on your keyboard
- Speak your question (e.g., "Who is the Chief Guest?")
- Release the 'S' key to hear the response

---

# 📂 Project Structure

```plaintext
Astronix-Local-RAG-Pipeline/
├── main.py              # Core logic: Integrates ASR, RAG, LLM, and TTS loops
├── build_memory.py      # ETL script: Loads text -> Chunks -> Embeds -> VectorDB
├── technex_data.txt     # The Source of Truth (Domain Knowledge)
├── technex_db/          # Persistent Vector Database (ChromaDB)
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```
---

# 🔧 Troubleshooting

- **Audio Access Error:** Ensure your microphone is set as the default system input.
- **Ollama Connection Refused:** Make sure the Ollama app is running in the background tray.
- **"FFmpeg not found":** Verify FFmpeg is added to your Windows Environment Variables (PATH).

---

# 🔮 Future Improvements

- GUI Integration: Replace terminal input with a Tkinter/PyQt5 interface.
- Visual Avatar: Sync TTS output with a lip-syncing 3D avatar.
- Hybrid Mode: Switch to Groq API (Cloud) when internet is available for higher model accuracy.

---

# 📜 License

This project is licensed under the MIT License.

Built with ❤️ by **Rochan Awasthi** for **Technex 2025**.

---


