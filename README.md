# AI-Powered Universal Document Analyst (RAG System)

## 📌 Project Overview
This project is an **Enterprise-grade Retrieval-Augmented Generation (RAG)** application. It allows users to upload any PDF document and interact with it using an AI assistant. The system uses **Mistral AI** for high-level reasoning and a **Local CLIP Model** for lightning-fast, privacy-focused document searching.

## 🚀 Key Features
- **Universal Analysis**: Works with transcripts, invoices, reports, or any text-based PDF.
- **Local Embeddings**: Uses `CLIP-ViT-B-32` locally to ensure no API limits on document indexing.
- **Structured Insights**: Automatically generates a one-sentence summary and a list of key findings.
- **Glassmorphism UI**: A modern, responsive dashboard with real-time system status indicators.

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API).
- **Backend**: Python, Flask, Flask-CORS.
- **AI/LLM**: Mistral AI (`mistral-large-latest`).
- **Vector Database**: ChromaDB.
- **Embeddings**: Sentence-Transformers (CLIP).

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd <project-folder>

Install Dependencies
Bash

pip install -r requirements.txt
3. Environment Configuration
Create a .env file in the root directory and add your Mistral API key:

Code snippet

MISTRAL_API_KEY=your_actual_key_here
4. Run the Application
Bash

python app.py
The application will automatically open in your default browser at http://127.0.0.1:5000.