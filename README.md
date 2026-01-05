# AI-Powered Universal Document Analyst (RAG System)

## 📌 Project Overview
This project is an **Enterprise-grade Retrieval-Augmented Generation (RAG)** application. It allows users to upload any PDF document and interact with it using an AI assistant. The system uses **Mistral AI** for high-level reasoning and a **Local CLIP Model** for lightning-fast, privacy-focused document searching.

## 🚀 Key Features
- **Universal Analysis**: Works with transcripts, invoices, reports, or any text-based PDF.
- **Local Embeddings**: Uses `CLIP-ViT-B-32` locally to ensure no API limits on document indexing.
- **Structured Insights**: Automatically generates a concise summary and a list of key findings.
- **Glassmorphism UI**: A modern, responsive dashboard with real-time system status indicators.

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript (Fetch API).
- **Backend**: Python, Flask, Flask-CORS.
- **AI/LLM**: Mistral AI (`mistral-large-latest`).
- **Vector Database**: ChromaDB.
- **Embeddings**: Sentence-Transformers (CLIP).

## 📂 Project Structure
```text
AI-KnowledgeHUB/
├── app.py              # Flask Backend & AI Logic
├── site.html           # Frontend UI
├── requirements.txt    # Python Dependencies
├── .env                # API Keys (Keep Private!)
├── .gitignore          # Git exclusion rules
├── uploads/            # Stored PDF files
└── chroma_db/          # Vector Database files

Setup Instructions
1. Clone the Repository
Bash

git clone [https://github.com/OmMadekar91/AI-KnowledgeHUB.git](https://github.com/OmMadekar91/AI-KnowledgeHUB.git)
cd AI-KnowledgeHUB

2. Install Dependencies
Bash

pip install -r requirements.txt
3. Environment Configuration
Create a .env file in the root directory and add your Mistral API key:

Code snippet

MISTRAL_API_KEY=your_actual_key_here

4. Run the Application
Bash

python app.py
The application will automatically open at http://127.0.0.1:5000