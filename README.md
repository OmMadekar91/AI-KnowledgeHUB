KnowledgeHub AI | Enterprise Searchable Knowledge Base
KnowledgeHub AI is an advanced Retrieval-Augmented Generation (RAG) platform designed to transform static enterprise PDF documents into a searchable, interactive intelligence repository. By leveraging high-performance embeddings and a structured LLM response engine, it provides precise, documented answers from complex datasets.

🚀 Key Features
Intelligent Ingestion: Automatically processes PDFs, extracts text via PyPDFLoader, and optimizes retrieval using RecursiveCharacterTextSplitter.

Vectorized Search: Utilizes ChromaDB for high-speed similarity searches across document embeddings.

Structured Intelligence: Unlike standard chatbots, this system returns a structured schema including high-level summaries, detailed answers, key bullet points, and confidence scores.

Source Attribution: Every answer identifies the specific source file and page number from which the information was retrieved.

Enterprise Dashboard: A modern, responsive UI built for real-time data ingestion and system analytics.

🛠️ Technical Architecture
The system follows a modular RAG pipeline:

Data Layer: PDF ingestion and persistent vector storage in ChromaDB.

Inference Layer: Powered by Google Gemini 2.5 Flash (for reasoning) and Text-Embedding-004 (for semantic search).

API Layer: A Flask-based REST API manages communication between the RAG logic and the UI.

Frontend Layer: A sleek, interactive dashboard using Plus Jakarta Sans and FontAwesome for a professional enterprise feel.

⚙️ Installation & Setup
Prerequisites
Python 3.9+

Google Gemini API Key

Environment Configuration
Create a .env file in the root directory:

Code snippet

GEMINI_API_KEY=your_api_key_here
Installation
Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:

Bash

pip install flask flask-cors python-dotenv langchain langchain-google-genai langchain-chroma langchain-community pypdf pydantic
Run the application:

Bash

python app.py
The system will automatically open your default browser to http://127.0.0.1:5000.

📂 Project Structure
app.py: Backend logic, AI chain configuration, and API endpoints.

templates/site.html: The frontend dashboard and UI logic.

/uploads: Local storage for ingested PDF documents.

/chroma_db: Persistent vector database storage.

🛠 Technical Challenges & Solutions
1. Handling Unstructured PDF Data
Challenge: Enterprise PDFs often contain complex layouts that can break semantic meaning when split into chunks.

Solution: Implemented RecursiveCharacterTextSplitter with a chunk_size of 1000 and a chunk_overlap of 150. This ensures that context is preserved across chunks, preventing the AI from losing the "thread" of a paragraph.

2. Ensuring Model Hallucination Control
Challenge: LLMs can sometimes provide generic answers or "hallucinate" information not present in the uploaded documents.

Solution: I utilized a Structured Output approach using Pydantic. By defining a strict EnterpriseKnowledgeResponse schema (Summary, Details, Key Points, and Confidence Score), the model is forced to categorize its findings based only on the retrieved context.

3. Real-time Response Monitoring
Challenge: Users need to know if a response is reliable.

Solution: The system calculates a confidence_score and displays the specific Source File and Page Number for every answer. This creates a "transparent" AI that allows users to verify information instantly.

4. Cross-Platform Environment Management
Challenge: Managing API keys and file paths across different operating systems can lead to "Path not found" errors.

Solution: Used pathlib for OS-independent path handling and python-dotenv for secure environment variable management.