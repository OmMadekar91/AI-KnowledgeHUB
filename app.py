import os
import logging
import webbrowser 
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from dotenv import load_dotenv

#1. Configuration for cross-platforms.
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnterpriseRAG")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
DB_FOLDER = BASE_DIR / "chroma_db"

# Ensure directories exist.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

# Core LangChain and Google GenAI imports.
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup of Gemini-API key.
API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = API_KEY

app = Flask(__name__)
CORS(app)

#2. for automation of backend and frontend.
@app.route("/")
def index():
    """Serves the site.html file from the templates folder."""
    try:
        return render_template("site.html")
    except Exception as e:
        return f"Error: Could not find site.html inside the templates folder. {str(e)}", 404

#3. For structured output of AI search.
class EnterpriseKnowledgeResponse(BaseModel):
    summary: str = Field(description="A 1-2 sentence high-level summary of the answer.")
    detailed_answer: str = Field(description="The comprehensive answer derived from the PDF context.")
    key_points: List[str] = Field(description="3-5 bullet points highlighting critical facts.")
    confidence_score: float = Field(description="A value between 0 and 1 indicating certainty.")

#4. Model selection for AI search.
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
base_llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
structured_llm = base_llm.with_structured_output(EnterpriseKnowledgeResponse)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf_file")
    if not file: return "No file", 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(data)

        Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=str(DB_FOLDER)
        )
        return jsonify({"message": f"Successfully ingested {len(chunks)} chunks."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def ai_search():
    query = request.json.get("q")
    if not query: return jsonify({"error": "No query provided"}), 400

    vector_db = Chroma(persist_directory=str(DB_FOLDER), embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5}) 

    try:
        docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        system_prompt = (
            "You are a professional enterprise assistant. Use the provided context to answer. "
            "Strictly follow the output schema.\n\nContext:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        chain = prompt | structured_llm
        result = chain.invoke({"context": context_text, "input": query})
        
        sources = [
            {"file": os.path.basename(doc.metadata.get("source", "Unknown")), 
             "page": doc.metadata.get("page", 0) + 1}
            for doc in docs
        ]

        return jsonify({
            "structured_answer": {
                "summary": result.summary,
                "details": result.detailed_answer,
                "key_points": result.key_points,
                "confidence": result.confidence_score
            },
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": f"Search Error: {str(e)}"}), 500

@app.route("/files", methods=["GET"])
def list_files():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
    return jsonify({"files": files})

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"message": f"'{filename}' deleted."})
        return jsonify({"error": "File not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#5. Automation of browser.
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)