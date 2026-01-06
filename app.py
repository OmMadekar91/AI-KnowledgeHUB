import os
import shutil
import logging
import webbrowser
from pathlib import Path
from threading import Timer
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv
# AI & Vector Search Libraries 
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# Setup and Directories
load_dotenv()
# Basic logging to see what's happening in the terminal
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BASE_PATH = Path(__file__).resolve().parent
PDF_STORAGE = BASE_PATH / "uploads"
VECTOR_DB_DIR = BASE_PATH / "chroma_db"

# Create folders if they don't exist
PDF_STORAGE.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# API key calling
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("ERROR: Please add MISTRAL_API_KEY to your .env file!")
    exit()

app = Flask(__name__)
CORS(app)

# Defining the strucuted output (Pydantic Model)
class AIResponseSchema(BaseModel):
    summary: str = Field(description="The direct, short answer to the user's question. Max 15 words.")
    key_points: List[str] = Field(description="List of 3-5 specific facts relevant to the query.")

print("Loading Local CLIP Model")
# Local embeddings
local_embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/clip-ViT-B-32")

# Mistral model selection 
ai_brain = ChatMistralAI(
    model="mistral-large-latest", 
    temperature=0,
    api_key=MISTRAL_API_KEY
)
# Connect the brain to our structured output
smart_llm = ai_brain.with_structured_output(AIResponseSchema)

# Backend Logic

@app.route("/")
def home_page():
    """Serves the frontend site.html file"""
    with open(BASE_PATH / "site.html", "r", encoding="utf-8") as f:
        return render_template_string(f.read())

@app.route("/upload", methods=["POST"])
def process_pdf():
    "Handles PDF upload and Vector DB indexing"
    try:
        uploaded_file = request.files.get("pdf_file")
        if not uploaded_file:
            return jsonify({"error": "No file uploaded"}), 400
            
        save_path = PDF_STORAGE / uploaded_file.filename
        uploaded_file.save(str(save_path))
        pdf_loader = PyPDFLoader(str(save_path))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        content_chunks = text_splitter.split_documents(pdf_loader.load())
        
        # Reseting DB to remove old data issues
        if VECTOR_DB_DIR.exists():
            shutil.rmtree(VECTOR_DB_DIR)
            VECTOR_DB_DIR.mkdir()

        # Creating the searchable database
        Chroma.from_documents(
            documents=content_chunks, 
            embedding=local_embed_model, 
            persist_directory=str(VECTOR_DB_DIR)
        )
        
        return jsonify({"message": f"Successfully learned from: {uploaded_file.filename}"})
    
    except Exception as err:
        logging.error(f"Upload Crash: {err}")
        return jsonify({"error": str(err)}), 500

@app.route("/search", methods=["POST"])
def ask_question():
    """Searches the PDF and returns Key Findings"""
    user_query = request.json.get("q")
    try:
        # Connecting to existing database
        vector_store = Chroma(persist_directory=str(VECTOR_DB_DIR), embedding_function=local_embed_model)
        
        matched_docs = vector_store.as_retriever(search_kwargs={"k": 4}).invoke(user_query)
        context_text = "\n\n".join([doc.page_content for doc in matched_docs])
        
        #instructions for relevant data extraction
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precision data extractor. 
            Answer strictly a one liner using the provided context: {context}
            
            Rules:
            1. If the user asks for a specific value (like a PRN, Date, or Grade), provide simple line with value in the summary.
            2. Do not explain what the document is just explain when asked. 
            3. Do not use filler phrases like 'The document mentions...' or 'Based on the context...'.
            4. In key_points, list only 4-5 high-impact relevant facts."""),
            ("human", "{input}"),
        ])
        
        final_answer = (chat_prompt | smart_llm).invoke({"context": context_text, "input": user_query})
        source_info = [{"file": Path(doc.metadata.get("source")).name, "page": doc.metadata.get("page", 0) + 1} for doc in matched_docs]
        
        return jsonify({
            "structured_answer": {
                "summary": final_answer.summary,
                "key_points": final_answer.key_points
            },
            "sources": source_info
        })
        
    except Exception as err:
        logging.error(f"Query Error: {err}")
        return jsonify({"error": "Failed to generate answer"}), 500

@app.route('/wipe_all', methods=['POST'])
def reset_system():
    """Clears all stored data and memory"""
    shutil.rmtree(PDF_STORAGE, ignore_errors=True)
    shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
    PDF_STORAGE.mkdir(exist_ok=True)
    VECTOR_DB_DIR.mkdir(exist_ok=True)
    return jsonify({"message": "All data wiped successfully!"})

@app.route("/files", methods=["GET"])
def get_file_list():
    """Lists all uploaded documents"""
    files = [f.name for f in PDF_STORAGE.iterdir() if f.is_file()]
    return jsonify({"files": files})
    
def launch_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Open the browser automatically
    Timer(1.5, launch_browser).start()
    # Debug=False for final presentation stability
    app.run(port=5000, debug=False)
