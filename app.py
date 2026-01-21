import os
import shutil
import logging
import webbrowser
from pathlib import Path
from threading import Timer
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

#AI & Vector Search Libraries
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# Setup and directories
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

BASE_PATH = Path(__file__).resolve().parent
PDF_STORAGE = BASE_PATH / "uploads"
VECTOR_DB_DIR = BASE_PATH / "chroma_db"

PDF_STORAGE.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("ERROR: Please add MISTRAL_API_KEY to your .env file!")
    exit()

app = Flask(__name__)
CORS(app)

class AIResponseSchema(BaseModel):
    summary: str = Field(description="The direct, short answer to the user's question. Max 15 words.")
    key_points: List[str] = Field(description="List of 3-5 specific facts relevant to the query.")

# Initialize global models & database 
print("Loading Local Embedding Model...")
local_embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/clip-ViT-B-32")

vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR), 
    embedding_function=local_embed_model
)

# Mistral AI model selection 
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
    with open(BASE_PATH / "site.html", "r", encoding="utf-8") as f:
        return render_template_string(f.read())

@app.route("/upload", methods=["POST"])
def process_pdf():
    try:
        uploaded_file = request.files.get("pdf_file")
        if not uploaded_file:
            return jsonify({"error": "No file uploaded"}), 400
        
        save_path = PDF_STORAGE / uploaded_file.filename
        uploaded_file.save(str(save_path))
        
        # Load and Split
        pdf_loader = PyPDFLoader(str(save_path))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        content_chunks = text_splitter.split_documents(pdf_loader.load())
        
        vector_store.add_documents(content_chunks)
        
        return jsonify({"message": f"Successfully learned from: {uploaded_file.filename}"})
    
    except Exception as err:
        logging.error(f"Upload Crash: {err}")
        return jsonify({"error": str(err)}), 500

@app.route("/search", methods=["POST"])
def ask_question():
    user_query = request.json.get("q")
    try:
        # Use the global vector_store
        matched_docs = vector_store.as_retriever(search_kwargs={"k": 6}).invoke(user_query)
        context_text = "\n\n".join([doc.page_content for doc in matched_docs])
        
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precision data extractor. 
            Answer strictly a one liner using the provided context: {context}
            
            Rules:
            1. If the user asks for a specific value (like a PRN, Date, or Grade), provide simple line with value in the summary.
            2. Do not explain what the document is.
            3. Do not use filler phrases.
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
    try:
        # Clear the documents from the collection
        ids = vector_store.get()['ids']
        if ids:
            vector_store.delete(ids=ids)
            
        # Clean up files
        for item in PDF_STORAGE.iterdir():
            if item.is_file():
                item.unlink()
        return jsonify({"message": "Knowledge base cleared!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/files", methods=["GET"])
def get_file_list():
    files = [f.name for f in PDF_STORAGE.iterdir() if f.is_file()]
    return jsonify({"files": files})

# Running the application
def launch_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Open the browser automatically
    Timer(1.5, launch_browser).start()
    # Debug=False for final presentation stability
    app.run(port=5000, debug=True, use_reloader=False)
