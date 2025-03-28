import os
import subprocess
import fitz # PyMuPDF for PDF text extraction
from flask import Flask, render_template, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from deep_translator import GoogleTranslator
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --------------------------
# Configuration
# --------------------------
PDF_FOLDER = "pdf_folder" # Ensure this folder exists and contains your PDFs
ALLOWED_EXTENSIONS = {"pdf"}
ALLOWED_MODELS = ["llama3.2", "phi4", "gemma3:12B", "llama3.3"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------------
# PDF Extraction & Processing
# --------------------------

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF and return it as a string."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def process_folder(folder_path):
    """Extract text from all PDFs in a folder and return a list of (filename, text) tuples."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            documents.append((filename, text))
    return documents

def split_text(documents, chunk_size=500, chunk_overlap=50):
    """Split text from multiple documents into smaller chunks with source metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    doc_chunks = []
    for filename, text in documents:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc_chunks.append(Document(page_content=chunk, metadata={"source": filename}))
    return doc_chunks

def create_vector_db(doc_chunks):
    """Create a FAISS vector database from document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(doc_chunks, embeddings)
    return vector_db

def retrieve_context(vector_db, query, k=3):
    """Retrieve the top-k most relevant text chunks for the query."""
    return vector_db.similarity_search(query, k=k)

# --------------------------
# Build/Update Vector Database at Startup
# --------------------------
print("Extracting text from PDFs...")
documents = process_folder(PDF_FOLDER)
print("Splitting text into chunks...")
doc_chunks = split_text(documents)
print("Creating vector database...")
vector_db = create_vector_db(doc_chunks)
print("PDF processing complete. Ready to answer queries.")

# --------------------------
# Query Answering
# --------------------------

def generate_with_ollama(prompt, model="llama3.2"):
    """Generate a response using the Ollama CLI with proper encoding."""
    command = ["ollama", "run", model, prompt]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace', # replace undecodable characters
            check=True
        )
        return result.stdout.strip() if result.stdout is not None else ""
    except subprocess.CalledProcessError as e:
        print("Error running Ollama:", e)
        return "Error generating response."

# --------------------------
# Flask Routes
# --------------------------

@app.route("/")
def index():
    # Pass the allowed models to the template
    return render_template("index.html", allowed_models=ALLOWED_MODELS)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    selected_lang = data.get("language", "en")
    history = data.get("history", []) # conversation history list
    model = data.get("model", "llama3.2")
    
    # Validate model; default if invalid
    if model not in ALLOWED_MODELS:
        model = "llama3.2"

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Build conversation history text (assumed in English)
    history_text = ""
    for turn in history:
        history_text += f"User: {turn.get('query', '')}\nAssistant: {turn.get('englishAnswer', '')}\n"
    
    # Retrieve relevant context from the vector database
    retrieved_docs = retrieve_context(vector_db, query)
    context = "\n".join(
        [f"[Source: {doc.metadata['source']}] {doc.page_content}" for doc in retrieved_docs]
    )

    # Build prompt including conversation history and context
    prompt = (
        "You are a knowledgeable police officer of the Tamil Nadu government with access to multiple PDF documents. "
        "Review the conversation history below along with the provided context, then answer the current query. "
        "Answer in bullet points (each starting with a number) and maintain continuity.\n\n"
        f"Conversation History:\n{history_text}\n"
        f"Context:\n{context}\n\n"
        f"Current Query: {query}\n\n"
        "Answer:"
    )

    english_answer = generate_with_ollama(prompt, model=model)
    print("English Answer:", english_answer)

    answer = english_answer
    if selected_lang == "ta":
        answer = GoogleTranslator(source="en", target="ta").translate(english_answer)

    return jsonify({
        "answer": answer, 
        "english_answer": english_answer,
        "context": context
    })

@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.get_json()
    text = data.get("text", "")
    target = data.get("target", "en")
    if target == "ta":
        translated = GoogleTranslator(source="en", target="ta").translate(text)
        return jsonify({"translated": translated})
    else:
        return jsonify({"translated": text})

# --------------------------
# File Upload Route
# --------------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    new_docs = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(PDF_FOLDER, filename)
            file.save(file_path)
            print(f"Uploaded and saved: {filename}")
            text = extract_text_from_pdf(file_path)
            new_doc_chunks = split_text([(filename, text)])
            new_docs.extend(new_doc_chunks)
        else:
            return jsonify({"error": f"File {file.filename} not allowed."}), 400

    if new_docs:
        vector_db.add_documents(new_docs)
        return jsonify({"message": "Files uploaded and processed successfully."})
    else:
        return jsonify({"error": "No valid PDF files found."}), 400

if __name__ == "__main__":
    app.run(debug=True)
