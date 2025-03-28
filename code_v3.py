import os
import subprocess
import fitz  # PyMuPDF for PDF text extraction
from flask import Flask, render_template, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from deep_translator import GoogleTranslator

app = Flask(__name__)

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
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            documents.append((filename, text))
    return documents

def split_text(documents, chunk_size=500, chunk_overlap=50):
    """Split text from multiple documents into smaller chunks, adding the PDF source as metadata."""
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
            errors='replace',  # replaces undecodable characters
            check=True
        )
        return result.stdout.strip() if result.stdout is not None else ""
    except subprocess.CalledProcessError as e:
        print("Error running Ollama:", e)
        return "Error generating response."

# --------------------------
# Pre-Process PDFs and Build Vector DB at Startup
# --------------------------
PDF_FOLDER = "pdf_folder"  # Make sure this folder exists and contains your PDFs
print("Extracting text from PDFs...")
documents = process_folder(PDF_FOLDER)
print("Splitting text into chunks...")
doc_chunks = split_text(documents)
print("Creating vector database...")
vector_db = create_vector_db(doc_chunks)
print("PDF processing complete. Ready to answer queries.")

# --------------------------
# Flask Routes
# --------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    selected_lang = data.get("language", "en")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Retrieve relevant context from the vector database
    retrieved_docs = retrieve_context(vector_db, query)
    context = "\n".join(
        [f"[Source: {doc.metadata['source']}] {doc.page_content}" for doc in retrieved_docs]
    )

    # Build the prompt
    prompt = (
        "You are a knowledgeable police of the tamilnadu government with access to multiple PDF documents. "
        "Please carefully review the provided context (including source information) and answer the following query, try to give the results in points and make the answer neately formated after every point give a new line character /n "
        "showing understanding and continuity. "
        "Please answer in the same language as the query.\n\n"
        f"Context:\n{context}\n\n"
        f"Current Query: {query}\n\n"
        "Answer:"
    )

    # Always generate the answer in English first
    english_answer = generate_with_ollama(prompt)

    # If Tamil is selected, translate the answer
    answer = english_answer
    if selected_lang == "ta":
        answer = GoogleTranslator(source="en", target="ta").translate(english_answer)

    return jsonify({"answer": answer, "english_answer": english_answer})

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

if __name__ == "__main__":
    app.run(debug=True)
