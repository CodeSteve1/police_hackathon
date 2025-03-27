import fitz # PyMuPDF for PDF text extraction
import subprocess
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF and return it as a string."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def process_folder(folder_path):
    """Extract text from all PDFs in a folder and return a list of Documents."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            text = extract_text_from_pdf(pdf_path)
            documents.append((filename, text))
    return documents

def split_text(documents, chunk_size=500, chunk_overlap=50):
    """Split text from multiple documents into smaller chunks, 
    adding the PDF source as metadata."""
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

def generate_with_ollama(prompt, model="phi4"):
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
        return ""

def build_conversation_history(history):
    """Build a formatted conversation history string from the history list."""
    history_lines = []
    for turn in history:
        history_lines.append(f"User: {turn['query']}")
        history_lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(history_lines)

def main():
    folder_path = "pdf_folder" # Change this to your folder containing PDFs

    print("Extracting text from PDFs...")
    documents = process_folder(folder_path)

    print("Splitting text into chunks...")
    doc_chunks = split_text(documents)

    print("Creating vector database...")
    vector_db = create_vector_db(doc_chunks)

    print("All PDFs processed. You can now ask questions!")
    
    # This will store the previous conversation history
    conversation_history = []

    while True:
        query = input("\nEnter your query (type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting conversation. Goodbye!")
            break

        retrieved_docs = retrieve_context(vector_db, query)
        # Build a context string with source info and content snippets
        context = "\n".join(
            [f"[Source: {doc.metadata['source']}] {doc.page_content}" for doc in retrieved_docs]
        )
        
        # Build the conversation history string
        history_text = build_conversation_history(conversation_history)
        if history_text:
            history_text = "Previous conversation:\n" + history_text + "\n\n"
        
        # Enhanced prompt including conversation history, retrieved context, and current query.
        # Instruction added to answer in the same language as the query.
        prompt = (
            "You are a knowledgeable assistant with access to multiple PDF documents. "
            "Please carefully review the provided context (including source information) and the previous conversation, "
            "and then answer the following query in a way that shows understanding and continuity. "
            "Please answer in the same language as the query.\n\n"
            f"{history_text}"
            f"Context:\n{context}\n\n"
            f"Current Query: {query}\n\n"
            "Answer:"
        )

        answer = generate_with_ollama(prompt)
        print("\nAnswer from Ollama:")
        print(answer)
        
        # Save the current turn in the conversation history
        conversation_history.append({
            "query": query,
            "answer": answer
        })

if __name__ == "__main__":
    main()
