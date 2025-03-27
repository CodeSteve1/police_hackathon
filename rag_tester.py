import fitz  # PyMuPDF for PDF text extraction
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def extract_text_from_pdf(pdf_path):
    """Extract text from each page of the PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def split_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into smaller overlapping chunks using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def create_vector_db(chunks):
    """
    Create a FAISS vector database from text chunks.
    This version uses HuggingFaceEmbeddings with the 'all-MiniLM-L6-v2' model,
    which is fully local and doesn't require an API key.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db

def retrieve_context(vector_db, query, k=3):
    """
    Retrieve the top-k most relevant text chunks for the given query.
    Returns a list of Document objects.
    """
    return vector_db.similarity_search(query, k=k)

def generate_with_ollama(prompt, model="llama3.2"):
    """
    Generate a response using the Ollama CLI with the specified model.
    Make sure the Ollama CLI is installed and the 'llama3.2' model is available.
    """
    command = ["ollama", "run", model, prompt]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Error running Ollama:", e)
        return ""

def main():
    # Change this to the path of your PDF file
    pdf_path = "constitution.pdf"
    
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    #print(text)
    
    print("Splitting text into chunks...")
    chunks = split_text(text)
    print(chunks)
    
    print("Creating vector database...")
    vector_db = create_vector_db(chunks)
    
    print("PDF processed. You can now ask questions about the document!")
    
    # Conversation loop: continue until the user types 'exit' or 'quit'
    while True:
        query = input("\nEnter your query (type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting conversation. Goodbye!")
            break
        
        # Retrieve relevant context chunks for the query
        retrieved_docs = retrieve_context(vector_db, query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Build the prompt by combining the context with the query
        prompt = (
            f"Based on the following context, answer the query:\n\n"
            f"Context:\n{context}\n\n"
            f"Query: {query}\n\n"
            f"Answer:"
        )
        
        # Generate the answer using Ollama with the llama3.2 model
        answer = generate_with_ollama(prompt)
        print("\nAnswer from Ollama:")
        print(answer)

if __name__ == "__main__":
    main()
