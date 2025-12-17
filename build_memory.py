import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 1. CHANGED: Import Ollama instead of FastEmbed
from langchain_community.embeddings import OllamaEmbeddings

# Configuration
DATA_FILE = "technex_data.txt"
DB_PATH = "./technex_db"

def build_database():
    # 1. Check if data exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found!")
        return

    # PRE-CLEANUP: Automatically delete old DB to prevent dimension errors
    if os.path.exists(DB_PATH):
        print(f"🧹 Deleting old database at {DB_PATH} to avoid conflicts...")
        shutil.rmtree(DB_PATH)

    print("📄 Loading data...")
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    documents = loader.load()

    # 2. Split text into chunks
    print("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,    
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Created {len(chunks)} chunks.")

    # 3. Create Vector Store
    print("🧠 Embedding and storing in ChromaDB (using Local Ollama)...")
    
    # 2. CHANGED: Use Ollama with the Nomic model (Size 768)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # This creates the DB folder
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print(f"✅ Success! Database created at '{DB_PATH}'")

if __name__ == "__main__":
    build_database()