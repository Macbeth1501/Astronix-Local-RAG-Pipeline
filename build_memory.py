import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

# Configuration
DATA_FILE = "technex_data.txt"
DB_PATH = "./technex_db"

def build_database():
    # 1. Check if data exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found!")
        return

    print("📄 Loading data...")
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    documents = loader.load()

    # 2. Split text into chunks (AI can't read whole books at once)
    print("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # Characters per chunk
        chunk_overlap=50    # Overlap to keep context
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Created {len(chunks)} chunks.")

    # 3. Create Vector Store
    print("🧠 Embedding and storing in ChromaDB (this runs on CPU)...")
    # We use FastEmbed (lightweight, no GPU needed)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # This creates the DB folder
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print(f"✅ Success! Database created at '{DB_PATH}'")

if __name__ == "__main__":
    build_database()