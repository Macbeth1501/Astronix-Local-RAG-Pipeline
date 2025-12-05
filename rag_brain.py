import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

DB_PATH = "./technex_db"
MODEL_NAME = "llama3.2:1b"

# Simpler Prompt - Less restrictive, encourages using the text
SYSTEM_PROMPT = """
You are Astronix, the AI mascot for Technex 2025.
You are given a snippet of text from the official event manual (Context).
Your job is to answer the student's question ONLY using that text.
Do not say "I don't have info" if the answer is clearly in the Context.
Be direct and helpful.
"""

def main():
    print("⏳ Loading Memory...")
    embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    print("✅ System Ready! (Type 'exit' to quit)")
    print("------------------------------------------------")

    while True:
        query = input("\nStudent: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Retrieve Context
        results = db.similarity_search(query, k=3)
        context_text = "\n---\n".join([doc.page_content for doc in results])
        
        # *** DEBUGGING PRINT ***
        # This shows us what the AI is actually reading
        print(f"\n🔍 [DEBUG: FOUND IN DB]\n{context_text}\n--------------------")

        final_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        print("🤖 Astronix: ", end="", flush=True)
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': final_prompt}
            ],
            stream=True,
        )

        for chunk in stream:
            print(chunk['message']['content'], end="", flush=True)
        print("\n")

if __name__ == "__main__":
    main()