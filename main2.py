import os
import time
import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import ollama
import pyttsx3
from faster_whisper import WhisperModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- CONFIGURATION ---
WHISPER_DEVICE = "cpu"       
WHISPER_COMPUTE = "int8"
#LLM_MODEL = "llama3.2:1b" #smaller model| we will use 8b model
LLM_MODEL = "llama3.1"
# 2. NEW: Define the embedding model name
EMBEDDING_MODEL = "nomic-embed-text"
DB_PATH = "./technex_db"
KEYWORDS = "Technex,  Rochan Awasthi, Hackathon, Neo-Celestia, Synergy Sphere, St. Vincent Pallotti College of Engineering & Technology"

# 1. NEW: Triggers that force the bot to look into the Database
DOMAIN_TRIGGERS = [
    "technex", "technics","techniques" ,"pallotti", "college", "svpcet", "nagpur",
    "event", "competition", "hackathon", "gameathon", "overdrive", 
    "vortex", "envision", "mind2market", "coastal clash", "robo sumo", 
    "drift", "gamers conquest", "scrutinizing", "techquest", "designx",
    "synergy sphere", "rochan", "shankar", "kartik", "gunjan", "vaibhav",
    "guest", "prize", "fee", "register", "date", "venue", "location",
    "coordinator", "hod", "department", "schedule", "rule", "winner",
    "list", "many events", "all events" ,"Student " , "Coordinator"
]

# The Persona
RAG_SYSTEM_PROMPT = """
You are Astronix, the AI mascot for Technex 2025.
Use the provided Context to answer the student's question.
If the answer is not in the context, say "I don't have that info."
Keep your answers SHORT and ENTHUSIASTIC.
If the user asks for "Technics", assume they mean "Technex".
If a user mentions Techniques or techniques, interpret it as a reference to Technex.
"""

GENERAL_SYSTEM_PROMPT = """
You are Astronix, a helpful AI assistant.
1. The user is asking a general knowledge question (not about Technex).
2. Answer based on your own general knowledge.
3. Keep it SHORT, friendly, and smart.
"""

#Keep your answers SHORT (1-2 sentences max) and enthusiastic!
def speak(text):
    """
    The Mouth: Optimized to prevent freezing.
    We initialize a NEW engine instance every time to avoid the loop bug.
    """
    try:
        print(f"🗣️ Astronix: {text}")
        
        # Re-initialize engine each time to prevent 'stuck' loop
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)    
        engine.setProperty('volume', 1.0)
        
        # Clean up text (remove asterisks that confuse TTS)
        clean_text = text.replace("*", "").replace("#", "")
        
        engine.say(clean_text)
        engine.runAndWait()
        
        # Explicitly stop the engine
        engine.stop()
        del engine # Delete the object to free memory
        
    except Exception as e:
        print(f"❌ Voice Error: {e}")

# 3. NEW: The Router Function
def is_domain_query(text):
    """Checks if the user is asking about the Event/College"""
    text = text.lower()
    for trigger in DOMAIN_TRIGGERS:
        if trigger in text:
            return True
    return False

def main():
    print("\n🚀 INITIALIZING ASTRONIX 1.0...")
    
    # --- LOAD EARS ---
    print("   EARS: Loading Whisper...")
    ear_model = WhisperModel("base.en", device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    
   
    # 3. CHANGED: Initialize OllamaEmbeddings (Runs locally, no HuggingFace download needed)
    print(f"   MEMORY: Loading ChromaDB with {EMBEDDING_MODEL}...")
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Ensure the DB directory exists to avoid errors on first run
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
        print("   ⚠️ WARNING: Database folder not found. Created empty folder.")
        
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    
    print("\n✅ SYSTEM ONLINE! (Press and HOLD 's' to speak)")
    print("--------------------------------------------------")
    speak("Hello! I am Astronix, the AI mascot of Technex 2025. Ask me anything about the fest!")


    while True:
        # 1. WAIT FOR TRIGGER
        print("\n🔵 Ready. Press 's' to speak...")
        keyboard.wait('s') 
        time.sleep(0.1) 
        
        print("🔴 LISTENING... (Release 's' to stop)")
        
        # 2. RECORD AUDIO
        recorded_audio = []
        def callback(indata, frames, time, status):
            recorded_audio.append(indata.copy())

        with sd.InputStream(samplerate=16000, channels=1, callback=callback):
            while keyboard.is_pressed('s'):
                time.sleep(0.1)
        
        print("⏳ Processing...")
        
        # 3. TRANSCRIBE
        if len(recorded_audio) > 0:
            audio_np = np.concatenate(recorded_audio, axis=0).flatten().astype(np.float32)
            
            segments, _ = ear_model.transcribe(
                audio_np, 
                beam_size=5, 
                initial_prompt=f"Context: {KEYWORDS}"
            )
            user_text = " ".join([s.text for s in segments]).strip()
            
            print(f"👉 You said: '{user_text}'")
            
            if len(user_text) < 2:
                continue 
            
            # --- SOCIAL FILTER ---
            # Handles greetings without checking the database
            lower_text = user_text.lower()
            if "thank" in lower_text:
                speak("You are most welcome! Let me know if you have more questions.")
                continue
            elif "hello" in lower_text or "hi" == lower_text:
                speak("Hello there! I am Astronix. Ask me anything about Technex.")
                continue
            elif "bye" in lower_text:
                speak("Goodbye! Hope to see you at the event.")
                break
            # -------------------------------

            # 4. HYBRID ROUTING LOGIC (NEW FEATURE)
            # We decide which "Brain" to use based on the question
            
            if is_domain_query(user_text):
                print("🔍 Mode: RAG (Checking Database...)")
                # -- RAG PATH --
                results = db.similarity_search(user_text, k=3)
                context_text = "\n".join([doc.page_content for doc in results])
                
                # Use the RAG Prompt
                final_system_prompt = RAG_SYSTEM_PROMPT
                final_user_prompt = f"Context: {context_text}\n\nQuestion: {user_text}"
                
            else:
                print("🧠 Mode: GENERAL (Using General Knowledge...)")
                # -- GENERAL PATH --
                # No database search, no context injection
                final_system_prompt = GENERAL_SYSTEM_PROMPT
                final_user_prompt = user_text

            # 5. THINK (Corrected to use variables from Router)
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': final_system_prompt},
                    {'role': 'user', 'content': final_user_prompt}
                ]
            )
            
            ai_answer = response['message']['content']
            
            # 6. SPEAK
            speak(ai_answer)

if __name__ == "__main__":
    main()