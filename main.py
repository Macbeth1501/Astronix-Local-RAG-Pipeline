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
from langchain_community.embeddings import FastEmbedEmbeddings

# --- CONFIGURATION ---
WHISPER_DEVICE = "cpu"       
WHISPER_COMPUTE = "int8"
LLM_MODEL = "llama3.2:1b"
DB_PATH = "./technex_db"
KEYWORDS = "Technex, Ravindra Singhal, Rochan Awasthi, Hackathon, Neo-Celestia"

# The Persona
SYSTEM_PROMPT = """
You are Astronix, the AI mascot for Technex 2025.
Use the provided Context to answer the student's question.
If the answer is not in the context, say "I don't have that info."
Keep your answers SHORT (1-2 sentences max) and enthusiastic!
If the user asks for "Technics", assume they mean "Technex".
"""

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

def main():
    print("\n🚀 INITIALIZING ASTRONIX 1.0...")
    
    # --- LOAD EARS ---
    print("   EARS: Loading Whisper...")
    ear_model = WhisperModel("base.en", device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    
    # --- LOAD MEMORY ---
    print("   MEMORY: Loading ChromaDB...")
    embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    
    print("\n✅ SYSTEM ONLINE! (Press and HOLD 's' to speak)")
    print("--------------------------------------------------")

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
            # --- SOCIAL FILTER (The Fix) ---
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

            # 4. RETRIEVE
            results = db.similarity_search(user_text, k=3)
            context_text = "\n".join([doc.page_content for doc in results])
            
            # 5. THINK
            final_prompt = f"Context: {context_text}\n\nQuestion: {user_text}"
            
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': final_prompt}
                ]
            )
            
            ai_answer = response['message']['content']
            
            # 6. SPEAK
            speak(ai_answer)

if __name__ == "__main__":
    main()