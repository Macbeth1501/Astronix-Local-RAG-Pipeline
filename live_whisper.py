import keyboard
import sounddevice as sd
import numpy as np
import time
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
WHISPER_DEVICE = "cpu" 
WHISPER_COMPUTE = "int8"

# *** KEYWORD FIX ***
# Add all your special college words here in a sentence format.
# This guides the model's vocabulary.
CUSTOM_KEYWORDS = "Technex, Zenith, Rochan, Astronix, HOD, Pallotti ,CSE Department. "

def main():
    # 1. LOAD MODEL
    print("⏳ Loading Whisper Model... (Please wait)")
    model = WhisperModel("base.en", device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print("✅ Model Loaded! System Ready.")
    print(f"ℹ️  Context Hints Active: {CUSTOM_KEYWORDS}")
    print("--------------------------------------------------")

    while True:
        print("\n🔵 Press 's' to START listening...")
        keyboard.wait('s') 
        time.sleep(0.2) 
        
        print("🔴 RECORDING... (Press 's' to STOP)")
        
        recorded_audio = []

        def callback(indata, frames, time, status):
            recorded_audio.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            keyboard.wait('s')
        
        print("✅ Processing...")
        
        if len(recorded_audio) > 0:
            audio_np = np.concatenate(recorded_audio, axis=0)
            audio_np = audio_np.flatten().astype(np.float32)

            # *** HERE IS THE FIX ***
            # We pass 'initial_prompt' to guide the transcription
            segments, _ = model.transcribe(
                audio_np, 
                beam_size=5, 
                initial_prompt=f"The following is a discussion about {CUSTOM_KEYWORDS}"
            )
            
            print("\n📝 TRANSCRIPTION:")
            full_text = ""
            for segment in segments:
                print(f"👉 {segment.text}")
                full_text += segment.text + " "
            
            # Fallback: Hard replace if the prompt hint isn't 100% perfect
            # This is a common "safety net" in production ASR
            full_text = full_text.replace("Technics", "Technex")
            # print(f"\n(Corrected): {full_text}")

        else:
            print("⚠️ No audio recorded.")
            
        print("--------------------------------------------------")
        time.sleep(0.5) 

if __name__ == "__main__":
    main()