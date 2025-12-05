from faster_whisper import WhisperModel
import os

# Define the file we just recorded
audio_file = "test_recording.wav"

if not os.path.exists(audio_file):
    print("❌ Error: test_recording.wav not found! Run test_mic.py first.")
    exit()

print("⏳ Loading Whisper Model on CPU (saving GPU for Llama)...")

# CRITICAL SETTING: device="cpu"
# We force this to run on your i7 Processor so your RTX 3050 is free for the chatbot later.
model = WhisperModel("base.en", device="cpu", compute_type="int8")

print("✅ Model loaded. Transcribing...")

# Transcribe the audio
segments, info = model.transcribe(audio_file, beam_size=5)

print("\n--- TRANSCRIPTION ---")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
print("---------------------")