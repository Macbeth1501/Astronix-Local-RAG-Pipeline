import sounddevice as sd
import soundfile as sf
import numpy as np

# Configuration
DURATION = 5  # Recording duration in seconds
SAMPLE_RATE = 16000  # Whisper prefers 16kHz
CHANNELS = 1  # Mono audio is enough for voice
OUTPUT_FILE = "test_recording.wav"

def record_audio():
    print("---------------------------------------")
    print(f"🎤 Recording for {DURATION} seconds... Speak now!")
    
    # This line captures audio from your default microphone
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), 
                        samplerate=SAMPLE_RATE, 
                        channels=CHANNELS, 
                        dtype='float32')
    
    sd.wait()  # Wait until the recording is finished
    
    print("✅ Recording complete.")
    
    # Save the recorded data as a WAV file
    sf.write(OUTPUT_FILE, audio_data, SAMPLE_RATE)
    print(f"💾 Saved to: {OUTPUT_FILE}")
    print("---------------------------------------")

if __name__ == "__main__":
    record_audio()