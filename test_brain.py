import ollama

# Configuration
MODEL = "llama3.2:1b"

# *** HERE IS THE IDENTITY ***
SYSTEM_PROMPT = """
You are Astronix, the official in-house AI mascot for Technex.
Your mission is to help students and visitors with queries related to the Technex event.
Always be enthusiastic, futuristic, and helpful.
If asked who you are, introduce yourself as Astronix, the guide for Technex.
Keep your answers concise and energetic.
"""

def think(prompt):
    print(f"\n🧠 Thinking about: '{prompt}'...\n")
    print("🤖 Astronix: ", end="", flush=True)

    stream = ollama.chat(
        model=MODEL,
        # We send the System Prompt FIRST, then the User Prompt
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt}
        ],
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        part = chunk['message']['content']
        print(part, end="", flush=True)
        full_response += part
    
    print("\n\n✅ Done.")

if __name__ == "__main__":
    # Test Identity
    user_input = "Who are you and what is this event?"
    think(user_input)