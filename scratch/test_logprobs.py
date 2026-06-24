from openai import OpenAI
import numpy as np

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def test_logprobs():
    try:
        response = client.chat.completions.create(
            model="gemma3:4b",
            messages=[{"role": "user", "content": "Explain 2+2."}],
            max_tokens=5,
            logprobs=True,
            top_logprobs=1
        )
        print("Logprobs result:")
        print(response.choices[0].logprobs)
        return True
    except Exception as e:
        print(f"Logprobs not supported or failed: {e}")
        return False

if __name__ == "__main__":
    test_logprobs()
