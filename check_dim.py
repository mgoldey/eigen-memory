import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

try:
    res = client.embeddings.create(input="Current embedding dimension test", model="embeddinggemma:latest")
    vec = res.data[0].embedding
    print(f"Dimension: {len(vec)}")
except Exception as e:
    print(f"Error: {e}")
