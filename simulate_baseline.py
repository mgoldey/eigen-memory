import time
import json
from openai import OpenAI
from src.dataset import generate_dataset

# Configuration
MODEL = "gemma3:4b"
# MODEL = "gemma3:4b" # Uncomment if using the user's specific model
API_BASE = "http://localhost:11434/v1"
API_KEY = "ollama"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

def run_baseline(sliding_window_size=5):
    dataset = generate_dataset(num_samples=50) # 50 trials for baseline
    history = [] # List of {"role":, "content":}
    
    score = 0
    results = []

    print(f"Starting Baseline Simulation (N={len(dataset)}) with Sliding Window {sliding_window_size}...")

    for i, item in enumerate(dataset):
        x = item['input']
        true_label = item['label']
        
        # Construct Prompt from History (Sliding Window)
        # We only keep the last N interactions to simulate "Short Term Memory" but no "Long Term retrieval"
        context_messages = history[-(sliding_window_size * 2):] if sliding_window_size > 0 else []
        
        # System Prompt
        messages = [
            {"role": "system", "content": "You are playing a game. Valid outputs are only: RED, BLUE, GREEN. Figure out the rule."}
        ]
        
        # Add Context
        messages.extend(context_messages)
        
        # Current Turn
        user_prompt = f"Input: {x}\nOutput:"
        messages.append({"role": "user", "content": user_prompt})
        
        # Predict
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0, # Deterministic for baseline
                max_tokens=10
            )
            prediction = completion.choices[0].message.content.strip()
            # Normalize prediction (remove extra punctuation if any)
            prediction = prediction.split()[0].upper().replace(".", "")
            
            latency = time.time() - start_time
        except Exception as e:
            print(f"Error on trial {i}: {e}")
            prediction = "ERROR"
            latency = 0
        
        # Evaluate
        is_correct = (prediction == true_label)
        if is_correct:
            score += 1
            
        print(f"Trial {i+1}: Input={x} | Pred={prediction} | True={true_label} | {'CORRECT' if is_correct else 'WRONG'}")
        
        # Update History with Correction (Feedback)
        # The agent sees its own input, and then the *correct* answer as feedback effectively.
        # In a chat loop, we'd say "User: Input X. Asst: Red. User: Correct/Wrong, it was Red."
        # For sliding window optimization, we treat the history as "Perfect Past":
        # Q: Input X -> A: TrueLabel
        # This simulates the user correcting the agent immediately.
        
        history.append({"role": "user", "content": user_prompt})
        history.append({"role": "assistant", "content": true_label})
        
        results.append({
            "trial": i,
            "input": x,
            "prediction": prediction,
            "truth": true_label,
            "correct": is_correct,
            "latency": latency
        })

    accuracy = score / len(dataset)
    print(f"\nFinal Accuracy: {accuracy:.2%}")
    
    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_baseline()
