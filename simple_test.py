import os
import time
from src.eigen_memory_agent.agent import AgenticMemoryLoop

def main():
    print("Initializing AgenticMemoryLoop...")
    # Using the DB string from docker-compose
    db_string = "postgresql://postgres:password@localhost:5432/memory_agent"
    
    # Wait for DB to be ready potentially, though loop handles connection
    
    try:
        # Initialize agent
        # Note: We rely on vLLM being up. It might take a moment to load the model.
        agent = AgenticMemoryLoop(db_string)
        print("Agent initialized.")
        
        query = "What happens if I try to divide by zero in Python?"
        print(f"Running query: {query}")
        
        response = agent.run(query)
        print("Response received:")
        print(response)
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
