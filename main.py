import os
from dotenv import load_dotenv
from rag_pipeline.qa_chain import run_query

load_dotenv()

if __name__ == "__main__":
    print("Welcome to the Game Dev RAG Assistant!")
    while True:
        query = input("Ask a game dev question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = run_query(query)
        print("\nAnswer:\n", result)
        print("-" * 50)
