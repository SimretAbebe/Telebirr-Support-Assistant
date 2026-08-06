import sys
import os
sys.path.append(os.path.dirname(__file__))

from retriever import retrieve
from generator import generate_answer

def ask(user_question):
    top_match = retrieve(user_question, top_k=1)[0]
    score, item = top_match
    
    print(f"[Matched: '{item['question']}' | Confidence: {score:.2f}]")
    
    answer = generate_answer(user_question, item['answer'])
    return answer

if __name__ == "__main__":
    while True:
        question = input("\nAsk about telebirr (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        response = ask(question)
        print(f"\n{response}")