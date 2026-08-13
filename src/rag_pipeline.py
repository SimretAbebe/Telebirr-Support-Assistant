import sys
import os

sys.path.append(os.path.dirname(__file__))

from retriever import retrieve
from generator import generate_answer

def ask(user_question, top_k=3):
    matches = retrieve(user_question, top_k=top_k)

    if not matches:
        return "I don't have enough information to answer that question."

    combined_context = ""
    for score, item in matches:
        combined_context += f"Q: {item['question']}\nA: {item['answer']}\n\n"
        print(f"  Candidate: '{item['question']}' (score: {score:.2f})")

    print(f"[Top {len(matches)} candidates retrieved, best score: {matches[0][0]:.2f}]")       
    
    answer = generate_answer(user_question, combined_context)
    return answer

if __name__ == "__main__":
    while True:
        question = input("\nAsk about telebirr (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        response = ask(question)
        print(f"\n{response}")