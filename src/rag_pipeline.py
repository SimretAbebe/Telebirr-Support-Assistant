import sys
sys.stdout.reconfigure(encoding='utf-8')
import os

sys.path.append(os.path.dirname(__file__))

from retriever import retrieve
from generator import generate_answer

def ask(user_question, language="en", top_k=3):
    matches = retrieve(user_question, language=language, top_k=top_k)
    
    combined_context = ""
    for score, item in matches:
        combined_context += f"Q: {item['question']}\nA: {item['answer']}\n\n"
        print(f"  Candidate: '{item['question']}' (score: {score:.2f})")

    print(f"[Top {top_k} candidates retrieved, best score: {matches[0][0]:.2f}]")

    answer = generate_answer(user_question, combined_context, language=language)
    return answer

if __name__ == "__main__":
    print("=== English test ===")
    response_en = ask("How do I send money?", language="en")
    print(response_en)
    
    print("\n=== Amharic test ===")
    response_am = ask("ገንዘብ እንዴት እልካለሁ?", language="am")
    print(response_am)