import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag_pipeline import ask
from data.knowledge_base import knowledge_base

print("Testing with original base questions")

for i, item in enumerate(knowledge_base, start=1):
    print(f"\n Test {i}")
    print(f"Question: {item['question']}")
    response = ask(item['question'])
    print(f"Response: {response}")
    

print("TESTING WITH REWORDED / REAL-WORLD PHRASED QUESTIONS")

reworded_tests = [
    "My money didn't arrive after I sent it",
    "how can i transfer money?",
    "How do I cash out at an agent shop?",
    "Can I send birr to my brother in America?",
    "will telebirr tell me when my problem is fixed?"
]

for i, question in enumerate(reworded_tests, start=1):
    print(f"\n Reworded Test {i} ")
    print(f"Question: {question}")
    response = ask(question)
    print(f"Response: {response}")