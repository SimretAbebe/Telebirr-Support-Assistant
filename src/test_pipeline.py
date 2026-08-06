import sys
import os
sys.path.append(os.path.dirname(__file__))

from rag_pipeline import ask
from data.knowledge_base import knowledge_base

print("Testing with original base questions")

for i, item in enumerate(knowledge_base, start=1):
    print(f"\n Test {i}")
    print(f"Question: {item['question']}")
    response = ask(item['question'])
    print(f"Response: {response}")
    