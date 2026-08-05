from sentence_transformers import SentenceTransformer
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.knowledge_base import knowledge_base

model = SentenceTransformer('all-MiniLM-L6-v2')

questions = [item["question"] for item in knowledge_base]
question_embeddings = model.encode(questions)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(user_question, top_k=1):
    user_embedding = model.encode([user_question])[0]
    
    scores = []
    for i, q_embedding in enumerate(question_embeddings):
        score = cosine_similarity(user_embedding, q_embedding)
        scores.append((score, knowledge_base[i]))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    return scores[:top_k]

# Quick test
if __name__ == "__main__":
    test_question = "My money didn't arrive after I sent it"
    results = retrieve(test_question, top_k=2)
    
    for score, item in results:
        print(f"Score: {score:.4f}")
        print(f"Matched Q: {item['question']}")
        print(f"Answer: {item['answer']}")