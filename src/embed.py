from sentence_transformers import SentenceTransformer
import sys
import os
import numpy as np

#finding the path to the data folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.knowledge_base import knowledge_base

#load the embedding model
model=SentenceTransformer('all-MiniLM-L6-v2')

#pull out just the questions to embed
questions =[item['question'] for item in knowledge_base]

#turn each question into vectors
question_embeddings = model.encode(questions)

print("Number of questions:",  len(questions))
print("Shape of embeddings:", question_embeddings.shape)
print("First embedding(first 10 numbers only):", question_embeddings[0][:10])


#test embed a brand new question
new_question= "My money didn't arrive after I sent it!"
new_embedding = model.encode([new_question])

def cosine_similarity(a,b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

for i,q in enumerate(questions):
    score = cosine_similarity(new_embedding[0], question_embeddings[i])
    print(f"Similarity to '{q}': {score:.4f}")
