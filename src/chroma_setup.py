import chromadb
from chromadb.utils import embedding_functions
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.knowledge_base import knowledge_base

CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="telebirr_qa",
    embedding_function=sentence_transformer_ef
)

questions = [item["question"] for item in knowledge_base]
answers = [item["answer"] for item in knowledge_base]
ids = [f"qa_{i}" for i in range(len(knowledge_base))]

collection.upsert(
    documents=questions,
    metadatas=[{"answer": a} for a in answers],
    ids=ids
)

print(f"Added {len(questions)} entries to ChromaDB collection.")
print(f"Collection now has {collection.count()} total entries.")

# Quick test query
results = collection.query(
    query_texts=["My money didn't arrive after I sent it"],
    n_results=3
)

for i, doc in enumerate(results['documents'][0]):
    distance = results['distances'][0][i]
    answer = results['metadatas'][0][i]['answer']
    print(f"\nMatch {i+1}: {doc}")
    print(f"Distance: {distance:.4f}")
    print(f"Answer preview: {answer[:80]}...")