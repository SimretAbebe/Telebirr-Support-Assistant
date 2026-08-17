import chromadb
from chromadb.utils import embedding_functions
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.knowledge_base_am import knowledge_base_am

client = chromadb.PersistentClient(path="./chroma_db")

amharic_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="rasyosef/roberta-amharic-text-embedding-base"
)

collection = client.get_or_create_collection(
    name="telebirr_qa_am",
    embedding_function=amharic_ef
)

questions = [item["question"] for item in knowledge_base_am]
answers = [item["answer"] for item in knowledge_base_am]
ids = [f"qa_am_{i}" for i in range(len(knowledge_base_am))]

collection.upsert(
    documents=questions,
    metadatas=[{"answer": a} for a in answers],
    ids=ids
)

print(f"Added {len(questions)} Amharic entries to ChromaDB collection.")
print(f"Collection now has {collection.count()} total entries.")