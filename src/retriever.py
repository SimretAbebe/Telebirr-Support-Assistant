import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="telebirr_qa",
    embedding_function=sentence_transformer_ef
)

def retrieve(user_question, top_k=3):
    results = collection.query(
        query_texts=[user_question],
        n_results=top_k
    )

    matches = []
    for i in range(len(results['documents'][0])):
        question = results['documents'][0][i]
        answer = results['metadatas'][0][i]['answer']
        distance = results['distances'][0][i]
        similarity = 1 - distance
        matches.append((similarity, {"question": question, "answer": answer}))

    return matches