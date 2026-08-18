import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

english_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
amharic_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="rasyosef/roberta-amharic-text-embedding-base"
)

english_collection = client.get_or_create_collection(
    name="telebirr_qa",
    embedding_function=english_ef
)
amharic_collection = client.get_or_create_collection(
    name="telebirr_qa_am",
    embedding_function=amharic_ef
)

def retrieve(user_question, language="en", top_k=3):
    collection = amharic_collection if language == "am" else english_collection

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