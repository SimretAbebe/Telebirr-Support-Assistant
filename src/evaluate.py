import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import ask
from retriever import retrieve
from data.knowledge_base import knowledge_base
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

# Build the evaluation dataset using your real questions and known-correct answers
eval_questions = [item["question"] for item in knowledge_base]
ground_truths = [item["answer"] for item in knowledge_base]

# Collect actual system outputs for each question
generated_answers = []
retrieved_contexts = []

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

groq_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)
ragas_llm = LangchainLLMWrapper(groq_llm)

for q in eval_questions:
    matches = retrieve(q, top_k=3)
    contexts = [item["answer"] for score, item in matches]
    retrieved_contexts.append(contexts)
    
    answer = ask(q)
    generated_answers.append(answer)
    print(f"Processed: {q[:50]}...")

print("\nDone collecting outputs. Ready for RAGAS evaluation.")



# Package everything into the format RAGAS expects
eval_data = {
    "question": eval_questions,
    "answer": generated_answers,
    "contexts": retrieved_contexts,
    "ground_truth": ground_truths
}

eval_dataset = Dataset.from_dict(eval_data)

results = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print("RAGAS EVALUATION RESULTS")
print(results)

results_df = results.to_pandas()
results_df.to_csv("data/ragas_results.csv", index=False)
print("\nDetailed results saved to data/ragas_results.csv")
