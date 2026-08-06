import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(user_question, retrieved_answer):
    prompt = f"""You are a helpful assistant answering questions about telebirr mobile money.

ONLY use the information below to answer. Do not add anything you know from elsewhere.
Include all relevant details from the retrieved information below — don't leave out specifics like conditions, exceptions, or steps, even if the answer becomes longer.
If the information below doesn't actually answer the question, say you don't have that information.

Retrieved information:
{retrieved_answer}

User's question: {user_question}

Answer clearly and naturally, based only on the retrieved information above."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    from retriever import retrieve
    
    test_question = "My money didn't arrive after I sent it"
    top_match = retrieve(test_question, top_k=1)[0]
    score, item = top_match
    
    answer = generate_answer(test_question, item['answer'])
    print(answer)