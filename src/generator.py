import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(user_question, retrieved_context):
    prompt = f"""You are a helpful assistant answering questions about telebirr mobile money.

Below are some retrieved pieces of information, each with its own question and answer. One of them is likely the best match for the user's actual question below — use that one to answer.

Only use facts from the matching piece of information. Do not invent details. Include all relevant specifics from it, even if the answer becomes longer.

If NONE of the pieces below actually relate to the user's question, say you don't have that information.

Retrieved information:
{retrieved_context}

User's question: {user_question}

Answer clearly, based on the most relevant piece of information above."""

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