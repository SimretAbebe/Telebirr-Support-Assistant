# Telebirr Support Assistant

A bilingual (English & Amharic) RAG assistant that answers real telebirr questions — grounded strictly in documented information, no made-up answers.

**Live demo:** []

## How it works

Question → embedding (English: `all-MiniLM-L6-v2`, Amharic: `rasyosef/roberta-amharic-text-embedding-base`) → hybrid retrieval (semantic + keyword, via ChromaDB) → grounded generation (Groq, `openai/gpt-oss-20b`) → clear answer, or an honest "I don't have that information" if nothing retrieved actually fits.

## Data

- Official Ethio Telecom telebirr FAQ (facts)
- Real Google Play reviews, scraped and mined for realistic question phrasing — this is how "app keeps crashing" got added as its own category, since it turned out to be the most common real complaint
- Amharic content written by me(no official Amharic FAQ exists)

## Setup

```bash
git clone <this repo>
cd Telebirr-Support-Assistant
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

Add a `.env` file with `GROQ_API_KEY=your_key_here`, then:

```bash
python src/chroma_setup.py
python src/chroma_setup_am.py
python app.py
```

## Evaluation

Evaluated with RAGAS (faithfulness, relevancy, precision, recall) instead of manual judging. Scores hover around 0.79–0.97 across runs — some run-to-run variance, since RAGAS itself uses an LLM to judge quality.

## Real bugs found and fixed

- Retrieval missed an indirectly-phrased question due to a synonym gap ("tell me" vs "confirms") — fixed by rewriting the stored question, not just tuning the algorithm
- An over-strict grounding prompt caused false refusals on perfect matches — simplified and fixed
- A missing line of code silently passed empty context to the LLM, looking like an AI failure but was a plain data bug
- Groq deprecated the original model mid-project — migrated to `openai/gpt-oss-20b`
- General multilingual embeddings underperform badly on Amharic — verified this directly before choosing a dedicated Amharic model

## Limitations

10 entries per language, static knowledge base, no live telebirr connection.