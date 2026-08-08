# Telebirr Support Assistant

A retrieval-augmented (RAG) assistant that answers real questions about telebirr mobile money grounded strictly in documented information, not general AI guesses.

## The problem

telebirr is Ethiopia's largest mobile money platform, integrated across 52+ institutions with a large agent network. But real user reviews show recurring frustration: vague error messages ("Service Unavailable" with no explanation), unclear transaction failure processes, and confusion around what telebirr can and can't do — like whether it supports sending money internationally (it doesn't, though it does receive from abroad).

This project builds a small assistant that answers those kinds of questions clearly — using only real, documented information, and explicitly saying "I don't have that information" rather than guessing when it doesn't know.

## How it works

1. **Embedding** — every stored question is converted into a 384-dimensional vector using `sentence-transformers` (`all-MiniLM-L6-v2`), capturing meaning rather than exact wording.
2. **Retrieval** — a new user question is embedded the same way, then compared against all stored questions using cosine similarity. The top 3 closest matches are retrieved.
3. **Grounding** — the retrieved Q&A pairs (not just the single best one) are passed to an LLM (Groq, Llama 3.1) with an explicit instruction: only answer using the retrieved information, say so if none of it actually fits, and never add outside knowledge.
4. **Generation** — the LLM produces a natural, complete answer grounded in the retrieved content.

## Tech stack

- Python
- `sentence-transformers` (embeddings)
- NumPy (cosine similarity, implemented manually — not via a vector database, by design, to understand the core mechanism)
- Groq API (Llama 3.1 8B Instant) for generation
- Streamlit (chat interface)

## Setup

```bash
git clone <this repo>
cd Telebirr-Support-Assistant
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

Run the chat interface:
```bash
streamlit run app.py
```

Or run the terminal version:
```bash
python src/rag_pipeline.py
```

Run the test suite:
```bash
python src/test_pipeline.py
```

## Known limitations

- Small knowledge base (5 core Q&A entries, covering transfers, cash-out, and failed transactions only)
- English only — Amharic support was part of the original scope and is planned future work
- No live connection to telebirr's actual systems; all answers come from a manually written, static knowledge base
- Retrieval quality depends on how closely a user's phrasing matches the stored questions' embedding space; very indirect or unusual phrasing can still occasionally retrieve a suboptimal match

## Future work

- Amharic language support
- Expand knowledge base to cover more features (tolls, tax payments, cross-border limits)
- Swap manual cosine similarity for a proper vector database (ChromaDB) at scale
- Add automated evaluation against the gold-standard test set (rather than manual review)