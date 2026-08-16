from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('rasyosef/roberta-amharic-text-embedding-base')

amharic_text = "እንዴት ገንዘብ መላክ እችላለሁ?"  # "How can I send money?"
english_text = "How do I send money?"

# Test 1: Amharic-to-Amharic (should score high if the model understands Amharic meaning)
amharic_text_2 = "ገንዘብ እንዴት እልካለሁ?"  # a differently-phrased Amharic version of the same question

embeddings = model.encode([amharic_text, amharic_text_2])
similarity = util.cos_sim(embeddings[0], embeddings[1])

print(f"Similarity between two differently-phrased Amharic questions (same meaning): {similarity.item():.4f}")