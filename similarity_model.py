from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk


# Load semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")


def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def compute_overall_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(float(similarity) * 100, 2)


def sentence_level_similarity(text1, text2, threshold=0.75):
    sentences1 = split_into_sentences(text1)
    sentences2 = split_into_sentences(text2)

    if not sentences1 or not sentences2:
        return []

    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    similarity_matrix = cosine_similarity(embeddings1, embeddings2)

    matches = []

    for i, row in enumerate(similarity_matrix):
        max_score = max(row)
        if max_score >= threshold:
            matched_index = row.argmax()
            matches.append({
                "input_sentence": sentences1[i],
                "matched_sentence": sentences2[matched_index],
                "similarity": round(float(max_score) * 100, 2)
            })

    return matches


def analyze_similarity(user_text, web_text):
    overall_score = compute_overall_similarity(user_text, web_text)
    sentence_matches = sentence_level_similarity(user_text, web_text)

    return {
        "overall_similarity": overall_score,
        "matched_sentences": sentence_matches
    }
