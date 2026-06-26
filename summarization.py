import pandas as pd
import numpy as np
from collections import Counter
import nltk

# Скачиваем нужные данные для nltk (если ещё не скачали)
nltk.download('punkt_tab')

def tf_based_summary(sentences, num_sentences=2):
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    # TF-based ranking
    all_words = []
    sentence_words = []
    for s in sentences:
        words = nltk.word_tokenize(s.lower())
        filtered = [w for w in words if w.isalnum() and len(w) > 2]
        all_words.extend(filtered)
        sentence_words.append(filtered)

    word_freq = Counter(all_words)
    max_freq = max(word_freq.values()) if word_freq else 1

    scores = []
    for i, words in enumerate(sentence_words):
        score = sum(word_freq.get(w, 0) / max_freq for w in words) / max(len(words), 1)
        scores.append((sentences[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return ' '.join([s for s, _ in scores[:num_sentences]])


def mmr_summary(sentences, num_sentences=2, lambda_param=0.7):
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    # Упрощенный MMR
    words_set = set()
    for s in sentences:
        words = nltk.word_tokenize(s.lower())
        words_set.update([w for w in words if w.isalnum()])
    words_list = list(words_set)

    vectors = []
    for s in sentences:
        words = nltk.word_tokenize(s.lower())
        vec = [1 if w in words else 0 for w in words_list]
        vectors.append(vec)

    def cosine(v1, v2):
        dot = np.dot(v1, v2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        return dot / (n1 * n2) if n1 * n2 > 0 else 0

    relevance = []
    for i, vi in enumerate(vectors):
        sim = sum(cosine(vi, vj) for j, vj in enumerate(vectors) if i != j)
        relevance.append(sim / max(len(vectors) - 1, 1))

    selected = [np.argmax(relevance)]
    remaining = [i for i in range(len(sentences)) if i != selected[0]]

    while len(selected) < num_sentences and remaining:
        mmr_scores = []
        for idx in remaining:
            rel = relevance[idx]
            red = max(cosine(vectors[idx], vectors[s]) for s in selected)
            mmr = lambda_param * rel - (1 - lambda_param) * red
            mmr_scores.append((idx, mmr))
        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best)
        remaining.remove(best)

    selected.sort()
    return ' '.join([sentences[i] for i in selected])


# Загружаем классифицированные данные
classified_df = pd.read_csv('reviews_classified.csv')
print(f"Колонки в файле: {classified_df.columns.tolist()}")

# Используем правильное название колонки - 'restaurant' (не 'business_name')
restaurants = classified_df['restaurant'].unique()
print(f"Суммаризация для {len(restaurants)} ресторанов...")

results = []
for restaurant in restaurants:
    for aspect in ['food', 'service', 'atmosphere', 'price']:
        data = classified_df[
            (classified_df['restaurant'] == restaurant) &
            (classified_df['aspect'] == aspect)
            ]

        if len(data) < 2:
            continue

        sentences = data['sentence'].tolist()

        results.append({
            'restaurant': restaurant,
            'aspect': aspect,
            'sentence_count': len(sentences),
            'tf_summary': tf_based_summary(sentences),
            'mmr_summary': mmr_summary(sentences)
        })

summaries_df = pd.DataFrame(results)
summaries_df.to_csv('final_summaries.csv', index=False)
print(f"Создано {len(summaries_df)} суммаризаций")
print("\nПример результата:")
print(summaries_df.head(10))