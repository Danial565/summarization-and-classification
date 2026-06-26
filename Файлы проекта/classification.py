import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize

# Загружаем очищенные данные
df = pd.read_csv('reviews_cleaned.csv')
print(f"Загружено {len(df)} отзывов от {df['Restaurant'].nunique()} ресторанов")

# Ключевые слова для аспектов
ASPECT_PATTERNS = {
    'food': {
        'strong': [
            r'\b(taste|tasty|delicious|flavors?|dishes?|cuisine|meal|menu|dessert|soup|salad|steak|chicken|fish|meat|vegetarian|buffet|portion|ingredients?|spicy|sweet|salty|grilled|fried|boiled|baked|roasted|bbq|spread|dish)\b',
            r'\b(food was|food is|dishes? (was|is|are)|cuisine (was|is)|meal (was|is)|tastes? like)\b',
        ],
        'weak': [r'\b(food|dish|eat|eating|ate|cook|cooked|kitchen)\b']
    },
    'service': {
        'strong': [
            r'\b(staff|waiter|waitress|server|manager|service|friendly|polite|rude|helpful|attentive|accommodating|welcomed|greeted)\b',
            r'\b(service was|service is|staff (was|is|are)|waiter (was|is))\b',
            r'\b(wait(ing|ed)?|long wait|quick service|fast service|slow service)\b',
        ],
        'weak': [r'\b(waiter|staff|order|serve|served|service)\b']
    },
    'atmosphere': {
        'strong': [
            r'\b(atmosphere|ambiance|ambience|decor|interior|environment|vibe|setting|beautiful|cozy|comfortable|view|scenery|picture|photo|surrounding)\b',
            r'\b(place (was|is)|restaurant (was|is)|atmosphere (was|is))\b',
            r'\b(noisy|quiet|loud|crowded|spacious|cramped|clean|dirty)\b',
        ],
        'weak': [r'\b(place|restaurant|hotel|resort)\b']
    },
    'price': {
        'strong': [
            r'\b(price|prices|cost|expensive|cheap|affordable|reasonable|overpriced|inexpensive|value|worth|money|bill|buffet)\b',
            r'\b(price (was|is|are)|cost (was|is)|paid|pay)\b',
            r'\b(worth (it|the money)|good value|great value|bad value)\b',
            r'\b(cheap price|low price|high price|fair price|affordable price)\b',
        ],
        'weak': [r'\b(pay|paid|cash|card|ewallet|boost|tng|rm)\b']
    }
}


def classify_sentence_improved(sentence):
    if not isinstance(sentence, str) or sentence.strip() == '':
        return ['other']

    sentence_lower = sentence.lower()
    aspect_scores = {}

    for aspect, patterns in ASPECT_PATTERNS.items():
        score = 0
        for pattern in patterns['strong']:
            matches = re.findall(pattern, sentence_lower)
            if matches:
                score += 3 * len(matches)
        for pattern in patterns['weak']:
            matches = re.findall(pattern, sentence_lower)
            if matches:
                score += 1 * len(matches)
        if score > 0:
            aspect_scores[aspect] = score

    if not aspect_scores:
        return ['other']

    sorted_aspects = sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True)
    threshold = max(sorted_aspects[0][1] * 0.5, 2)
    selected = [a for a, s in sorted_aspects if s >= threshold]

    return selected[:2] if selected else ['other']


def classify_reviews(df):
    sentences_data = []
    seen = set()

    for idx, row in df.iterrows():
        text = row['cleaned_text']
        if not isinstance(text, str):
            continue

        try:
            sentences = sent_tokenize(text)
        except:
            sentences = [text]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            sent_key = f"{row['Restaurant']}_{sentence[:100]}"
            if sent_key in seen:
                continue
            seen.add(sent_key)

            cleaned = re.sub(r'[^\w\s]', ' ', sentence.lower())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

            aspects = classify_sentence_improved(cleaned)

            sentences_data.append({
                'restaurant': row['Restaurant'],
                'sentence': cleaned,
                'aspect': aspects[0]
            })

    return pd.DataFrame(sentences_data)


print("\nКлассификация предложений...")
classified_df = classify_reviews(df)
print(f"Получено {len(classified_df)} предложений")

# Статистика
print("\nРаспределение по аспектам:")
print(classified_df['aspect'].value_counts())

classified_df.to_csv('reviews_classified.csv', index=False)
print("\nСохранено в reviews_classified.csv")