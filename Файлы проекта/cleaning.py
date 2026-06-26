import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Загружаем отфильтрованный датасет
df = pd.read_csv('filtered_restaurants.csv')


# Извлекаем числовой рейтинг
def extract_rating(rating_str):
    match = re.search(r'(\d+\.?\d*)', str(rating_str))
    return float(match.group(1)) if match else 0.0


df['rating_num'] = df['Rating'].apply(extract_rating)


# Очистка текста
def clean_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ""

    # Удаляем "More" маркер Tripadvisor
    text = re.sub(r'вЂ¦More', ' ', text)
    text = re.sub(r'…More', ' ', text)

    # Удаление эмодзи и всех не-ASCII символов
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление знаков препинания
    text = re.sub(r'[^\w\s]', ' ', text)

    # Удаление цифр
    text = re.sub(r'\d+', '', text)

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return ""

    # Токенизация
    tokens = word_tokenize(text)

    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


print("Очистка текста...")
df['cleaned_text'] = df['Review'].apply(clean_text)

# Удаляем пустые отзывы
df = df[df['cleaned_text'] != '']

# Оставляем ресторан и очищенный отзыв
df_final = df[['Restaurant', 'cleaned_text']].copy()

# Сохраняем
df_final.to_csv('reviews_cleaned.csv', index=False)

print(f"Сохранено {len(df_final)} очищенных отзывов")