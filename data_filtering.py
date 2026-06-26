import pandas as pd
import numpy as np

# Загружаем датасет
df = pd.read_csv('GoogleReview_Petaling Jaya.csv')

# Считаем количество отзывов на ресторан
restaurant_counts = df['Restaurant'].value_counts()

# Отбираем 50 ресторанов с наибольшим количеством отзывов
top_50_restaurants = restaurant_counts.nlargest(50).index.tolist()

# Убеждаемся, что у всех 100+ отзывов (если нет - добираем)
restaurants_100plus = restaurant_counts[restaurant_counts >= 100].index.tolist()

if len(restaurants_100plus) >= 50:
    top_50_restaurants = restaurants_100plus[:50]
else:
    remaining = 50 - len(restaurants_100plus)
    others = restaurant_counts[~restaurant_counts.index.isin(restaurants_100plus)]
    top_others = others.nlargest(remaining).index.tolist()
    top_50_restaurants = restaurants_100plus + top_others

# Фильтруем данные
df_filtered = df[df['Restaurant'].isin(top_50_restaurants)].copy()

# Для каждого ресторана оставляем 100-110 отзывов
np.random.seed(42)
final_dfs = []
for restaurant in top_50_restaurants:
    restaurant_data = df_filtered[df_filtered['Restaurant'] == restaurant]
    if len(restaurant_data) > 110:
        restaurant_data = restaurant_data.sample(n=min(110, len(restaurant_data)), random_state=42)
    final_dfs.append(restaurant_data)

df_final = pd.concat(final_dfs, ignore_index=True)

# Сохраняем результат
df_final.to_csv('filtered_restaurants.csv', index=False)

print(f"Готово: {df_final['Restaurant'].nunique()} ресторанов, {len(df_final)} отзывов")
summary = df_final.groupby('Restaurant').agg(num_reviews=('Review', 'count')).sort_values('num_reviews', ascending=False)
print(summary)