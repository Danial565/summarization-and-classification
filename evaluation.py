import pandas as pd
import numpy as np
from rouge import Rouge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class AspectSummarizationEvaluator:

    def __init__(self):
        self.rouge = Rouge()

    def evaluate_summarization_rouge(self, summaries_df, classified_df):
        print(f"\nОценка качества суммаризации (ROUGE)")

        methods = ['tf_summary', 'mmr_summary']
        rouge_scores = {method: {'rouge-1': {'f': []}, 'rouge-2': {'f': []}, 'rouge-l': {'f': []}}
                        for method in methods}

        for _, row in summaries_df.iterrows():
            restaurant = row['restaurant']
            aspect = row['aspect']

            reference_sentences = classified_df[
                (classified_df['restaurant'] == restaurant) &
                (classified_df['aspect'] == aspect)
                ]['sentence'].tolist()

            if len(reference_sentences) == 0:
                continue

            reference_text = ' '.join(reference_sentences)

            for method in methods:
                summary = str(row[method])
                if summary == '' or len(summary) < 10:
                    continue
                try:
                    scores = self.rouge.get_scores(summary, reference_text)[0]
                    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
                        rouge_scores[method][metric]['f'].append(scores[metric]['f'])
                except:
                    continue

        print("\nСредние ROUGE scores:\n")
        for method in methods:
            print(f"{method}:")
            print(f"  ROUGE-1 F1: {np.mean(rouge_scores[method]['rouge-1']['f']):.4f}")
            print(f"  ROUGE-2 F1: {np.mean(rouge_scores[method]['rouge-2']['f']):.4f}")
            print(f"  ROUGE-L F1: {np.mean(rouge_scores[method]['rouge-l']['f']):.4f}")
            print()

        best_method = max(methods, key=lambda m: np.mean(rouge_scores[m]['rouge-1']['f']))
        print(f"Лучший метод: {best_method}\n")

    def evaluate_aspect_classification(self, classified_df):
        print(f"\nОценка точности аспектной классификации")

        df_filtered = classified_df[classified_df['aspect'] != 'other'].copy()

        if len(df_filtered) == 0:
            print("Недостаточно данных")
            return

        print(f"Количество предложений: {len(df_filtered)}")

        le = LabelEncoder()
        y = le.fit_transform(df_filtered['aspect'])

        X_train, X_test, y_train, y_test = train_test_split(
            df_filtered['sentence'], y,
            test_size=0.2, random_state=42, stratify=y
        )

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42, class_weight='balanced')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nТочность классификации: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

        # Кросс-валидация
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42))
        ])
        cv_scores = cross_val_score(pipeline, df_filtered['sentence'], y, cv=5, scoring='accuracy')
        print(f"Кросс-валидация (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")


def main():
    print("Загрузка данных...\n")

    summaries_df = pd.read_csv('final_summaries.csv')
    classified_df = pd.read_csv('reviews_classified.csv')

    evaluator = AspectSummarizationEvaluator()
    evaluator.evaluate_summarization_rouge(summaries_df, classified_df)
    evaluator.evaluate_aspect_classification(classified_df)


if __name__ == "__main__":
    main()