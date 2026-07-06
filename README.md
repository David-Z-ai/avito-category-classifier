# Avito Category Classifier

Классификация категорий объявлений на основе текста (заголовок + описание) с использованием классических методов машинного обучения.

Проект выполнен для портфолио при подаче на **Avito Data Science Bootcamp** (трек «Классический ML»).

## Задача

Предсказать категорию товара по тексту объявления. Датасет **Avito Category Prediction** с Kaggle. Для ускорения обучения используется сэмпл из 100 000 строк.

## Данные

- Источник: [Avito Category Prediction on Kaggle](https://www.kaggle.com/competitions/avito-category-prediction)

## Подход

1. **Предобработка:** объединение текста, заполнение пропусков.
2. **Признаки:** TF-IDF (unigrams + bigrams) + 6 числовых признаков (длина, количество слов, уникальных слов, заглавных букв, цифр, знаков препинания).
3. **Модели:**
   - Logistic Regression (baseline)
   - LightGBM
   - CatBoost (финальная модель, обучена на GPU в Colab)
4. **Оценка:** Accuracy, F1-macro, F1-weighted, ROC-AUC (macro ovr).

