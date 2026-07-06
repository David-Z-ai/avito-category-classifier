import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data, preprocess_text, encode_target, split_data
from features import extract_numeric_features, build_tfidf_vectorizer, transform_text, combine_features
from train import train_catboost, save_model
from predict import predict, save_submission

def main():
    print("="*60)
    print("Avito Category Classification - Full Pipeline")
    print("="*60)

    print("\n[1] Загрузка данных...")
    train_df, test_df = load_data(train_sample=100000, test_sample=20000)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    print("\n[2] Предобработка текста...")
    train_df = preprocess_text(train_df)
    test_df = preprocess_text(test_df)
    train_df, encoder = encode_target(train_df)
    X = train_df['text']
    y = train_df['target']

    print("\n[3] Разбиение на train/val...")
    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    print("\n[4] Создание признаков...")
    tfidf = build_tfidf_vectorizer()
    print("   TF-IDF обучение...")
    X_train_tfidf = transform_text(X_train, tfidf)
    X_val_tfidf = transform_text(X_val, tfidf)
    print(f"   TF-IDF shapes: train {X_train_tfidf.shape}, val {X_val_tfidf.shape}")

    print("   Числовые признаки...")
    X_train_num = extract_numeric_features(X_train)
    X_val_num = extract_numeric_features(X_val)
    print(f"   Numeric shapes: train {X_train_num.shape}, val {X_val_num.shape}")

    X_train_final = combine_features(X_train_tfidf, X_train_num)
    X_val_final = combine_features(X_val_tfidf, X_val_num)
    print(f"   Финальные признаки: train {X_train_final.shape}, val {X_val_final.shape}")

    print("\n[5] Обучение CatBoost...")
    model = train_catboost(X_train_final, y_train, X_val_final, y_val, use_gpu=True)

    print("\n[6] Сохранение модели...")
    save_model(model, 'models/catboost.pkl')
    import pickle
    with open('models/tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print("Векторизатор и энкодер сохранены.")

    print("\n[7] Предсказание на тестовой выборке...")
    X_test_tfidf = transform_text(test_df['text'], tfidf)
    X_test_num = extract_numeric_features(test_df['text'])
    X_test_final = combine_features(X_test_tfidf, X_test_num)
    preds = predict(model, X_test_final)
    save_submission(preds, encoder, test_df, 'submissions/submission.csv')

    print("\nГотово!")

if __name__ == '__main__':
    main()
