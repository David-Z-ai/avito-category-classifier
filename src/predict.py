import pickle
import os
import pandas as pd

def load_model(path='models/catboost.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, X_test_final):
    return model.predict(X_test_final).astype(int)

def save_submission(predictions, encoder, test_df, path='submissions/submission.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    category_names = encoder.inverse_transform(predictions)
    submission = pd.DataFrame({
        'id': test_df.index,
        'category_id': category_names
    })
    submission.to_csv(path, index=False)
    print(f"Submission сохранён в {path}")
