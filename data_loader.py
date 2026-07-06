import os
import pandas as pd
import kagglehub
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from getpass import getpass

def setup_kaggle_token():
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    token_file = os.path.join(kaggle_dir, 'access_token')
    if not os.path.exists(token_file):
        token = getpass("Введите ваш KAGGLE_API_TOKEN (KGAT_...): ")
        with open(token_file, 'w') as f:
            f.write(token.strip())
        os.chmod(token_file, 0o600)
        print("Токен сохранён.")
    with open(token_file, 'r') as f:
        os.environ['KAGGLE_API_TOKEN'] = f.read().strip()

def load_data(train_sample=100000, test_sample=20000):
    setup_kaggle_token()
    competition_path = kagglehub.competition_download('avito-category-prediction')
    train_path = os.path.join(competition_path, 'train.csv')
    test_path = os.path.join(competition_path, 'test.csv')
    train_df = pd.read_csv(train_path, nrows=train_sample)
    test_df = pd.read_csv(test_path, nrows=test_sample)
    return train_df, test_df

def preprocess_text(df):
    df = df.copy()
    df['text'] = df['title'].astype(str) + ' ' + df['description'].fillna('').astype(str)
    return df

def encode_target(train_df):
    train_df = train_df.copy()
    le = LabelEncoder()
    train_df['target'] = le.fit_transform(train_df['Category'])
    return train_df[['text', 'target']], le

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
