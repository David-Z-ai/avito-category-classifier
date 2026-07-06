import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm

class TqdmCallback:
    def __init__(self):
        self.pbar = None
        self.total = None
    def __call__(self, env):
        if self.pbar is None:
            self.total = env.params.get('num_iterations', 500)
            self.pbar = tqdm(total=self.total, desc="LightGBM", unit='iter', leave=True)
        self.pbar.update(1)
        if env.iteration + 1 >= self.total:
            self.pbar.close()

def train_baseline(X_train, y_train, X_val, y_val):
    lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, solver='lbfgs', verbose=1)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    print(f"Baseline LR - Accuracy: {acc:.4f}, F1-macro: {f1_macro:.4f}, F1-weighted: {f1_weighted:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0, digits=3))
    return lr

def train_lgbm(X_train, y_train, X_val, y_val, params=None):
    if params is None:
        params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 127,
            'max_depth': 8,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='multi_logloss',
              callbacks=[TqdmCallback()])
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
    print(f"LightGBM - Accuracy: {acc:.4f}, F1-macro: {f1_macro:.4f}, F1-weighted: {f1_weighted:.4f}, ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0, digits=3))
    return model

def train_catboost(X_train, y_train, X_val, y_val, use_gpu=True):
    try:
        import torch
        gpu_available = use_gpu and torch.cuda.is_available()
    except:
        gpu_available = False
    task_type = 'GPU' if gpu_available else 'CPU'
    devices = '0' if gpu_available else None
    print(f"CatBoost использует: {task_type}")

    model = CatBoostClassifier(
        iterations=150,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        task_type=task_type,
        devices=devices,
        early_stopping_rounds=30,
        loss_function='MultiClass',
        thread_count=-1,
        border_count=128,
        verbose=50
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    y_pred = model.predict(X_val).astype(int)
    y_proba = model.predict_proba(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
    print(f"CatBoost - Accuracy: {acc:.4f}, F1-macro: {f1_macro:.4f}, F1-weighted: {f1_weighted:.4f}, ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0, digits=3))

    importance = model.feature_importances_
    print("Топ-10 важнейших признаков (индексы):", np.argsort(importance)[-10:][::-1])
    return model

def save_model(model, path='models/catboost.pkl'):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель сохранена в {path}")
