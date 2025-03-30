import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_and_preprocess

# Завантаження даних
train_df = load_and_preprocess('data/train.csv')
train_df = pd.get_dummies(train_df, dtype=np.uint8)
X_train = train_df.drop('satisfaction', axis=1)
y_train = train_df['satisfaction']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Навчання моделі
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_train_scaled)
print("F1 Score:", f1_score(y_train, y_pred))
print("ROC AUC:", roc_auc_score(y_train, y_pred))

