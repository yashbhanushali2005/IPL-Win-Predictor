import pandas as pd
import numpy as np

df = pd.read_csv("data/IPL.csv", low_memory=False)

# Keep only 2nd innings
df = df[df['innings'] == 2].copy()

# Feature engineering (same as before)
df['current_score'] = df.groupby('match_id')['runs_total'].cumsum()
df['balls_bowled'] = df.groupby('match_id').cumcount() + 1

df['runs_left'] = df['runs_target'] - df['current_score']
df['balls_left'] = 120 - df['balls_bowled']
df['wickets_left'] = 10 - df['team_wicket']

df['current_run_rate'] = np.where(
    df['balls_bowled'] == 0,
    0,
    (df['current_score'] * 6) / df['balls_bowled']
)

df['required_run_rate'] = np.where(
    df['balls_left'] == 0,
    0,
    (df['runs_left'] * 6) / df['balls_left']
)

df['result'] = (df['match_won_by'] == df['batting_team']).astype(int)

features = [
    'runs_left',
    'balls_left',
    'wickets_left',
    'current_run_rate',
    'required_run_rate'
]

X = df[features]
y = df['result']

import numpy as np

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values
df.dropna(inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

import pickle

pickle.dump(model, open("ipl_win_model.pkl", "wb"))
print("Model saved successfully!")