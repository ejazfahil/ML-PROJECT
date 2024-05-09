"""Baseline models. 2024-05-09"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

def train_lr(X,y): return LogisticRegression(max_iter=500,random_state=42).fit(X,y)
def train_rf(X,y): return RandomForestClassifier(n_estimators=100,random_state=42).fit(X,y)
def evaluate(m,X,y,name=""):
    pred=m.predict(X); prob=m.predict_proba(X)[:,1]
    print(f"=== {name} ===\n{classification_report(y,pred)}\nROC-AUC: {roc_auc_score(y,prob):.4f}")
