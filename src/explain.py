"""SHAP explanations. 2024-05-28"""
import shap, numpy as np

def shap_importance(model, X, feature_names, top_k=10):
    exp = shap.TreeExplainer(model)
    vals = exp.shap_values(X)
    if isinstance(vals,list): vals=vals[1]
    mean_abs = np.abs(vals).mean(axis=0)
    return sorted(zip(feature_names,mean_abs),key=lambda x:x[1],reverse=True)[:top_k]
