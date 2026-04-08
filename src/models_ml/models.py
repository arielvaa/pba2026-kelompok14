"""
ML Model configurations
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def get_models():
    """
    Get all models for benchmarking
    Returns dict of {model_name: model_instance}
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'Naive Bayes': MultinomialNB(
            alpha=0.1
        ),
        'SVM (LinearSVC)': LinearSVC(
            max_iter=1000,
            random_state=42
        )
    }
    
    return models


def get_model(model_name):
    """Get a specific model by name"""
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    return models[model_name]
