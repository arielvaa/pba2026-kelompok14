"""
Utility functions for ML pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import time


def load_data(filepath):
    """Load preprocessed dataset"""
    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape}")
    return df


def split_data(df, text_col='text', target_col='emotion', test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X = df[text_col]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test, max_features=5000, ngram_range=(1, 2)):
    """Convert text to TF-IDF features"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.8
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✓ TF-IDF vectorization complete!")
    print(f"  Features: {X_train_tfidf.shape[1]:,}")
    print(f"  Sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.2f}%")
    
    return X_train_tfidf, X_test_tfidf, vectorizer


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, verbose=True):
    """Train and evaluate a model"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Training {model_name}...")
        print(f"{'='*80}")
    
    # Training
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results = {
        'model': model,
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'pred_time': pred_time,
        'y_pred': y_pred
    }
    
    if verbose:
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Prediction time: {pred_time:.4f}s")
    
    return results


def compare_models(results_dict):
    """Compare multiple models and return best one"""
    comparison_df = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy': [results_dict[m]['accuracy'] for m in results_dict.keys()],
        'Precision': [results_dict[m]['precision'] for m in results_dict.keys()],
        'Recall': [results_dict[m]['recall'] for m in results_dict.keys()],
        'F1-Score': [results_dict[m]['f1_score'] for m in results_dict.keys()],
        'Train Time (s)': [results_dict[m]['train_time'] for m in results_dict.keys()],
        'Pred Time (s)': [results_dict[m]['pred_time'] for m in results_dict.keys()]
    })
    
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)
    print(comparison_df.to_string(index=False))
    print("="*100)
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_f1 = comparison_df.iloc[0]['F1-Score']
    print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")
    
    return comparison_df, best_model_name


def save_results(comparison_df, best_model_name, results_dict, vectorizer, 
                 output_dir='../reports/tables', model_dir='../models/ml'):
    """Save model comparison and best model"""
    import joblib
    import os
    
    # Save comparison
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = f'{output_dir}/ml_model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Model comparison saved to: {comparison_path}")
    
    # Save best model
    os.makedirs(model_dir, exist_ok=True)
    best_model = results_dict[best_model_name]['model']
    model_path = f'{model_dir}/{best_model_name.lower().replace(" ", "_")}_model.pkl'
    vectorizer_path = f'{model_dir}/tfidf_vectorizer.pkl'
    
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"✓ Best model saved to: {model_path}")
    print(f"✓ Vectorizer saved to: {vectorizer_path}")
    
    return model_path, vectorizer_path
