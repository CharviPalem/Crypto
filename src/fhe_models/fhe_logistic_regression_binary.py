#!/usr/bin/env python3
"""
FHE Logistic Regression for Binary Medical Classification
(Cardiovascular vs Non-Cardiovascular Diseases)

This version simplifies the problem to binary classification for better accuracy.
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Concrete-ML imports
try:
    from concrete.ml.sklearn import LogisticRegression as FHELogisticRegression
    CONCRETE_ML_AVAILABLE = True
except ImportError:
    print("Warning: Concrete-ML not available. Using sklearn fallback.")
    CONCRETE_ML_AVAILABLE = False
    from sklearn.linear_model import LogisticRegression as FHELogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define disease categories
CARDIOVASCULAR_DISEASES = {
    'Coronary Artery Disease',
    'Atrial Fibrillation',
    'Hypertension',
    'Heart Disease',
    'Cardiovascular Disease'
}

def categorize_disease(disease: str) -> str:
    """
    Categorize disease into binary classes
    
    Args:
        disease: Original disease name
        
    Returns:
        'Cardiovascular' or 'Non-Cardiovascular'
    """
    if disease in CARDIOVASCULAR_DISEASES:
        return 'Cardiovascular'
    else:
        return 'Non-Cardiovascular'


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load and prepare data with binary classification"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'data' not in data:
            raise ValueError("Dataset must contain 'data' field")
        
        df = pd.DataFrame(data['data'])
        
        # Create binary target
        df['disease_category'] = df['disease'].apply(categorize_disease)
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        logger.info(f"Disease distribution:")
        logger.info(f"\n{df['disease_category'].value_counts()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)


def prepare_features_and_labels(df: pd.DataFrame, max_features: int = 500) -> Tuple:
    """
    Prepare features and binary labels
    
    Args:
        df: DataFrame with medical records
        max_features: Maximum number of features
        
    Returns:
        Tuple of (features, labels, vectorizer, label_encoder)
    """
    # Use medical notes as features
    if 'medical_notes_clean' in df.columns:
        texts = df['medical_notes_clean'].fillna('').astype(str)
    elif 'medical_notes' in df.columns:
        texts = df['medical_notes'].fillna('').astype(str)
    else:
        raise ValueError("Dataset must contain 'medical_notes' or 'medical_notes_clean'")
    
    # Binary target
    labels = df['disease_category']
    
    # Simpler vectorizer for binary classification
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    
    # Transform text to features
    X = vectorizer.fit_transform(texts).toarray()
    
    # Encode labels (0 or 1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Classes: {label_encoder.classes_}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y, vectorizer, label_encoder


def train_fhe_model(X_train, y_train, n_bits: int = 8):
    """
    Train FHE Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_bits: Number of bits for quantization (lower = faster, less accurate)
        
    Returns:
        Trained model and training time
    """
    logger.info("Training FHE Logistic Regression model...")
    
    start_time = time.time()
    
    if CONCRETE_ML_AVAILABLE:
        # Use Concrete-ML with quantization
        model = FHELogisticRegression(n_bits=n_bits, max_iter=1000)
    else:
        # Fallback to sklearn
        model = FHELogisticRegression(max_iter=1000, random_state=42)
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_time


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model performance"""
    logger.info("Evaluating model on clear data...")
    
    # Clear inference
    start_time = time.time()
    y_pred_clear = model.predict(X_test)
    clear_inference_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred_clear)
    
    logger.info(f"Clear Accuracy: {accuracy:.4f}")
    logger.info(f"Clear Inference Time: {clear_inference_time:.4f}s")
    
    # Classification report
    report = classification_report(
        y_test, y_pred_clear,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_clear)
    
    return {
        'accuracy': accuracy,
        'inference_time': clear_inference_time,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def compile_and_test_fhe(model, X_test, y_test):
    """Compile model for FHE and test (if Concrete-ML available)"""
    if not CONCRETE_ML_AVAILABLE:
        logger.warning("Concrete-ML not available, skipping FHE compilation")
        return None
    
    try:
        logger.info("Compiling model for FHE execution...")
        
        # Compile the model
        start_time = time.time()
        model.compile(X_test[:10])  # Use small sample for compilation
        compile_time = time.time() - start_time
        
        logger.info(f"FHE compilation completed in {compile_time:.2f} seconds")
        
        # Test FHE inference on a few samples
        logger.info("Testing FHE inference on sample data...")
        start_time = time.time()
        
        # Test on first 5 samples
        X_test_sample = X_test[:5]
        y_pred_fhe = model.predict(X_test_sample, fhe="execute")
        
        fhe_inference_time = time.time() - start_time
        
        # Compare with clear predictions
        y_pred_clear = model.predict(X_test_sample)
        fhe_accuracy = accuracy_score(y_pred_clear, y_pred_fhe)
        
        logger.info(f"FHE Inference Time (5 samples): {fhe_inference_time:.2f}s")
        logger.info(f"FHE vs Clear Match: {fhe_accuracy:.4f}")
        
        return {
            'compile_time': compile_time,
            'fhe_inference_time': fhe_inference_time,
            'fhe_accuracy': fhe_accuracy,
            'samples_tested': len(X_test_sample)
        }
        
    except Exception as e:
        logger.error(f"FHE compilation/execution failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="FHE Binary Logistic Regression for Medical Classification"
    )
    parser.add_argument('--train', required=True, help='Path to training data JSON')
    parser.add_argument('--max-features', type=int, default=500, help='Max features')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--n-bits', type=int, default=8, help='Quantization bits')
    parser.add_argument('--output', default='data/results/fhe_lr_binary_results.json')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-fhe', action='store_true', help='Skip FHE compilation')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("FHE Binary Logistic Regression Pipeline")
    logger.info("="*60)
    logger.info(f"Training data: {args.train}")
    logger.info(f"Max features: {args.max_features}")
    logger.info(f"Quantization bits: {args.n_bits}")
    
    # Load and prepare data
    df = load_and_prepare_data(args.train)
    X, y, vectorizer, label_encoder = prepare_features_and_labels(df, args.max_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    model, training_time = train_fhe_model(X_train, y_train, args.n_bits)
    
    # Evaluate on clear data
    clear_results = evaluate_model(model, X_test, y_test, label_encoder)
    
    # FHE compilation and testing
    fhe_results = None
    if not args.skip_fhe and CONCRETE_ML_AVAILABLE:
        fhe_results = compile_and_test_fhe(model, X_test, y_test)
    
    # Prepare results
    results = {
        'experiment_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'Binary Logistic Regression',
            'classification_type': 'Cardiovascular vs Non-Cardiovascular',
            'max_features': args.max_features,
            'n_bits': args.n_bits,
            'test_size': args.test_size,
            'concrete_ml_available': CONCRETE_ML_AVAILABLE
        },
        'dataset_info': {
            'total_samples': len(X),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X.shape[1],
            'classes': label_encoder.classes_.tolist(),
            'class_distribution': {
                label_encoder.classes_[i]: int(np.sum(y == i))
                for i in range(len(label_encoder.classes_))
            }
        },
        'timing': {
            'training_time': training_time,
            'clear_inference_time': clear_results['inference_time']
        },
        'clear_results': {
            'accuracy': clear_results['accuracy'],
            'classification_report': clear_results['classification_report'],
            'confusion_matrix': clear_results['confusion_matrix']
        }
    }
    
    if fhe_results:
        results['fhe_results'] = fhe_results
        results['timing']['fhe_compile_time'] = fhe_results['compile_time']
        results['timing']['fhe_inference_time'] = fhe_results['fhe_inference_time']
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("Results Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Binary Classification Accuracy: {clear_results['accuracy']:.4f}")
    logger.info(f"Training Time: {training_time:.2f}s")
    if fhe_results:
        logger.info(f"FHE Compile Time: {fhe_results['compile_time']:.2f}s")
        logger.info(f"FHE Inference Time: {fhe_results['fhe_inference_time']:.2f}s")
    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
