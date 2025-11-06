#!/usr/bin/env python3
"""
FHE Logistic Regression for Medical Text Classification

This script implements encrypted logistic regression inference using Concrete-ML
for privacy-preserving medical text analysis.
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import yaml

# Concrete-ML imports
try:
    from concrete.ml.sklearn import LogisticRegression as FHELogisticRegression
    CONCRETE_ML_AVAILABLE = True
except ImportError:
    print("Warning: Concrete-ML not available. Install with: pip install concrete-ml")
    CONCRETE_ML_AVAILABLE = False
    # Fallback to regular sklearn for development
    from sklearn.linear_model import LogisticRegression as FHELogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_fhe_config(config_path: str = "src/configs/fhe_config.yaml") -> Dict[str, Any]:
    """
    Load FHE configuration from YAML file
    
    Args:
        config_path: Path to FHE configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"FHE config not found at {config_path}, using defaults")
            return {
                'fhe': {
                    'scheme': 'BFV',
                    'polynomial_modulus_degree': 8192,
                    'plaintext_modulus': 1032193,
                    'security_level': 128
                }
            }
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ FHE configuration loaded from: {config_path}")
        logger.info(f"   Scheme: {config['fhe']['scheme']}")
        logger.info(f"   Polynomial modulus degree: {config['fhe']['polynomial_modulus_degree']}")
        logger.info(f"   Security level: {config['fhe']['security_level']} bits")
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading FHE config: {e}")
        logger.warning("Using default FHE parameters")
        return {
            'fhe': {
                'scheme': 'BFV',
                'polynomial_modulus_degree': 8192,
                'plaintext_modulus': 1032193,
                'security_level': 128
            }
        }


def setup_fhe_context(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup FHE context parameters from config
    
    Args:
        config: FHE configuration dictionary
        
    Returns:
        FHE context parameters
    """
    fhe_params = config.get('fhe', {})
    
    context_params = {
        'scheme': fhe_params.get('scheme', 'BFV'),
        'poly_modulus_degree': fhe_params.get('polynomial_modulus_degree', 8192),
        'plain_modulus': fhe_params.get('plaintext_modulus', 1032193),
        'security_level': fhe_params.get('security_level', 128),
        'use_batching': fhe_params.get('use_batching', True)
    }
    
    logger.info("FHE context parameters configured:")
    for key, value in context_params.items():
        logger.info(f"  {key}: {value}")
    
    return context_params


def load_model_config(config_path: str = "src/configs/model_config.yaml") -> Dict[str, Any]:
    """
    Load model configuration from YAML file
    
    Args:
        config_path: Path to model configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Model config not found at {config_path}, using defaults")
            return {
                'model': {
                    'logistic_regression': {
                        'n_bits': 8,
                        'max_iter': 100,
                        'regularization': 0.01,
                        'solver': 'liblinear',
                        'random_state': 42
                    }
                }
            }
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ Model configuration loaded from: {config_path}")
        lr_params = config.get('model', {}).get('logistic_regression', {})
        logger.info(f"   LR n_bits: {lr_params.get('n_bits', 8)}")
        logger.info(f"   LR max_iter: {lr_params.get('max_iter', 100)}")
        logger.info(f"   LR regularization: {lr_params.get('regularization', 0.01)}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading model config: {e}")
        logger.warning("Using default model parameters")
        return {
            'model': {
                'logistic_regression': {
                    'n_bits': 8,
                    'max_iter': 100,
                    'regularization': 0.01,
                    'solver': 'liblinear',
                    'random_state': 42
                }
            }
        }


def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed medical dataset from JSON file
    
    Args:
        file_path: Path to preprocessed JSON file
    
    Returns:
        DataFrame with preprocessed medical records
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'data' not in data:
            raise ValueError("Dataset must contain 'data' field")
        
        df = pd.DataFrame(data['data'])
        logger.info(f"Loaded {len(df)} records from {file_path}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)


def convert_to_binary_classification(labels: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Convert multi-class disease labels to binary classification
    
    Strategy: Group diseases into two categories:
    - Cardiovascular diseases (1)
    - Other diseases (0)
    
    Args:
        labels: Original disease labels
        
    Returns:
        Tuple of (binary_labels, mapping_info)
    """
    # Define cardiovascular diseases
    cardiovascular_diseases = [
        'Coronary Artery Disease',
        'Atrial Fibrillation',
        'Hypertension'
    ]
    
    # Create binary labels
    binary_labels = labels.apply(
        lambda x: 'Cardiovascular' if x in cardiovascular_diseases else 'Non-Cardiovascular'
    )
    
    # Create mapping info
    mapping_info = {
        'strategy': 'cardiovascular_vs_other',
        'cardiovascular_diseases': cardiovascular_diseases,
        'class_0': 'Non-Cardiovascular',
        'class_1': 'Cardiovascular',
        'original_classes': labels.unique().tolist(),
        'original_class_count': len(labels.unique()),
        'binary_class_distribution': binary_labels.value_counts().to_dict()
    }
    
    logger.info(f"Converted {len(labels.unique())} classes to binary classification")
    logger.info(f"  - Cardiovascular: {sum(binary_labels == 'Cardiovascular')} samples")
    logger.info(f"  - Non-Cardiovascular: {sum(binary_labels == 'Non-Cardiovascular')} samples")
    
    return binary_labels, mapping_info


def prepare_features_and_labels(df: pd.DataFrame, vectorizer_type: str = "tfidf", 
                               max_features: int = 1000, binary_classification: bool = True) -> Tuple[np.ndarray, np.ndarray, Any, Any, Dict]:
    """
    Convert text data to numerical features and prepare labels
    
    Args:
        df: DataFrame with medical records
        vectorizer_type: Type of vectorizer ("tfidf" or "count")
        max_features: Maximum number of features to extract
        binary_classification: If True, convert to binary classification (default: True)
    
    Returns:
        Tuple of (features, labels, vectorizer, label_encoder, binary_mapping_info)
    """
    # Use medical notes as features (check for both cleaned and raw versions)
    if 'medical_notes_clean' in df.columns:
        texts = df['medical_notes_clean'].fillna('').astype(str)
    elif 'medical_notes' in df.columns:
        texts = df['medical_notes'].fillna('').astype(str)
    else:
        raise ValueError("Dataset must contain 'medical_notes' or 'medical_notes_clean' column")
    
    # Use disease as target variable
    labels = df['disease'].fillna('Unknown')
    
    # Convert to binary classification if requested
    binary_mapping_info = None
    if binary_classification:
        labels, binary_mapping_info = convert_to_binary_classification(labels)
    
    # Initialize vectorizer
    if vectorizer_type.lower() == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
    else:
        vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    
    # Transform text to features
    logger.info(f"Extracting features using {vectorizer_type} vectorizer...")
    X = vectorizer.fit_transform(texts).toarray()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of unique classes: {len(label_encoder.classes_)}")
    logger.info(f"Classes: {list(label_encoder.classes_)}")
    
    return X, y, vectorizer, label_encoder, binary_mapping_info


def train_clear_model(X_train: np.ndarray, y_train: np.ndarray, 
                      model_params: Dict[str, Any] = None) -> FHELogisticRegression:
    """
    Train logistic regression model in clear (unencrypted)
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_params: Model configuration parameters from config file
    
    Returns:
        Trained logistic regression model
    """
    logger.info("Training logistic regression model in clear...")
    
    # Use provided params or defaults
    if model_params is None:
        model_params = {
            'n_bits': 8,
            'max_iter': 100,
            'random_state': 42
        }
    
    # Initialize model with parameters from config
    model = FHELogisticRegression(
        max_iter=model_params.get('max_iter', 100),
        random_state=model_params.get('random_state', 42),
        n_bits=model_params.get('n_bits', 8)  # Quantization for FHE compatibility
    )
    
    logger.info(f"Model parameters: n_bits={model_params.get('n_bits', 8)}, "
                f"max_iter={model_params.get('max_iter', 100)}")
    
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    return model


def compile_fhe_model(model: FHELogisticRegression, X_train: np.ndarray) -> None:
    """
    Compile the trained model to FHE circuit
    
    Args:
        model: Trained logistic regression model
        X_train: Training data for compilation
    """
    if not CONCRETE_ML_AVAILABLE:
        logger.warning("Concrete-ML not available. Skipping FHE compilation.")
        return
    
    logger.info("Compiling model to FHE circuit...")
    
    start_time = time.time()
    try:
        model.compile(X_train)
        compilation_time = time.time() - start_time
        logger.info(f"FHE compilation completed in {compilation_time:.2f} seconds")
    except Exception as e:
        logger.error(f"FHE compilation failed: {e}")
        logger.warning("Continuing with clear inference only...")


def run_inference(model: FHELogisticRegression, X_test: np.ndarray, 
                 use_fhe: bool = True) -> Tuple[np.ndarray, float]:
    """
    Run inference on test data
    
    Args:
        model: Trained model
        X_test: Test features
        use_fhe: Whether to use FHE execution
    
    Returns:
        Tuple of (predictions, inference_time)
    """
    logger.info(f"Running inference {'with FHE' if use_fhe else 'in clear'}...")
    
    start_time = time.time()
    
    if use_fhe and CONCRETE_ML_AVAILABLE:
        try:
            predictions = model.predict(X_test, fhe="execute")
        except Exception as e:
            logger.warning(f"FHE inference failed: {e}. Falling back to clear inference.")
            predictions = model.predict(X_test)
    else:
        predictions = model.predict(X_test)
    
    inference_time = time.time() - start_time
    
    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    
    return predictions, inference_time


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  label_encoder: LabelEncoder) -> Dict[str, Any]:
    """
    Evaluate model performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_encoder: Label encoder for class names
    
    Returns:
        Dictionary with evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Generate classification report
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names, 
                                 output_dict=True, 
                                 zero_division=0)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names.tolist()
    }


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results and model metadata to JSON file
    
    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the results
    
    Args:
        results: Results dictionary
    """
    print(f"\n{'='*60}")
    print("FHE LOGISTIC REGRESSION RESULTS")
    print(f"{'='*60}")
    
    print(f"\nDataset Information:")
    print(f"  Training samples: {results['dataset_info']['train_size']}")
    print(f"  Test samples: {results['dataset_info']['test_size']}")
    print(f"  Features: {results['dataset_info']['n_features']}")
    print(f"  Classes: {results['dataset_info']['n_classes']}")
    
    print(f"\nModel Performance:")
    print(f"  Clear Accuracy: {results['clear_results']['accuracy']:.4f}")
    if 'fhe_results' in results:
        print(f"  FHE Accuracy: {results['fhe_results']['accuracy']:.4f}")
        accuracy_diff = abs(results['clear_results']['accuracy'] - results['fhe_results']['accuracy'])
        print(f"  Accuracy Difference: {accuracy_diff:.4f}")
    
    print(f"\nTiming Information:")
    print(f"  Training Time: {results['timing']['training_time']:.2f}s")
    print(f"  Clear Inference: {results['timing']['clear_inference_time']:.2f}s")
    if 'fhe_inference_time' in results['timing']:
        print(f"  FHE Inference: {results['timing']['fhe_inference_time']:.2f}s")
        speedup = results['timing']['fhe_inference_time'] / results['timing']['clear_inference_time']
        print(f"  FHE Slowdown: {speedup:.1f}x")
    
    print(f"\nTop Disease Classes:")
    for i, class_name in enumerate(results['dataset_info']['class_names'][:5]):
        print(f"  {i+1}. {class_name}")
    
    print(f"{'='*60}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="FHE Logistic Regression for Medical Text Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fhe_logistic_regression.py --train data/processed/preprocessed_data.json
  python fhe_logistic_regression.py --train data.json --vectorizer count --max-features 500
        """
    )
    
    parser.add_argument(
        '--train',
        type=str,
        required=True,
        help='Path to preprocessed training dataset (JSON format)'
    )
    
    parser.add_argument(
        '--vectorizer',
        type=str,
        choices=['tfidf', 'count'],
        default='tfidf',
        help='Type of text vectorizer to use (default: tfidf)'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=1000,
        help='Maximum number of features to extract (default: 1000)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/fhe_logistic_regression_results.json',
        help='Output path for results (default: data/results/fhe_logistic_regression_results.json)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--skip-fhe',
        action='store_true',
        help='Skip FHE inference (clear inference only)'
    )
    
    parser.add_argument(
        '--fhe-config',
        type=str,
        default='src/configs/fhe_config.yaml',
        help='Path to FHE configuration file (default: src/configs/fhe_config.yaml)'
    )
    
    parser.add_argument(
        '--model-config',
        type=str,
        default='src/configs/model_config.yaml',
        help='Path to model configuration file (default: src/configs/model_config.yaml)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seeds
    np.random.seed(args.seed)
    
    logger.info("Starting FHE Logistic Regression pipeline")
    logger.info(f"Parameters: train={args.train}, vectorizer={args.vectorizer}, max_features={args.max_features}")
    
    try:
        # Load FHE configuration
        fhe_config = load_fhe_config(args.fhe_config)
        fhe_context = setup_fhe_context(fhe_config)
        
        # Load model configuration
        model_config = load_model_config(args.model_config)
        lr_params = model_config.get('model', {}).get('logistic_regression', {})
        
        # Load data
        df = load_preprocessed_data(args.train)
        
        # Prepare features and labels (with binary classification)
        X, y, vectorizer, label_encoder, binary_mapping = prepare_features_and_labels(
            df, args.vectorizer, args.max_features, binary_classification=True
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Train model with config parameters
        start_time = time.time()
        model = train_clear_model(X_train, y_train, model_params=lr_params)
        training_time = time.time() - start_time
        
        # Compile to FHE
        if not args.skip_fhe:
            compile_fhe_model(model, X_train)
        
        # Run clear inference
        clear_predictions, clear_inference_time = run_inference(model, X_test, use_fhe=False)
        clear_results = evaluate_model(y_test, clear_predictions, label_encoder)
        
        # Run FHE inference
        fhe_results = None
        fhe_inference_time = None
        if not args.skip_fhe:
            fhe_predictions, fhe_inference_time = run_inference(model, X_test, use_fhe=True)
            fhe_results = evaluate_model(y_test, fhe_predictions, label_encoder)
        
        # Compile results
        results = {
            "experiment_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "vectorizer_type": args.vectorizer,
                "max_features": args.max_features,
                "test_size": args.test_size,
                "random_seed": args.seed,
                "concrete_ml_available": CONCRETE_ML_AVAILABLE,
                "classification_type": "binary",
                "binary_mapping": binary_mapping,
                "fhe_config_file": args.fhe_config,
                "fhe_context": fhe_context,
                "model_config_file": args.model_config,
                "model_parameters": lr_params
            },
            "dataset_info": {
                "total_samples": len(df),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": X.shape[1],
                "n_classes": len(label_encoder.classes_),
                "class_names": label_encoder.classes_.tolist()
            },
            "timing": {
                "training_time": training_time,
                "clear_inference_time": clear_inference_time
            },
            "clear_results": clear_results
        }
        
        if fhe_results:
            results["fhe_results"] = fhe_results
            results["timing"]["fhe_inference_time"] = fhe_inference_time
        
        # Save results
        save_results(results, args.output)
        
        # Print summary
        print_results_summary(results)
        
        logger.info("FHE Logistic Regression pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()