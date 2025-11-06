#!/usr/bin/env python3
"""
Privacy-Preserving SVM with Homomorphic Encryption

This module implements a privacy-preserving SVM using the "public model + private encrypted data" approach.
The model is trained on public data, but inference is performed on encrypted test data using Pyfhel.
"""

import argparse
import json
import logging
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

# Scikit-learn imports
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer

# Pyfhel imports
try:
    from Pyfhel import Pyfhel, PyCtxt
    PYFHEL_AVAILABLE = True
    print("✅ Using Pyfhel for homomorphic encryption")
except ImportError as e:
    print(f"⚠️ Pyfhel not available: {e}")
    print("Install with: pip install Pyfhel")
    PYFHEL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrivacyPreservingSVM:
    """
    Privacy-Preserving SVM using homomorphic encryption
    
    This class implements the "public model + private encrypted data" approach:
    1. Train SVM on public/clear data
    2. Encrypt test data using Pyfhel
    3. Perform inference on encrypted data
    4. Decrypt results for evaluation
    """
    
    def __init__(self, kernel='linear', C=1.0):
        """
        Initialize the Privacy-Preserving SVM
        
        Args:
            kernel: SVM kernel type (default: 'linear')
            C: Regularization parameter (default: 1.0)
        """
        self.kernel = kernel
        self.C = C
        self.model = None
        self.scaler = StandardScaler()
        self.label_binarizer = LabelBinarizer()
        self.HE = None
        self.is_trained = False
        
        # Initialize Pyfhel if available
        if PYFHEL_AVAILABLE:
            self._setup_homomorphic_encryption()
    
    def _setup_homomorphic_encryption(self):
        """Setup Pyfhel homomorphic encryption context"""
        try:
            self.HE = Pyfhel()
            # Configure for integer operations with sufficient precision
            self.HE.contextGen(scheme='bfv', n=8192, t_bits=20, sec=128)
            self.HE.keyGen()
            self.HE.relinKeyGen()
            self.HE.rotateKeyGen()
            logger.info("Homomorphic encryption context initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup homomorphic encryption: {e}")
            self.HE = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train the SVM model on clear/public data
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training metrics dictionary
        """
        logger.info("Training SVM model on clear data...")
        
        start_time = time.time()
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Handle multi-class classification
        y_train_encoded = self.label_binarizer.fit_transform(y_train)
        if y_train_encoded.shape[1] == 1:
            y_train_encoded = y_train_encoded.ravel()
        
        # Train SVM
        self.model = SVC(kernel=self.kernel, C=self.C, probability=True, random_state=42)
        self.model.fit(X_train_scaled, y_train_encoded)
        
        training_time = time.time() - start_time
        self.is_trained = True
        
        # Calculate training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, train_pred)
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        return {
            "training_time": training_time,
            "training_accuracy": float(train_accuracy),
            "n_support_vectors": int(self.model.n_support_.sum()) if hasattr(self.model, 'n_support_') else 0,
            "kernel": self.kernel,
            "C": self.C
        }
    
    def _encrypt_data(self, X: np.ndarray) -> List[List[PyCtxt]]:
        """
        Encrypt test data using Pyfhel
        
        Args:
            X: Data to encrypt
            
        Returns:
            List of encrypted data samples
        """
        if not PYFHEL_AVAILABLE or self.HE is None:
            raise RuntimeError("Pyfhel not available for encryption")
        
        logger.info(f"Encrypting {X.shape[0]} samples with {X.shape[1]} features each...")
        
        encrypted_data = []
        for i, sample in enumerate(X):
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Encrypted {i + 1}/{X.shape[0]} samples")
            
            # Convert to integers (scale by 1000 for precision)
            sample_int = (sample * 1000).astype(int)
            
            # Encrypt each feature
            encrypted_sample = []
            for feature_val in sample_int:
                encrypted_feature = self.HE.encryptInt(int(feature_val))
                encrypted_sample.append(encrypted_feature)
            
            encrypted_data.append(encrypted_sample)
        
        logger.info("Data encryption completed")
        return encrypted_data
    
    def _decrypt_results(self, encrypted_results: List[PyCtxt]) -> np.ndarray:
        """
        Decrypt prediction results
        
        Args:
            encrypted_results: Encrypted prediction results
            
        Returns:
            Decrypted predictions as numpy array
        """
        if not PYFHEL_AVAILABLE or self.HE is None:
            raise RuntimeError("Pyfhel not available for decryption")
        
        logger.info("Decrypting prediction results...")
        
        decrypted_results = []
        for encrypted_result in encrypted_results:
            decrypted_val = self.HE.decryptInt(encrypted_result)
            # Scale back from integer representation
            decrypted_results.append(decrypted_val / 1000.0)
        
        return np.array(decrypted_results)
    
    def _homomorphic_svm_inference(self, encrypted_data: List[List[PyCtxt]]) -> List[PyCtxt]:
        """
        Perform SVM inference on encrypted data
        
        This simulates the SVM decision function: f(x) = sum(alpha_i * y_i * K(x_i, x)) + b
        For linear kernel: K(x_i, x) = x_i · x (dot product)
        
        Args:
            encrypted_data: Encrypted test samples
            
        Returns:
            Encrypted prediction results
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before inference")
        
        if not PYFHEL_AVAILABLE or self.HE is None:
            raise RuntimeError("Pyfhel not available for homomorphic operations")
        
        logger.info("Performing homomorphic SVM inference...")
        
        # Get SVM parameters
        support_vectors = self.model.support_vectors_
        dual_coef = self.model.dual_coef_[0] if len(self.model.dual_coef_) == 1 else self.model.dual_coef_
        intercept = self.model.intercept_[0] if len(self.model.intercept_) == 1 else self.model.intercept_
        
        # Scale parameters to match encrypted data scaling
        support_vectors_scaled = (support_vectors * 1000).astype(int)
        dual_coef_scaled = (dual_coef * 1000).astype(int)
        intercept_scaled = int(intercept * 1000)
        
        encrypted_results = []
        
        for i, encrypted_sample in enumerate(encrypted_data):
            if (i + 1) % 5 == 0 or i == 0:
                logger.info(f"Processing encrypted sample {i + 1}/{len(encrypted_data)}")
            
            # Initialize decision value with intercept
            decision_value = self.HE.encryptInt(intercept_scaled)
            
            # Compute sum over support vectors: sum(alpha_i * K(x_i, x))
            for j, (sv, coef) in enumerate(zip(support_vectors_scaled, dual_coef_scaled)):
                # Compute dot product K(x_i, x) = x_i · x for linear kernel
                dot_product = self.HE.encryptInt(0)
                
                for k, (sv_feature, encrypted_feature) in enumerate(zip(sv, encrypted_sample)):
                    # Multiply support vector feature with encrypted test feature
                    temp = encrypted_feature * sv_feature
                    dot_product += temp
                
                # Multiply by dual coefficient
                weighted_dot_product = dot_product * coef
                decision_value += weighted_dot_product
            
            encrypted_results.append(decision_value)
        
        logger.info("Homomorphic inference completed")
        return encrypted_results
    
    def predict_clear(self, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform clear (unencrypted) inference
        
        Args:
            X_test: Test features
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before inference")
        
        logger.info("Performing clear inference...")
        
        start_time = time.time()
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        inference_time = time.time() - start_time
        
        logger.info(f"Clear inference completed in {inference_time:.4f} seconds")
        
        return predictions, inference_time
    
    def predict_encrypted(self, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform encrypted inference
        
        Args:
            X_test: Test features
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before inference")
        
        if not PYFHEL_AVAILABLE:
            logger.warning("Pyfhel not available. Falling back to clear inference.")
            return self.predict_clear(X_test)
        
        logger.info("Performing encrypted inference...")
        
        start_time = time.time()
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encrypt test data
        encrypted_data = self._encrypt_data(X_test_scaled)
        
        # Perform homomorphic inference
        encrypted_results = self._homomorphic_svm_inference(encrypted_data)
        
        # Decrypt results
        decrypted_scores = self._decrypt_results(encrypted_results)
        
        # Convert decision scores to binary predictions
        predictions = (decrypted_scores > 0).astype(int)
        
        inference_time = time.time() - start_time
        
        logger.info(f"Encrypted inference completed in {inference_time:.4f} seconds")
        
        return predictions, inference_time


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load specified dataset
    
    Args:
        dataset_name: Name of dataset to load
        
    Returns:
        Tuple of (features, labels, feature_names)
    """
    if dataset_name.lower() == 'diabetes':
        # Convert diabetes regression to classification
        data = load_diabetes()
        X, y = data.data, data.target
        # Convert to binary classification (above/below median)
        y_binary = (y > np.median(y)).astype(int)
        return X, y_binary, data.feature_names
    
    elif dataset_name.lower() == 'breast_cancer':
        data = load_breast_cancer()
        return data.data, data.target, data.feature_names
    
    elif dataset_name.lower() == 'wine':
        data = load_wine()
        # Convert to binary classification (class 0 vs others)
        y_binary = (data.target == 0).astype(int)
        return data.data, y_binary, data.feature_names
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        label_names: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate prediction results
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional label names
        
    Returns:
        Evaluation metrics dictionary
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generate classification report
    target_names = label_names if label_names else ['Class 0', 'Class 1']
    report = classification_report(y_true, y_pred, 
                                 target_names=target_names, 
                                 output_dict=True, 
                                 zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "classification_report": report
    }


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save results to JSON file
    
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
    print("PRIVACY-PRESERVING SVM RESULTS")
    print(f"{'='*60}")
    
    print(f"\nDataset Information:")
    print(f"  Dataset: {results['experiment_info']['dataset']}")
    print(f"  Training samples: {results['dataset_info']['train_size']}")
    print(f"  Test samples: {results['dataset_info']['test_size']}")
    print(f"  Features: {results['dataset_info']['n_features']}")
    
    print(f"\nModel Information:")
    print(f"  Kernel: {results['training_info']['kernel']}")
    print(f"  C parameter: {results['training_info']['C']}")
    print(f"  Support vectors: {results['training_info']['n_support_vectors']}")
    
    print(f"\nPerformance Comparison:")
    print(f"  Clear Accuracy: {results['clear_results']['accuracy']:.4f}")
    if 'encrypted_results' in results:
        print(f"  Encrypted Accuracy: {results['encrypted_results']['accuracy']:.4f}")
        accuracy_diff = abs(results['clear_results']['accuracy'] - results['encrypted_results']['accuracy'])
        print(f"  Accuracy Difference: {accuracy_diff:.4f}")
    
    print(f"\nTiming Comparison:")
    print(f"  Training Time: {results['timing']['training_time']:.4f}s")
    print(f"  Clear Inference: {results['timing']['clear_inference_time']:.4f}s")
    if 'encrypted_inference_time' in results['timing']:
        print(f"  Encrypted Inference: {results['timing']['encrypted_inference_time']:.4f}s")
        slowdown = results['timing']['encrypted_inference_time'] / results['timing']['clear_inference_time']
        print(f"  Encryption Slowdown: {slowdown:.1f}x")
    
    print(f"\nPrivacy Features:")
    print(f"  Homomorphic Encryption: {'✅ Enabled' if results['experiment_info']['pyfhel_available'] else '❌ Disabled'}")
    print(f"  Data Privacy: {'✅ Test data encrypted' if results['experiment_info']['pyfhel_available'] else '❌ Clear inference only'}")
    
    print(f"{'='*60}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving SVM with Homomorphic Encryption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fhe_svm_model.py --dataset diabetes --output data/results/fhe_svm_results.json
  python fhe_svm_model.py --dataset breast_cancer --kernel rbf --C 0.1
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['diabetes', 'breast_cancer', 'wine'],
        default='diabetes',
        help='Dataset to use for demonstration (default: diabetes)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/fhe_svm_results.json',
        help='Output path for results (default: data/results/fhe_svm_results.json)'
    )
    
    parser.add_argument(
        '--kernel',
        type=str,
        choices=['linear', 'rbf', 'poly'],
        default='linear',
        help='SVM kernel type (default: linear)'
    )
    
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='SVM regularization parameter (default: 1.0)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (default: 0.2)'
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
        '--skip-encryption',
        action='store_true',
        help='Skip encrypted inference (clear inference only)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Privacy-Preserving SVM demonstration")
    logger.info(f"Parameters: dataset={args.dataset}, kernel={args.kernel}, C={args.C}")
    
    try:
        # Load dataset
        X, y, feature_names = load_dataset(args.dataset)
        logger.info(f"Loaded {args.dataset} dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Initialize Privacy-Preserving SVM
        ppsvm = PrivacyPreservingSVM(kernel=args.kernel, C=args.C)
        
        # Train model
        training_info = ppsvm.train(X_train, y_train)
        
        # Perform clear inference
        clear_predictions, clear_inference_time = ppsvm.predict_clear(X_test)
        clear_results = evaluate_predictions(y_test, clear_predictions)
        
        # Perform encrypted inference
        encrypted_results = None
        encrypted_inference_time = None
        if not args.skip_encryption and PYFHEL_AVAILABLE:
            try:
                encrypted_predictions, encrypted_inference_time = ppsvm.predict_encrypted(X_test)
                encrypted_results = evaluate_predictions(y_test, encrypted_predictions)
            except Exception as e:
                logger.error(f"Encrypted inference failed: {e}")
                logger.warning("Continuing with clear inference results only")
        
        # Compile results
        results = {
            "experiment_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": args.dataset,
                "kernel": args.kernel,
                "C": args.C,
                "test_size": args.test_size,
                "random_seed": args.seed,
                "pyfhel_available": PYFHEL_AVAILABLE
            },
            "dataset_info": {
                "total_samples": len(X),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": X.shape[1],
                "feature_names": list(feature_names) if feature_names is not None else []
            },
            "training_info": training_info,
            "timing": {
                "training_time": training_info["training_time"],
                "clear_inference_time": clear_inference_time
            },
            "clear_results": clear_results
        }
        
        if encrypted_results:
            results["encrypted_results"] = encrypted_results
            results["timing"]["encrypted_inference_time"] = encrypted_inference_time
        
        # Save results
        save_results(results, args.output)
        
        # Print summary
        print_results_summary(results)
        
        logger.info("Privacy-Preserving SVM demonstration completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()