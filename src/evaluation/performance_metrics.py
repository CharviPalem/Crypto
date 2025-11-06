#!/usr/bin/env python3
"""
Performance Metrics Module

This module provides comprehensive performance evaluation for machine learning models
including accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and 
classification reports using sklearn.metrics.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    log_loss
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [PerformanceMetrics] %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation class for machine learning models
    
    This class provides methods to calculate various performance metrics,
    save results, and generate detailed performance reports.
    """
    
    def __init__(self, task_type: str = "binary", class_names: Optional[List[str]] = None):
        """
        Initialize the Performance Evaluator
        
        Args:
            task_type: Type of classification task ("binary", "multiclass", "multilabel")
            class_names: List of class names for better reporting
        """
        self.task_type = task_type.lower()
        self.class_names = class_names or []
        self.supported_tasks = ["binary", "multiclass", "multilabel"]
        
        if self.task_type not in self.supported_tasks:
            raise ValueError(f"Task type must be one of {self.supported_tasks}")
        
        logger.info(f"PerformanceEvaluator initialized for {self.task_type} classification")
    
    def evaluate(
        self, 
        y_true: Union[np.ndarray, List], 
        y_pred: Union[np.ndarray, List],
        y_pred_proba: Optional[Union[np.ndarray, List]] = None,
        average: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
            average: Averaging strategy for multiclass metrics
        
        Returns:
            Dictionary containing all performance metrics
        """
        logger.info("Calculating performance metrics...")
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_pred_proba is not None:
            y_pred_proba = np.array(y_pred_proba)
        
        # Validate inputs
        self._validate_inputs(y_true, y_pred, y_pred_proba)
        
        # Calculate metrics
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(y_true, y_pred, average))
        
        # Confusion matrix
        metrics.update(self._calculate_confusion_matrix(y_true, y_pred))
        
        # Classification report
        metrics.update(self._calculate_classification_report(y_true, y_pred))
        
        # ROC-AUC and probability-based metrics
        if y_pred_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_pred_proba))
        
        # Additional metrics
        metrics.update(self._calculate_additional_metrics(y_true, y_pred))
        
        # Summary statistics
        metrics["summary"] = self._generate_summary(metrics)
        
        logger.info("✅ Performance metrics calculation completed")
        return metrics
    
    def _validate_inputs(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: Optional[np.ndarray]
    ) -> None:
        """Validate input arrays"""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if y_pred_proba is not None and len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have the same length")
        
        # Check for valid labels
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        if self.task_type == "binary" and (len(unique_true) > 2 or len(unique_pred) > 2):
            logger.warning("More than 2 classes detected for binary classification")
    
    def _calculate_basic_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        average: str
    ) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        metrics = {}
        
        # Accuracy
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        
        # Balanced accuracy
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        
        # Precision, Recall, F1-score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.task_type == "binary":
                # Binary classification
                metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
                metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
                
                # Per-class metrics
                metrics["precision_per_class"] = precision_score(
                    y_true, y_pred, average=None, zero_division=0
                ).tolist()
                metrics["recall_per_class"] = recall_score(
                    y_true, y_pred, average=None, zero_division=0
                ).tolist()
                metrics["f1_score_per_class"] = f1_score(
                    y_true, y_pred, average=None, zero_division=0
                ).tolist()
                
            else:
                # Multiclass classification
                metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
                metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
                metrics["f1_score"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
                
                # Per-class metrics
                metrics["precision_per_class"] = precision_score(
                    y_true, y_pred, average=None, zero_division=0
                ).tolist()
                metrics["recall_per_class"] = recall_score(
                    y_true, y_pred, average=None, zero_division=0
                ).tolist()
                metrics["f1_score_per_class"] = f1_score(
                    y_true, y_pred, average=None, zero_division=0
                ).tolist()
                
                # Macro and micro averages
                metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
                metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
                metrics["f1_score_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                
                metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
                metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
                metrics["f1_score_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        
        return metrics
    
    def _calculate_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate confusion matrix and related metrics"""
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
        }
        
        # For binary classification, extract TP, TN, FP, FN
        if self.task_type == "binary" and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "positive_predictive_value": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                "negative_predictive_value": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            })
        
        return metrics
    
    def _calculate_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate detailed classification report"""
        target_names = self.class_names if self.class_names else None
        
        # Get classification report as dictionary
        report_dict = classification_report(
            y_true, y_pred, 
            target_names=target_names, 
            output_dict=True, 
            zero_division=0
        )
        
        return {"classification_report": report_dict}
    
    def _calculate_probability_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate probability-based metrics like ROC-AUC"""
        metrics = {}
        
        try:
            if self.task_type == "binary":
                # Binary ROC-AUC
                if y_pred_proba.ndim == 1:
                    proba_positive = y_pred_proba
                else:
                    proba_positive = y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba[:, 0]
                
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_positive))
                
                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_true, proba_positive)
                metrics["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist()
                }
                
                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_true, proba_positive)
                metrics["precision_recall_curve"] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist()
                }
                
                # Area under PR curve
                metrics["pr_auc"] = float(auc(recall, precision))
                
                # Log loss
                metrics["log_loss"] = float(log_loss(y_true, proba_positive))
                
            elif self.task_type == "multiclass":
                # Multiclass ROC-AUC
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_pred_proba, multi_class="ovr"))
                metrics["roc_auc_ovo"] = float(roc_auc_score(y_true, y_pred_proba, multi_class="ovo"))
                
                # Log loss
                metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
                
        except Exception as e:
            logger.warning(f"Could not calculate probability-based metrics: {e}")
        
        return metrics
    
    def _calculate_additional_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate additional performance metrics"""
        metrics = {}
        
        try:
            # Matthews Correlation Coefficient
            metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
            
            # Cohen's Kappa
            metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
            
            # Class distribution
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            
            metrics["class_distribution"] = {
                "true": dict(zip(unique_true.tolist(), counts_true.tolist())),
                "predicted": dict(zip(unique_pred.tolist(), counts_pred.tolist()))
            }
            
            # Sample counts
            metrics["sample_counts"] = {
                "total_samples": len(y_true),
                "unique_classes_true": len(unique_true),
                "unique_classes_pred": len(unique_pred)
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate additional metrics: {e}")
        
        return metrics
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of key metrics"""
        summary = {
            "accuracy": metrics.get("accuracy", 0.0),
            "f1_score": metrics.get("f1_score", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "task_type": self.task_type
        }
        
        if "roc_auc" in metrics:
            summary["roc_auc"] = metrics["roc_auc"]
        
        if "matthews_corrcoef" in metrics:
            summary["matthews_corrcoef"] = metrics["matthews_corrcoef"]
        
        return summary
    
    def save_results(
        self, 
        metrics: Dict[str, Any], 
        path: Union[str, Path],
        include_timestamp: bool = True
    ) -> str:
        """
        Save performance metrics to JSON file
        
        Args:
            metrics: Dictionary of performance metrics
            path: Path to save the results
            include_timestamp: Whether to include timestamp in the results
        
        Returns:
            Path to the saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        results = {
            "metadata": {
                "task_type": self.task_type,
                "class_names": self.class_names,
                "evaluator_version": "1.0.0"
            },
            "metrics": metrics
        }
        
        if include_timestamp:
            import datetime
            results["metadata"]["timestamp"] = datetime.datetime.now().isoformat()
        
        # Save to JSON
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Performance metrics saved to: {path}")
            return str(path)
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            raise
    
    def print_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Print a formatted summary of performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
        """
        print(f"\n{'='*60}")
        print("PERFORMANCE EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        # Basic metrics
        print(f"\n📊 Basic Metrics:")
        print(f"   Accuracy:           {metrics.get('accuracy', 0.0):.4f}")
        print(f"   Balanced Accuracy:  {metrics.get('balanced_accuracy', 0.0):.4f}")
        print(f"   Precision:          {metrics.get('precision', 0.0):.4f}")
        print(f"   Recall:             {metrics.get('recall', 0.0):.4f}")
        print(f"   F1-Score:           {metrics.get('f1_score', 0.0):.4f}")
        
        # Additional metrics
        if "roc_auc" in metrics:
            print(f"   ROC-AUC:            {metrics['roc_auc']:.4f}")
        
        if "matthews_corrcoef" in metrics:
            print(f"   Matthews Corr:      {metrics['matthews_corrcoef']:.4f}")
        
        if "cohen_kappa" in metrics:
            print(f"   Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")
        
        # Confusion Matrix
        if "confusion_matrix" in metrics:
            print(f"\n🔢 Confusion Matrix:")
            cm = np.array(metrics["confusion_matrix"])
            for i, row in enumerate(cm):
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                print(f"   {class_name:>12}: {row}")
        
        # Binary classification specific metrics
        if self.task_type == "binary" and "sensitivity" in metrics:
            print(f"\n🎯 Binary Classification Metrics:")
            print(f"   Sensitivity (TPR):  {metrics['sensitivity']:.4f}")
            print(f"   Specificity (TNR):  {metrics['specificity']:.4f}")
            print(f"   PPV (Precision):    {metrics['positive_predictive_value']:.4f}")
            print(f"   NPV:                {metrics['negative_predictive_value']:.4f}")
        
        # Per-class metrics for multiclass
        if self.task_type == "multiclass" and "f1_score_per_class" in metrics:
            print(f"\n📈 Per-Class F1-Scores:")
            f1_scores = metrics["f1_score_per_class"]
            for i, f1 in enumerate(f1_scores):
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                print(f"   {class_name:>12}: {f1:.4f}")
        
        # Sample information
        if "sample_counts" in metrics:
            counts = metrics["sample_counts"]
            print(f"\n📋 Dataset Information:")
            print(f"   Total Samples:      {counts['total_samples']}")
            print(f"   Unique Classes:     {counts['unique_classes_true']}")
        
        # Class distribution
        if "class_distribution" in metrics:
            print(f"\n📊 Class Distribution:")
            true_dist = metrics["class_distribution"]["true"]
            for class_label, count in true_dist.items():
                class_name = self.class_names[class_label] if class_label < len(self.class_names) else f"Class {class_label}"
                percentage = (count / metrics["sample_counts"]["total_samples"]) * 100
                print(f"   {class_name:>12}: {count:>6} ({percentage:5.1f}%)")
        
        print(f"{'='*60}")
    
    def compare_models(
        self, 
        metrics_list: List[Dict[str, Any]], 
        model_names: List[str]
    ) -> Dict[str, Any]:
        """
        Compare performance metrics across multiple models
        
        Args:
            metrics_list: List of metrics dictionaries
            model_names: List of model names
        
        Returns:
            Comparison results dictionary
        """
        if len(metrics_list) != len(model_names):
            raise ValueError("Number of metrics and model names must match")
        
        comparison = {
            "models": model_names,
            "comparison_metrics": {}
        }
        
        # Compare key metrics
        key_metrics = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
        
        for metric in key_metrics:
            values = []
            for metrics in metrics_list:
                values.append(metrics.get(metric, None))
            
            if any(v is not None for v in values):
                comparison["comparison_metrics"][metric] = {
                    "values": values,
                    "best_model": model_names[np.nanargmax(values)] if any(v is not None for v in values) else None,
                    "best_value": np.nanmax([v for v in values if v is not None]) if any(v is not None for v in values) else None
                }
        
        return comparison
    
    def print_comparison(self, comparison: Dict[str, Any]) -> None:
        """Print model comparison results"""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        models = comparison["models"]
        metrics = comparison["comparison_metrics"]
        
        print(f"\n{'Metric':<15} {'Best Model':<15} {'Best Value':<12} {'All Values'}")
        print("-" * 60)
        
        for metric_name, metric_data in metrics.items():
            best_model = metric_data["best_model"]
            best_value = metric_data["best_value"]
            all_values = metric_data["values"]
            
            values_str = " | ".join([f"{v:.4f}" if v is not None else "N/A" for v in all_values])
            
            print(f"{metric_name:<15} {best_model:<15} {best_value:<12.4f} {values_str}")
        
        print(f"{'='*60}")


# Convenience functions
def evaluate_model(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    y_pred_proba: Optional[Union[np.ndarray, List]] = None,
    task_type: str = "binary",
    class_names: Optional[List[str]] = None,
    print_results: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate model performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        task_type: Type of classification task
        class_names: List of class names
        print_results: Whether to print results summary
        save_path: Path to save results (optional)
    
    Returns:
        Dictionary of performance metrics
    """
    evaluator = PerformanceEvaluator(task_type=task_type, class_names=class_names)
    metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba)
    
    if print_results:
        evaluator.print_summary(metrics)
    
    if save_path:
        evaluator.save_results(metrics, save_path)
    
    return metrics


# Example usage and testing
if __name__ == "__main__":
    """Example usage of PerformanceEvaluator"""
    
    print("Performance Metrics Demo")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Binary classification example
    print("\n1. Binary Classification Example:")
    y_true_binary = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    y_pred_binary = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    y_pred_proba_binary = np.random.random(n_samples)
    
    evaluator_binary = PerformanceEvaluator(
        task_type="binary", 
        class_names=["Negative", "Positive"]
    )
    
    metrics_binary = evaluator_binary.evaluate(
        y_true_binary, y_pred_binary, y_pred_proba_binary
    )
    
    evaluator_binary.print_summary(metrics_binary)
    
    # Multiclass classification example
    print("\n2. Multiclass Classification Example:")
    y_true_multi = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    y_pred_multi = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.4, 0.2])
    y_pred_proba_multi = np.random.random((n_samples, 3))
    y_pred_proba_multi = y_pred_proba_multi / y_pred_proba_multi.sum(axis=1, keepdims=True)
    
    evaluator_multi = PerformanceEvaluator(
        task_type="multiclass", 
        class_names=["Class A", "Class B", "Class C"]
    )
    
    metrics_multi = evaluator_multi.evaluate(
        y_true_multi, y_pred_multi, y_pred_proba_multi
    )
    
    evaluator_multi.print_summary(metrics_multi)
    
    # Model comparison example
    print("\n3. Model Comparison Example:")
    comparison = evaluator_binary.compare_models(
        [metrics_binary, metrics_multi], 
        ["Binary Model", "Multi Model"]
    )
    
    evaluator_binary.print_comparison(comparison)
    
    print("\n✅ Performance metrics demo completed!")