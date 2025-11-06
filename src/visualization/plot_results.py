#!/usr/bin/env python3
"""
Results Visualization Module

This module creates comprehensive visualizations for evaluation results including
performance metrics, privacy scores, and security analysis from FHE NLP experiments.
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [PlotResults] %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style and suppress warnings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Define consistent color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'info': '#5E548E',
    'warning': '#F4A261',
    'danger': '#E76F51',
    'light': '#F8F9FA',
    'dark': '#343A40'
}

# Color palette for metrics
METRIC_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5E548E', '#F4A261']


class ResultsVisualizer:
    """
    Comprehensive visualization tool for FHE NLP evaluation results
    
    This class creates various plots and charts to visualize performance metrics,
    privacy scores, security analysis, and other evaluation results.
    """
    
    def __init__(self, output_dir: str = "data/results"):
        """
        Initialize the Results Visualizer
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_data = {}
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        logger.info(f"Results Visualizer initialized. Output directory: {self.output_dir}")
    
    def load_evaluation_data(self, input_path: str) -> Dict[str, Any]:
        """
        Load evaluation data from JSON file
        
        Args:
            input_path: Path to evaluation JSON file
            
        Returns:
            Loaded evaluation data
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                self.evaluation_data = json.load(f)
            
            logger.info(f"✅ Evaluation data loaded from: {input_path}")
            return self.evaluation_data
            
        except Exception as e:
            logger.error(f"Failed to load evaluation data: {e}")
            raise ValueError(f"Cannot load evaluation data from {input_path}: {e}")
    
    def load_model_results(self, model_type: str = "logistic_regression") -> Dict[str, Any]:
        """
        Load results from specific FHE model implementations
        
        Args:
            model_type: Type of model ('logistic_regression' or 'svm')
            
        Returns:
            Model-specific results
        """
        result_files = {
            "logistic_regression": self.output_dir / "fhe_logistic_regression_results.json",
            "svm": self.output_dir / "fhe_svm_results.json"
        }
        
        file_path = result_files.get(model_type)
        if not file_path or not file_path.exists():
            logger.warning(f"Results file not found for {model_type}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {model_type} results: {e}")
            return {}
    
    def plot_performance_metrics(self, save_path: Optional[str] = None) -> str:
        """
        Create bar plot of performance metrics
        
        Args:
            save_path: Custom save path (optional)
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating performance metrics bar plot")
        
        # Extract performance metrics
        perf_data = self.evaluation_data.get("performance_metrics", {})
        
        # Define metrics to plot
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        values = [perf_data.get(metric, 0.0) for metric in metrics]
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bars
        bars = ax.bar(labels, values, color=METRIC_COLORS[:len(labels)], 
                     alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Customize the plot
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Metrics', fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Add horizontal reference lines
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
        ax.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Excellent (0.9)')
        
        # Customize appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "performance_metrics.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✅ Performance metrics plot saved to: {save_path}")
        
        plt.close()
        return str(save_path)
    
    def plot_privacy_scores(self, save_path: Optional[str] = None) -> str:
        """
        Create pie chart and bar chart of privacy scores
        
        Args:
            save_path: Custom save path (optional)
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating privacy scores visualization")
        
        # Extract privacy metrics
        privacy_data = self.evaluation_data.get("privacy_metrics", {})
        
        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Privacy levels pie chart
        privacy_levels = []
        privacy_scores = []
        privacy_labels = []
        
        for metric, data in privacy_data.items():
            if isinstance(data, dict) and 'privacy_level' in data:
                privacy_levels.append(data['privacy_level'])
                privacy_labels.append(metric.replace('_', ' ').title())
                
                # Convert privacy level to score
                level = data['privacy_level']
                if level == 'high':
                    privacy_scores.append(3)
                elif level == 'medium':
                    privacy_scores.append(2)
                elif level == 'low':
                    privacy_scores.append(1)
                else:
                    privacy_scores.append(0)
        
        if privacy_levels:
            # Count privacy levels
            level_counts = pd.Series(privacy_levels).value_counts()
            
            # Define colors for privacy levels
            level_colors = {'high': '#2E86AB', 'medium': '#F18F01', 'low': '#E76F51', 'very_low': '#C73E1D'}
            colors = [level_colors.get(level, '#808080') for level in level_counts.index]
            
            # Create pie chart
            wedges, texts, autotexts = ax1.pie(level_counts.values, labels=level_counts.index,
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            
            ax1.set_title('Privacy Levels Distribution', fontweight='bold', pad=20)
            
            # Enhance pie chart appearance
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax1.text(0.5, 0.5, 'No Privacy Data\nAvailable', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=14, alpha=0.5)
            ax1.set_title('Privacy Levels Distribution', fontweight='bold', pad=20)
        
        # Right plot: Privacy scores bar chart
        if privacy_labels and privacy_scores:
            bars = ax2.bar(range(len(privacy_labels)), privacy_scores, 
                          color=METRIC_COLORS[:len(privacy_labels)], alpha=0.8)
            
            ax2.set_xticks(range(len(privacy_labels)))
            ax2.set_xticklabels(privacy_labels, rotation=45, ha='right')
            ax2.set_ylabel('Privacy Score', fontweight='bold')
            ax2.set_title('Privacy Metrics Scores', fontweight='bold', pad=20)
            ax2.set_ylim(0, 3.5)
            
            # Add value labels
            for bar, score in zip(bars, privacy_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{score}', ha='center', va='bottom', fontweight='bold')
            
            # Add reference lines
            ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Low')
            ax2.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Medium')
            ax2.axhline(y=3, color='green', linestyle='--', alpha=0.5, label='High')
            
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No Privacy Scores\nAvailable', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14, alpha=0.5)
            ax2.set_title('Privacy Metrics Scores', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "privacy_scores.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✅ Privacy scores plot saved to: {save_path}")
        
        plt.close()
        return str(save_path)
    
    def plot_security_analysis(self, save_path: Optional[str] = None) -> str:
        """
        Create security analysis visualization
        
        Args:
            save_path: Custom save path (optional)
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating security analysis visualization")
        
        # Extract security metrics
        security_data = self.evaluation_data.get("security_metrics", {})
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Overall security score gauge
        overall_score = security_data.get("overall_security", {}).get("overall_security_score", 0)
        self._create_gauge_chart(ax1, overall_score, "Overall Security Score", max_value=100)
        
        # Plot 2: Attack resistance score gauge
        attack_score = security_data.get("attack_resistance", {}).get("overall_attack_resistance_score", 0)
        self._create_gauge_chart(ax2, attack_score, "Attack Resistance Score", max_value=100)
        
        # Plot 3: Security components bar chart
        security_components = {
            'Key Security': security_data.get("key_security", {}).get("key_security_strength", 0),
            'Noise Budget': min(100, security_data.get("noise_analysis", {}).get("initial_noise_budget_bits", 0) / 4),
            'Overall Score': overall_score,
            'Attack Resistance': attack_score
        }
        
        comp_names = list(security_components.keys())
        comp_values = list(security_components.values())
        
        bars = ax3.barh(comp_names, comp_values, color=METRIC_COLORS[:len(comp_names)], alpha=0.8)
        ax3.set_xlabel('Score', fontweight='bold')
        ax3.set_title('Security Components', fontweight='bold', pad=20)
        ax3.set_xlim(0, 130)
        
        # Add value labels
        for bar, value in zip(bars, comp_values):
            width = bar.get_width()
            ax3.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}', ha='left', va='center', fontweight='bold')
        
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Security level classification
        if overall_score >= 90:
            security_level = "Excellent"
            level_color = '#2E86AB'
        elif overall_score >= 80:
            security_level = "Very Good"
            level_color = '#5E548E'
        elif overall_score >= 70:
            security_level = "Good"
            level_color = '#F18F01'
        elif overall_score >= 60:
            security_level = "Acceptable"
            level_color = '#F4A261'
        else:
            security_level = "Needs Improvement"
            level_color = '#E76F51'
        
        # Create a simple classification display
        ax4.text(0.5, 0.6, f'Security Level:\n{security_level}', 
                ha='center', va='center', fontsize=20, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=level_color, alpha=0.8))
        
        ax4.text(0.5, 0.3, f'Score: {overall_score:.1f}/100', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Security Classification', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "security_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✅ Security analysis plot saved to: {save_path}")
        
        plt.close()
        return str(save_path)
    
    def _create_gauge_chart(self, ax, value: float, title: str, max_value: float = 100):
        """Create a gauge chart for a single metric"""
        # Normalize value to 0-1 range
        normalized_value = value / max_value
        
        # Define colors based on value
        if normalized_value >= 0.8:
            color = '#2E86AB'  # Blue for excellent
        elif normalized_value >= 0.6:
            color = '#5E548E'  # Purple for good
        elif normalized_value >= 0.4:
            color = '#F18F01'  # Orange for fair
        else:
            color = '#E76F51'  # Red for poor
        
        # Create gauge
        theta1 = 0
        theta2 = normalized_value * 180
        
        # Background arc
        wedge_bg = Wedge((0.5, 0.1), 0.4, theta1, 180, 
                        facecolor='lightgray', alpha=0.3)
        ax.add_patch(wedge_bg)
        
        # Value arc
        wedge_val = Wedge((0.5, 0.1), 0.4, theta1, theta2, 
                         facecolor=color, alpha=0.8)
        ax.add_patch(wedge_val)
        
        # Add text
        ax.text(0.5, 0.3, f'{value:.1f}', ha='center', va='center', 
               fontsize=20, fontweight='bold')
        ax.text(0.5, 0.2, f'/ {max_value}', ha='center', va='center', 
               fontsize=12, alpha=0.7)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontweight='bold', pad=20)
    
    def plot_model_comparison(self, save_path: Optional[str] = None) -> str:
        """
        Compare Logistic Regression and SVM model performance
        
        Args:
            save_path: Custom save path (optional)
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating model comparison chart")
        
        # Load both model results
        lr_results = self.load_model_results("logistic_regression")
        svm_results = self.load_model_results("svm")
        
        if not lr_results and not svm_results:
            logger.warning("No model results available for comparison")
            return ""
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Accuracy Comparison
        models = []
        accuracies = []
        if lr_results:
            models.append('Logistic\nRegression')
            accuracies.append(lr_results.get('clear_results', {}).get('accuracy', 0))
        if svm_results:
            models.append('SVM')
            accuracies.append(svm_results.get('clear_results', {}).get('accuracy', 0))
        
        bars = ax1.bar(models, accuracies, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold', pad=20)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Training Time Comparison
        train_times = []
        if lr_results:
            train_times.append(lr_results.get('timing', {}).get('training_time', 0))
        if svm_results:
            train_times.append(svm_results.get('timing', {}).get('training_time', 0))
        
        bars = ax2.bar(models, train_times, color=[COLORS['accent'], COLORS['warning']], alpha=0.8)
        ax2.set_ylabel('Time (seconds)', fontweight='bold')
        ax2.set_title('Training Time Comparison', fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, time in zip(bars, train_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(train_times)*0.02,
                    f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: FHE Inference Time (if available)
        if lr_results and 'fhe_inference_time' in lr_results.get('timing', {}):
            fhe_time_lr = lr_results['timing']['fhe_inference_time']
            clear_time_lr = lr_results['timing']['clear_inference_time']
            
            times = [clear_time_lr, fhe_time_lr]
            labels = ['Clear', 'FHE']
            bars = ax3.bar(labels, times, color=[COLORS['success'], COLORS['danger']], alpha=0.8)
            ax3.set_ylabel('Time (seconds)', fontweight='bold')
            ax3.set_title('Logistic Regression: Clear vs FHE Inference', fontweight='bold', pad=20)
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3, axis='y')
            
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                        f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'FHE Inference\nData Not Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14, alpha=0.5)
            ax3.set_title('FHE Inference Time', fontweight='bold', pad=20)
        
        # Plot 4: Dataset Information
        info_text = "Dataset Information:\n\n"
        if lr_results:
            dataset_info = lr_results.get('dataset_info', {})
            info_text += f"Logistic Regression:\n"
            info_text += f"  • Total Samples: {dataset_info.get('total_samples', 'N/A')}\n"
            info_text += f"  • Features: {dataset_info.get('n_features', 'N/A')}\n"
            info_text += f"  • Classes: {dataset_info.get('n_classes', 'N/A')}\n\n"
        
        if svm_results:
            dataset_info = svm_results.get('dataset_info', {})
            info_text += f"SVM:\n"
            info_text += f"  • Total Samples: {dataset_info.get('total_samples', 'N/A')}\n"
            info_text += f"  • Features: {dataset_info.get('n_features', 'N/A')}\n"
        
        ax4.text(0.1, 0.5, info_text, ha='left', va='center', fontsize=11, family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
        ax4.axis('off')
        ax4.set_title('Dataset Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "model_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✅ Model comparison chart saved to: {save_path}")
        
        plt.close()
        return str(save_path)
    
    def plot_comparison_chart(self, save_path: Optional[str] = None) -> str:
        """
        Create comparison chart of all metrics (radar chart)
        
        Args:
            save_path: Custom save path (optional)
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating comprehensive comparison chart")
        
        # Extract all metrics
        perf_data = self.evaluation_data.get("performance_metrics", {})
        privacy_data = self.evaluation_data.get("privacy_metrics", {})
        security_data = self.evaluation_data.get("security_metrics", {})
        
        # Prepare data for radar chart
        categories = []
        values = []
        
        # Performance metrics (0-1 scale)
        if 'accuracy' in perf_data:
            categories.append('Accuracy')
            values.append(perf_data['accuracy'])
        
        if 'f1_score' in perf_data:
            categories.append('F1-Score')
            values.append(perf_data['f1_score'])
        
        # Privacy score (0-1 scale)
        privacy_score = privacy_data.get("overall_assessment", {}).get("privacy_score", 0)
        if privacy_score > 0:
            categories.append('Privacy')
            values.append(min(1.0, privacy_score / 3.0))  # Normalize to 0-1
        
        # Security score (0-1 scale)
        security_score = security_data.get("overall_security", {}).get("overall_security_score", 0)
        if security_score > 0:
            categories.append('Security')
            values.append(security_score / 100.0)  # Normalize to 0-1
        
        if not categories:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No Data Available\nfor Comparison', 
                   ha='center', va='center', fontsize=16, alpha=0.5)
            ax.set_title('Comprehensive Metrics Comparison', fontweight='bold', pad=20)
            ax.axis('off')
        else:
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Calculate angles for each category
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, color=COLORS['primary'], alpha=0.8)
            ax.fill(angles, values, alpha=0.25, color=COLORS['primary'])
            
            # Customize
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for angle, value, category in zip(angles[:-1], values[:-1], categories):
                ax.text(angle, value + 0.05, f'{value:.2f}', 
                       ha='center', va='center', fontweight='bold', fontsize=10)
            
            ax.set_title('Comprehensive Metrics Comparison', fontweight='bold', pad=30, fontsize=16)
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "comparison_chart.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✅ Comparison chart saved to: {save_path}")
        
        plt.close()
        return str(save_path)
    
    def create_all_plots(self, input_path: str = None) -> List[str]:
        """
        Create all visualization plots
        
        Args:
            input_path: Path to evaluation JSON file (optional)
            
        Returns:
            List of paths to saved plots
        """
        logger.info("Creating all visualization plots")
        
        # Load data if path provided
        if input_path:
            try:
                self.load_evaluation_data(input_path)
            except Exception as e:
                logger.warning(f"Could not load evaluation data: {e}")
        
        # Create all plots
        plot_paths = []
        
        # Model comparison plot (uses individual model result files)
        try:
            model_comp_path = self.plot_model_comparison()
            if model_comp_path:
                plot_paths.append(model_comp_path)
        except Exception as e:
            logger.error(f"Failed to create model comparison plot: {e}")
        
        # Only create these if evaluation data is loaded
        if self.evaluation_data:
            try:
                plot_paths.append(self.plot_performance_metrics())
            except Exception as e:
                logger.error(f"Failed to create performance metrics plot: {e}")
            
            try:
                plot_paths.append(self.plot_privacy_scores())
            except Exception as e:
                logger.error(f"Failed to create privacy scores plot: {e}")
            
            try:
                plot_paths.append(self.plot_security_analysis())
            except Exception as e:
                logger.error(f"Failed to create security analysis plot: {e}")
            
            try:
                plot_paths.append(self.plot_comparison_chart())
            except Exception as e:
                logger.error(f"Failed to create comparison chart: {e}")
        
        logger.info(f"✅ Created {len(plot_paths)} visualization plots")
        return plot_paths
    
    def generate_summary_report(self) -> str:
        """Generate a text summary of the visualizations created"""
        summary_path = self.output_dir / "visualization_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("FHE Project - Visualization Summary\n")
            f.write("=" * 60 + "\n\n")
            
            # Model results summary
            f.write("Implemented Models:\n")
            f.write("-" * 60 + "\n")
            
            lr_results = self.load_model_results("logistic_regression")
            if lr_results:
                f.write("\n1. Logistic Regression (FHE):\n")
                f.write(f"   Accuracy: {lr_results.get('clear_results', {}).get('accuracy', 'N/A')}\n")
                f.write(f"   Training Time: {lr_results.get('timing', {}).get('training_time', 'N/A')}s\n")
                if 'fhe_inference_time' in lr_results.get('timing', {}):
                    f.write(f"   FHE Inference Time: {lr_results['timing']['fhe_inference_time']:.2f}s\n")
                f.write(f"   Dataset: {lr_results.get('dataset_info', {}).get('total_samples', 'N/A')} samples\n")
            
            svm_results = self.load_model_results("svm")
            if svm_results:
                f.write("\n2. SVM (FHE):\n")
                f.write(f"   Accuracy: {svm_results.get('clear_results', {}).get('accuracy', 'N/A')}\n")
                f.write(f"   Training Time: {svm_results.get('timing', {}).get('training_time', 'N/A')}s\n")
                f.write(f"   Dataset: {svm_results.get('dataset_info', {}).get('total_samples', 'N/A')} samples\n")
            
            # Privacy and Security metrics if available
            if self.evaluation_data:
                f.write("\n\nEvaluation Metrics:\n")
                f.write("-" * 60 + "\n")
                
                # Privacy metrics summary
                privacy_data = self.evaluation_data.get("privacy_metrics", {})
                if privacy_data:
                    overall_privacy = privacy_data.get("overall_assessment", {})
                    f.write("\nPrivacy Metrics:\n")
                    f.write(f"  Overall Level: {overall_privacy.get('overall_privacy_level', 'N/A')}\n")
                    f.write(f"  Privacy Score: {overall_privacy.get('privacy_score', 'N/A')}\n")
                
                # Security metrics summary
                security_data = self.evaluation_data.get("security_metrics", {})
                if security_data:
                    overall_security = security_data.get("overall_security", {})
                    f.write("\nSecurity Metrics:\n")
                    f.write(f"  Overall Score: {overall_security.get('overall_security_score', 'N/A')}\n")
                    f.write(f"  Attack Resistance: {security_data.get('attack_resistance', {}).get('overall_attack_resistance_score', 'N/A')}\n")
            
            f.write("\n\nGenerated Visualizations:\n")
            f.write("-" * 60 + "\n")
            f.write("  - model_comparison.png (Logistic Regression vs SVM)\n")
            if self.evaluation_data:
                f.write("  - performance_metrics.png\n")
                f.write("  - privacy_scores.png\n")
                f.write("  - security_analysis.png\n")
                f.write("  - comparison_chart.png (Radar chart)\n")
        
        logger.info(f"✅ Summary report saved to: {summary_path}")
        return str(summary_path)


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="FHE NLP Results Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_results.py --input data/results/evaluation.json
  python plot_results.py --input evaluation.json --output plots/
  python plot_results.py --verbose
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/results/evaluation.json',
        help='Path to evaluation JSON file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/results',
        help='Output directory for plots'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting results visualization...")
        
        # Initialize visualizer
        visualizer = ResultsVisualizer(output_dir=args.output)
        
        # Create all plots
        plot_paths = visualizer.create_all_plots(args.input)
        
        # Generate summary report
        summary_path = visualizer.generate_summary_report()
        
        # Print summary
        print(f"\n{'='*60}")
        print("RESULTS VISUALIZATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 Generated Plots:")
        for i, path in enumerate(plot_paths, 1):
            plot_name = Path(path).name
            print(f"   {i}. {plot_name}")
        
        print(f"\n📁 Output Directory: {args.output}")
        print(f"📄 Summary Report: {summary_path}")
        
        print(f"\n✅ Visualization completed successfully!")
        print(f"   Total plots created: {len(plot_paths)}")
        
        print(f"{'='*60}")
        
        logger.info("✅ Results visualization completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Visualization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()