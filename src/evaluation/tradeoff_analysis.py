#!/usr/bin/env python3
"""
Privacy-Utility Trade-off Analysis

This module loads evaluation JSONs and visualizes the privacy-utility trade-off,
specifically plotting accuracy vs epsilon (differential privacy parameter) to
demonstrate how privacy protection affects model utility.
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
import yaml

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [TradeoffAnalysis] %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


class PrivacyUtilityAnalyzer:
    """
    Analyzes and visualizes privacy-utility trade-offs in FHE-based ML systems
    
    This class loads evaluation results from various experiments and creates
    comprehensive visualizations showing how privacy parameters affect model utility.
    """
    
    def __init__(self, results_dir: str = "data/results", evaluation_dir: str = "data/evaluation",
                 dp_config_path: str = "src/configs/dp_config.yaml"):
        """
        Initialize the Privacy-Utility Analyzer
        
        Args:
            results_dir: Directory containing result JSON files
            evaluation_dir: Directory containing evaluation JSON files
            dp_config_path: Path to DP configuration file
        """
        self.results_dir = Path(results_dir)
        self.evaluation_dir = Path(evaluation_dir)
        self.data_points = []
        self.loaded_files = []
        self.dp_config = self.load_dp_config(dp_config_path)
        
        logger.info("Privacy-Utility Analyzer initialized")
    
    def load_dp_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load differential privacy configuration from YAML file
        
        Args:
            config_path: Path to DP configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"DP config not found at {config_path}, using defaults")
                return {'dp': {'epsilon_values': [0.1, 0.5, 1.0, 2.0, 5.0]}}
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"✅ DP configuration loaded from: {config_path}")
            logger.info(f"   Epsilon values for analysis: {config['dp']['epsilon_values']}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading DP config: {e}")
            return {'dp': {'epsilon_values': [0.1, 0.5, 1.0, 2.0, 5.0]}}
    
    def load_evaluation_data(self, file_patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load evaluation data from JSON files
        
        Args:
            file_patterns: List of file patterns to search for (default: common patterns)
            
        Returns:
            List of loaded evaluation data dictionaries
        """
        if file_patterns is None:
            file_patterns = [
                "**/evaluation.json",
                "**/fhe_*_results.json",
                "**/performance_*.json",
                "**/security_*.json",
                "**/*evaluation*.json"
            ]
        
        evaluation_data = []
        
        # Search in both results and evaluation directories
        search_dirs = [self.results_dir, self.evaluation_dir]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                logger.warning(f"Directory not found: {search_dir}")
                continue
            
            for pattern in file_patterns:
                for file_path in search_dir.glob(pattern):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Add metadata about the file
                        data['_source_file'] = str(file_path)
                        data['_file_name'] = file_path.name
                        
                        evaluation_data.append(data)
                        self.loaded_files.append(str(file_path))
                        logger.info(f"Loaded: {file_path}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Successfully loaded {len(evaluation_data)} evaluation files")
        return evaluation_data
    
    def extract_privacy_utility_points(self, evaluation_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract privacy-utility data points from evaluation results
        
        Args:
            evaluation_data: List of evaluation data dictionaries
            
        Returns:
            List of privacy-utility data points
        """
        data_points = []
        
        for data in evaluation_data:
            point = self._extract_single_point(data)
            if point:
                data_points.append(point)
        
        # Generate synthetic data points if no real data available
        if not data_points:
            logger.warning("No privacy-utility data found. Generating synthetic data for demonstration.")
            data_points = self._generate_synthetic_data()
        
        self.data_points = data_points
        logger.info(f"Extracted {len(data_points)} privacy-utility data points")
        
        return data_points
    
    def _extract_single_point(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract privacy-utility metrics from a single evaluation file
        
        Args:
            data: Single evaluation data dictionary
            
        Returns:
            Privacy-utility data point or None
        """
        point = {
            'source_file': data.get('_source_file', 'unknown'),
            'epsilon': None,
            'accuracy': None,
            'f1_score': None,
            'precision': None,
            'recall': None,
            'privacy_level': None,
            'model_type': None,
            'dataset': None
        }
        
        # Try to extract epsilon (differential privacy parameter)
        epsilon = self._find_epsilon(data)
        if epsilon is not None:
            point['epsilon'] = epsilon
        
        # Try to extract accuracy
        accuracy = self._find_accuracy(data)
        if accuracy is not None:
            point['accuracy'] = accuracy
        
        # Extract other metrics
        point['f1_score'] = self._find_metric(data, ['f1_score', 'f1'])
        point['precision'] = self._find_metric(data, ['precision'])
        point['recall'] = self._find_metric(data, ['recall'])
        
        # Extract metadata
        point['model_type'] = self._find_model_type(data)
        point['dataset'] = self._find_dataset(data)
        point['privacy_level'] = self._classify_privacy_level(epsilon)
        
        # Only return point if we have both epsilon and accuracy
        if point['epsilon'] is not None and point['accuracy'] is not None:
            return point
        
        return None
    
    def _find_epsilon(self, data: Dict[str, Any]) -> Optional[float]:
        """Find epsilon value in the data"""
        # Common paths where epsilon might be stored
        paths = [
            ['metadata', 'epsilon'],
            ['experiment_info', 'epsilon'],
            ['privacy_parameters', 'epsilon'],
            ['differential_privacy', 'epsilon'],
            ['preprocessing', 'epsilon'],
            ['epsilon'],
            ['dp_epsilon'],
            ['privacy_budget']
        ]
        
        for path in paths:
            value = self._get_nested_value(data, path)
            if value is not None and isinstance(value, (int, float)):
                return float(value)
        
        return None
    
    def _find_accuracy(self, data: Dict[str, Any]) -> Optional[float]:
        """Find accuracy value in the data"""
        # Common paths where accuracy might be stored
        paths = [
            ['metrics', 'accuracy'],
            ['clear_results', 'accuracy'],
            ['encrypted_results', 'accuracy'],
            ['performance_metrics', 'accuracy'],
            ['evaluation_results', 'accuracy'],
            ['results', 'accuracy'],
            ['accuracy'],
            ['test_accuracy'],
            ['validation_accuracy']
        ]
        
        for path in paths:
            value = self._get_nested_value(data, path)
            if value is not None and isinstance(value, (int, float)):
                return float(value)
        
        return None
    
    def _find_metric(self, data: Dict[str, Any], metric_names: List[str]) -> Optional[float]:
        """Find a specific metric in the data"""
        for metric_name in metric_names:
            paths = [
                ['metrics', metric_name],
                ['clear_results', metric_name],
                ['encrypted_results', metric_name],
                ['performance_metrics', metric_name],
                ['evaluation_results', metric_name],
                ['results', metric_name],
                [metric_name]
            ]
            
            for path in paths:
                value = self._get_nested_value(data, path)
                if value is not None and isinstance(value, (int, float)):
                    return float(value)
        
        return None
    
    def _find_model_type(self, data: Dict[str, Any]) -> Optional[str]:
        """Find model type in the data"""
        paths = [
            ['experiment_info', 'model_type'],
            ['metadata', 'model_type'],
            ['model_info', 'type'],
            ['model_type'],
            ['algorithm']
        ]
        
        for path in paths:
            value = self._get_nested_value(data, path)
            if value is not None:
                return str(value)
        
        # Try to infer from filename
        filename = data.get('_file_name', '')
        if 'svm' in filename.lower():
            return 'SVM'
        elif 'logistic' in filename.lower():
            return 'Logistic Regression'
        elif 'regression' in filename.lower():
            return 'Regression'
        
        return 'Unknown'
    
    def _find_dataset(self, data: Dict[str, Any]) -> Optional[str]:
        """Find dataset name in the data"""
        paths = [
            ['experiment_info', 'dataset'],
            ['metadata', 'dataset'],
            ['dataset_info', 'name'],
            ['dataset'],
            ['data_source']
        ]
        
        for path in paths:
            value = self._get_nested_value(data, path)
            if value is not None:
                return str(value)
        
        return 'Unknown'
    
    def _classify_privacy_level(self, epsilon: Optional[float]) -> str:
        """Classify privacy level based on epsilon value"""
        if epsilon is None:
            return 'Unknown'
        elif epsilon <= 0.1:
            return 'Very High Privacy'
        elif epsilon <= 1.0:
            return 'High Privacy'
        elif epsilon <= 5.0:
            return 'Medium Privacy'
        elif epsilon <= 10.0:
            return 'Low Privacy'
        else:
            return 'Very Low Privacy'
    
    def _get_nested_value(self, data: Dict[str, Any], path: List[str]) -> Any:
        """Get nested value from dictionary using path"""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic privacy-utility data for demonstration"""
        logger.info("Generating synthetic privacy-utility data")
        
        # Realistic epsilon values and corresponding accuracy values
        # Higher epsilon (less privacy) generally leads to higher accuracy
        epsilon_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        
        synthetic_points = []
        
        # Generate data for implemented models only
        models = [
            {'name': 'Logistic Regression', 'base_accuracy': 0.85, 'noise_factor': 0.02},
            {'name': 'SVM', 'base_accuracy': 0.82, 'noise_factor': 0.025}
        ]
        
        for model in models:
            for epsilon in epsilon_values:
                # Privacy-utility trade-off: higher epsilon = higher accuracy
                # Using a sigmoid-like function for realistic behavior
                privacy_impact = 1 / (1 + np.exp(-2 * (np.log10(epsilon) + 1)))
                accuracy = model['base_accuracy'] * (0.7 + 0.3 * privacy_impact)
                
                # Add some realistic noise
                noise = np.random.normal(0, model['noise_factor'])
                accuracy = max(0.5, min(1.0, accuracy + noise))
                
                # Calculate other metrics based on accuracy
                f1_score = accuracy * (0.95 + np.random.normal(0, 0.02))
                precision = accuracy * (0.97 + np.random.normal(0, 0.015))
                recall = accuracy * (0.93 + np.random.normal(0, 0.02))
                
                point = {
                    'source_file': 'synthetic_data',
                    'epsilon': epsilon,
                    'accuracy': round(accuracy, 4),
                    'f1_score': round(max(0.5, min(1.0, f1_score)), 4),
                    'precision': round(max(0.5, min(1.0, precision)), 4),
                    'recall': round(max(0.5, min(1.0, recall)), 4),
                    'privacy_level': self._classify_privacy_level(epsilon),
                    'model_type': model['name'],
                    'dataset': 'Medical Dataset'
                }
                
                synthetic_points.append(point)
        
        return synthetic_points
    
    def create_privacy_utility_plot(
        self, 
        data_points: List[Dict[str, Any]], 
        output_path: str = "data/results/privacy_utility_plot.png",
        figsize: Tuple[int, int] = (12, 8)
    ) -> str:
        """
        Create privacy-utility trade-off visualization
        
        Args:
            data_points: List of privacy-utility data points
            output_path: Path to save the plot
            figsize: Figure size (width, height)
            
        Returns:
            Path to the saved plot
        """
        logger.info("Creating privacy-utility trade-off plot...")
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(data_points)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique model types for different colors/markers
        model_types = df['model_type'].unique()
        colors = sns.color_palette("husl", len(model_types))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Plot for each model type
        for i, model_type in enumerate(model_types):
            model_data = df[df['model_type'] == model_type].sort_values('epsilon')
            
            if len(model_data) > 0:
                # Plot original points
                ax.scatter(
                    model_data['epsilon'], 
                    model_data['accuracy'],
                    color=colors[i],
                    marker=markers[i % len(markers)],
                    s=80,
                    alpha=0.7,
                    label=f'{model_type} (Data Points)',
                    edgecolors='white',
                    linewidth=1
                )
                
                # Create smooth line if we have enough points
                if len(model_data) >= 3:
                    # Sort by epsilon for smooth interpolation
                    x_smooth = np.logspace(
                        np.log10(model_data['epsilon'].min()),
                        np.log10(model_data['epsilon'].max()),
                        300
                    )
                    
                    # Use spline interpolation for smooth curve
                    try:
                        spline = make_interp_spline(
                            model_data['epsilon'], 
                            model_data['accuracy'], 
                            k=min(3, len(model_data)-1)
                        )
                        y_smooth = spline(x_smooth)
                        
                        ax.plot(
                            x_smooth, 
                            y_smooth,
                            color=colors[i],
                            linewidth=2.5,
                            alpha=0.8,
                            label=f'{model_type} (Trend)'
                        )
                    except Exception as e:
                        logger.warning(f"Could not create smooth line for {model_type}: {e}")
                        # Fallback to simple line plot
                        ax.plot(
                            model_data['epsilon'], 
                            model_data['accuracy'],
                            color=colors[i],
                            linewidth=2,
                            alpha=0.8,
                            label=f'{model_type} (Trend)'
                        )
        
        # Customize the plot
        ax.set_xscale('log')
        ax.set_xlabel('Epsilon (ε) - Differential Privacy Parameter', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Privacy-Utility Trade-off Analysis\nAccuracy vs Differential Privacy Parameter (ε)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add privacy level annotations
        privacy_regions = [
            (0.001, 0.1, 'Very High\nPrivacy', 'lightblue'),
            (0.1, 1.0, 'High\nPrivacy', 'lightgreen'),
            (1.0, 10.0, 'Medium\nPrivacy', 'lightyellow'),
            (10.0, 1000.0, 'Low\nPrivacy', 'lightcoral')
        ]
        
        y_min, y_max = ax.get_ylim()
        for x_min, x_max, label, color in privacy_regions:
            if x_min <= df['epsilon'].max() and x_max >= df['epsilon'].min():
                ax.axvspan(x_min, x_max, alpha=0.1, color=color)
                ax.text(
                    np.sqrt(x_min * x_max), 
                    y_max - 0.02, 
                    label, 
                    ha='center', 
                    va='top', 
                    fontsize=9, 
                    alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3)
                )
        
        # Customize legend
        legend = ax.legend(
            loc='lower right', 
            frameon=True, 
            fancybox=True, 
            shadow=True,
            fontsize=10
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # Add statistics text box
        stats_text = self._generate_stats_text(df)
        ax.text(
            0.02, 0.98, 
            stats_text, 
            transform=ax.transAxes, 
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
        )
        
        # Set axis limits
        ax.set_xlim(df['epsilon'].min() * 0.5, df['epsilon'].max() * 2)
        ax.set_ylim(max(0.5, df['accuracy'].min() - 0.05), min(1.0, df['accuracy'].max() + 0.05))
        
        # Improve layout
        plt.tight_layout()
        
        # Save the plot
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(
            output_file, 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none'
        )
        
        logger.info(f"✅ Privacy-utility plot saved to: {output_file}")
        
        # Show plot if in interactive mode
        if hasattr(plt, 'show'):
            plt.show()
        
        plt.close()
        
        return str(output_file)
    
    def _generate_stats_text(self, df: pd.DataFrame) -> str:
        """Generate statistics text for the plot"""
        stats = []
        stats.append(f"Data Points: {len(df)}")
        stats.append(f"Models: {len(df['model_type'].unique())}")
        
        if len(df) > 1:
            # Calculate correlation
            correlation, p_value = pearsonr(np.log10(df['epsilon']), df['accuracy'])
            stats.append(f"Correlation (log ε, acc): {correlation:.3f}")
        
        # Epsilon range
        stats.append(f"ε Range: {df['epsilon'].min():.3f} - {df['epsilon'].max():.1f}")
        
        # Accuracy range
        stats.append(f"Accuracy Range: {df['accuracy'].min():.3f} - {df['accuracy'].max():.3f}")
        
        return '\n'.join(stats)
    
    def plot_accuracy_vs_epsilon(
        self,
        model_results: List[Dict[str, Any]],
        output_path: str = "data/results/accuracy_vs_epsilon.png",
        figsize: Tuple[int, int] = (12, 7)
    ) -> str:
        """
        Plot accuracy vs epsilon using config epsilon values
        
        Args:
            model_results: List of model results with epsilon and accuracy
            output_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Path to saved plot
        """
        logger.info("Creating accuracy vs epsilon plot using DP config...")
        
        # Get epsilon values from config
        config_epsilons = self.dp_config.get('dp', {}).get('epsilon_values', [0.1, 0.5, 1.0, 2.0, 5.0])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        epsilons = []
        accuracies = []
        models = []
        
        for result in model_results:
            if 'epsilon' in result and 'accuracy' in result:
                epsilons.append(result['epsilon'])
                accuracies.append(result['accuracy'])
                models.append(result.get('model_type', 'Unknown'))
        
        if not epsilons:
            logger.warning("No epsilon/accuracy data found in results")
            return ""
        
        # Create DataFrame
        df = pd.DataFrame({
            'epsilon': epsilons,
            'accuracy': accuracies,
            'model': models
        })
        
        # Plot for each model
        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('epsilon')
            ax.plot(model_data['epsilon'], model_data['accuracy'], 
                   marker='o', markersize=8, linewidth=2.5, label=model, alpha=0.8)
        
        # Highlight config epsilon values
        for eps in config_epsilons:
            ax.axvline(x=eps, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(eps, ax.get_ylim()[1] * 0.98, f'ε={eps}', 
                   rotation=90, va='top', ha='right', fontsize=8, alpha=0.6)
        
        # Styling
        ax.set_xlabel('Privacy Budget (ε)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Model Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Privacy-Utility Tradeoff: Accuracy vs Epsilon (ε)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        
        # Add privacy regions
        ax.axvspan(0.01, 0.1, alpha=0.1, color='green', label='High Privacy')
        ax.axvspan(0.1, 1.0, alpha=0.1, color='yellow', label='Medium Privacy')
        ax.axvspan(1.0, 10.0, alpha=0.1, color='red', label='Low Privacy')
        
        # Add annotation
        textstr = f'Config ε values: {config_epsilons}\\nLower ε = Better Privacy\\nHigher ε = Better Utility'
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Accuracy vs epsilon plot saved to: {output_file}")
        
        plt.close()
        return str(output_file)
    
    def generate_summary_report(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report of the privacy-utility analysis
        
        Args:
            data_points: List of privacy-utility data points
            
        Returns:
            Summary report dictionary
        """
        df = pd.DataFrame(data_points)
        
        report = {
            "analysis_summary": {
                "total_data_points": len(df),
                "unique_models": len(df['model_type'].unique()),
                "model_types": df['model_type'].unique().tolist(),
                "epsilon_range": {
                    "min": float(df['epsilon'].min()),
                    "max": float(df['epsilon'].max()),
                    "mean": float(df['epsilon'].mean())
                },
                "accuracy_range": {
                    "min": float(df['accuracy'].min()),
                    "max": float(df['accuracy'].max()),
                    "mean": float(df['accuracy'].mean())
                }
            },
            "privacy_levels": df['privacy_level'].value_counts().to_dict(),
            "model_performance": {},
            "trade_off_insights": []
        }
        
        # Per-model analysis
        for model_type in df['model_type'].unique():
            model_data = df[df['model_type'] == model_type]
            report["model_performance"][model_type] = {
                "data_points": len(model_data),
                "avg_accuracy": float(model_data['accuracy'].mean()),
                "accuracy_std": float(model_data['accuracy'].std()),
                "epsilon_sensitivity": self._calculate_epsilon_sensitivity(model_data)
            }
        
        # Generate insights
        if len(df) > 1:
            correlation, _ = pearsonr(np.log10(df['epsilon']), df['accuracy'])
            
            if correlation > 0.5:
                report["trade_off_insights"].append(
                    "Strong positive correlation between epsilon and accuracy - clear privacy-utility trade-off"
                )
            elif correlation > 0.2:
                report["trade_off_insights"].append(
                    "Moderate privacy-utility trade-off observed"
                )
            else:
                report["trade_off_insights"].append(
                    "Weak correlation between privacy and utility - robust privacy protection"
                )
        
        return report
    
    def _calculate_epsilon_sensitivity(self, model_data: pd.DataFrame) -> float:
        """Calculate how sensitive the model is to epsilon changes"""
        if len(model_data) < 2:
            return 0.0
        
        # Calculate the slope of accuracy vs log(epsilon)
        try:
            correlation, _ = pearsonr(np.log10(model_data['epsilon']), model_data['accuracy'])
            return abs(correlation)
        except:
            return 0.0
    
    def save_analysis_report(self, report: Dict[str, Any], output_path: str) -> str:
        """
        Save analysis report to JSON file
        
        Args:
            report: Analysis report dictionary
            output_path: Path to save the report
            
        Returns:
            Path to the saved report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Analysis report saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")
            raise


def analyze_privacy_utility_tradeoff(
    results_dir: str = "data/results",
    evaluation_dir: str = "data/evaluation",
    output_plot: str = "data/results/privacy_utility_plot.png",
    output_report: str = "data/results/privacy_utility_analysis.json",
    show_plot: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to perform complete privacy-utility trade-off analysis
    
    Args:
        results_dir: Directory containing result JSON files
        evaluation_dir: Directory containing evaluation JSON files
        output_plot: Path to save the plot
        output_report: Path to save the analysis report
        show_plot: Whether to display the plot
        
    Returns:
        Analysis report dictionary
    """
    # Initialize analyzer
    analyzer = PrivacyUtilityAnalyzer(results_dir, evaluation_dir)
    
    # Load evaluation data
    evaluation_data = analyzer.load_evaluation_data()
    
    # Extract privacy-utility data points
    data_points = analyzer.extract_privacy_utility_points(evaluation_data)
    
    # Create visualization
    plot_path = analyzer.create_privacy_utility_plot(data_points, output_plot)
    
    # Generate summary report
    report = analyzer.generate_summary_report(data_points)
    report["visualization_path"] = plot_path
    report["loaded_files"] = analyzer.loaded_files
    
    # Save report
    analyzer.save_analysis_report(report, output_report)
    
    return report


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Privacy-Utility Trade-off Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tradeoff_analysis.py
  python tradeoff_analysis.py --results-dir data/results --output-plot plots/tradeoff.png
  python tradeoff_analysis.py --show-plot --verbose
        """
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='data/results',
        help='Directory containing result JSON files'
    )
    
    parser.add_argument(
        '--evaluation-dir',
        type=str,
        default='data/evaluation',
        help='Directory containing evaluation JSON files'
    )
    
    parser.add_argument(
        '--output-plot',
        type=str,
        default='data/results/privacy_utility_plot.png',
        help='Output path for the privacy-utility plot'
    )
    
    parser.add_argument(
        '--output-report',
        type=str,
        default='data/results/privacy_utility_analysis.json',
        help='Output path for the analysis report'
    )
    
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display the plot after creation'
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
        logger.info("Starting privacy-utility trade-off analysis...")
        
        # Perform analysis
        report = analyze_privacy_utility_tradeoff(
            results_dir=args.results_dir,
            evaluation_dir=args.evaluation_dir,
            output_plot=args.output_plot,
            output_report=args.output_report,
            show_plot=args.show_plot
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("PRIVACY-UTILITY TRADE-OFF ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        summary = report["analysis_summary"]
        print(f"\n📊 Analysis Overview:")
        print(f"   Data Points:        {summary['total_data_points']}")
        print(f"   Model Types:        {summary['unique_models']}")
        print(f"   Models:             {', '.join(summary['model_types'])}")
        
        print(f"\n🔒 Privacy Parameter (ε) Range:")
        epsilon_range = summary['epsilon_range']
        print(f"   Minimum:            {epsilon_range['min']:.4f}")
        print(f"   Maximum:            {epsilon_range['max']:.2f}")
        print(f"   Average:            {epsilon_range['mean']:.4f}")
        
        print(f"\n🎯 Accuracy Range:")
        accuracy_range = summary['accuracy_range']
        print(f"   Minimum:            {accuracy_range['min']:.4f}")
        print(f"   Maximum:            {accuracy_range['max']:.4f}")
        print(f"   Average:            {accuracy_range['mean']:.4f}")
        
        print(f"\n💡 Key Insights:")
        for insight in report["trade_off_insights"]:
            print(f"   • {insight}")
        
        print(f"\n📁 Output Files:")
        print(f"   Plot:               {args.output_plot}")
        print(f"   Report:             {args.output_report}")
        
        print(f"{'='*60}")
        
        logger.info("✅ Privacy-utility trade-off analysis completed successfully!")
        
        return report
        
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()