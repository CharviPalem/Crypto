#!/usr/bin/env python3
"""
Privacy Metrics Module

This module provides comprehensive privacy evaluation for datasets including
differential privacy analysis, k-anonymity, l-diversity, and information leakage metrics.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import Counter, defaultdict
import math
import hashlib
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [PrivacyMetrics] %(message)s'
)
logger = logging.getLogger(__name__)


class PrivacyEvaluator:
    """
    Comprehensive privacy evaluation for datasets and anonymization techniques
    
    This class provides methods to evaluate various privacy metrics including
    k-anonymity, l-diversity, information leakage, and differential privacy analysis.
    """
    
    def __init__(self, dp_config_path: str = "src/configs/dp_config.yaml"):
        """Initialize the Privacy Evaluator
        
        Args:
            dp_config_path: Path to DP configuration file
        """
        self.evaluation_results = {}
        self.dp_config = self.load_dp_config(dp_config_path)
        logger.info("Privacy Evaluator initialized")
    
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
                return {'dp': {'epsilon_values': [0.1, 0.5, 1.0, 2.0, 5.0], 'delta': 1e-5}}
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"✅ DP configuration loaded from: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading DP config: {e}")
            return {'dp': {'epsilon_values': [0.1, 0.5, 1.0, 2.0, 5.0], 'delta': 1e-5}}
    
    def compute_k_anonymity(
        self, 
        dataset: Union[pd.DataFrame, List[Dict]], 
        quasi_identifiers: List[str]
    ) -> Dict[str, Any]:
        """
        Compute k-anonymity for a dataset
        
        Args:
            dataset: Dataset as DataFrame or list of dictionaries
            quasi_identifiers: List of column names that are quasi-identifiers
            
        Returns:
            Dictionary with k-anonymity analysis results
        """
        logger.info(f"Computing k-anonymity for {len(quasi_identifiers)} quasi-identifiers")
        
        # Convert to DataFrame if needed
        if isinstance(dataset, list):
            df = pd.DataFrame(dataset)
        else:
            df = dataset.copy()
        
        # Check if quasi-identifiers exist in dataset
        missing_cols = [col for col in quasi_identifiers if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing quasi-identifier columns: {missing_cols}")
            quasi_identifiers = [col for col in quasi_identifiers if col in df.columns]
        
        if not quasi_identifiers:
            return {
                "k_value": 0,
                "privacy_level": "unknown",
                "equivalence_classes": 0,
                "error": "No valid quasi-identifiers found"
            }
        
        # Group by quasi-identifiers to find equivalence classes
        equivalence_classes = df.groupby(quasi_identifiers).size()
        
        # K-anonymity is the minimum group size
        k_value = int(equivalence_classes.min())
        
        # Calculate statistics
        total_classes = len(equivalence_classes)
        avg_class_size = float(equivalence_classes.mean())
        std_class_size = float(equivalence_classes.std())
        
        # Classify privacy level based on k-value
        if k_value >= 5:
            privacy_level = "high"
        elif k_value >= 3:
            privacy_level = "medium"
        else:
            privacy_level = "low"
        
        # Distribution of class sizes
        class_size_dist = equivalence_classes.value_counts().to_dict()
        
        result = {
            "k_value": k_value,
            "privacy_level": privacy_level,
            "total_records": len(df),
            "equivalence_classes": total_classes,
            "avg_class_size": round(avg_class_size, 2),
            "std_class_size": round(std_class_size, 2),
            "quasi_identifiers": quasi_identifiers,
            "class_size_distribution": class_size_dist,
            "suppression_rate": 0.0  # Could be calculated if suppression was applied
        }
        
        logger.info(f"K-anonymity computed: k={k_value}, privacy_level={privacy_level}")
        return result
    
    def compute_l_diversity(
        self, 
        dataset: Union[pd.DataFrame, List[Dict]], 
        sensitive_attribute: str,
        quasi_identifiers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute l-diversity for a dataset
        
        Args:
            dataset: Dataset as DataFrame or list of dictionaries
            sensitive_attribute: Column name of the sensitive attribute
            quasi_identifiers: List of quasi-identifier columns (optional)
            
        Returns:
            Dictionary with l-diversity analysis results
        """
        logger.info(f"Computing l-diversity for sensitive attribute: {sensitive_attribute}")
        
        # Convert to DataFrame if needed
        if isinstance(dataset, list):
            df = pd.DataFrame(dataset)
        else:
            df = dataset.copy()
        
        # Check if sensitive attribute exists
        if sensitive_attribute not in df.columns:
            return {
                "l_value": 0,
                "privacy_level": "unknown",
                "error": f"Sensitive attribute '{sensitive_attribute}' not found"
            }
        
        # If no quasi-identifiers provided, use all columns except sensitive attribute
        if quasi_identifiers is None:
            quasi_identifiers = [col for col in df.columns if col != sensitive_attribute]
        
        # Check quasi-identifiers
        missing_cols = [col for col in quasi_identifiers if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing quasi-identifier columns: {missing_cols}")
            quasi_identifiers = [col for col in quasi_identifiers if col in df.columns]
        
        if not quasi_identifiers:
            # If no quasi-identifiers, compute global diversity
            sensitive_values = df[sensitive_attribute].value_counts()
            l_value = len(sensitive_values)
            min_diversity = l_value
            avg_diversity = l_value
        else:
            # Group by quasi-identifiers
            groups = df.groupby(quasi_identifiers)
            
            diversities = []
            for name, group in groups:
                # Count unique values of sensitive attribute in each group
                unique_sensitive = group[sensitive_attribute].nunique()
                diversities.append(unique_sensitive)
            
            # L-diversity is the minimum diversity across all equivalence classes
            l_value = int(min(diversities)) if diversities else 0
            min_diversity = l_value
            avg_diversity = float(np.mean(diversities)) if diversities else 0.0
        
        # Classify privacy level
        if l_value >= 5:
            privacy_level = "high"
        elif l_value >= 3:
            privacy_level = "medium"
        else:
            privacy_level = "low"
        
        # Additional statistics
        total_sensitive_values = df[sensitive_attribute].nunique()
        sensitive_distribution = df[sensitive_attribute].value_counts().to_dict()
        
        result = {
            "l_value": l_value,
            "privacy_level": privacy_level,
            "total_records": len(df),
            "min_diversity": min_diversity,
            "avg_diversity": round(avg_diversity, 2),
            "total_sensitive_values": total_sensitive_values,
            "sensitive_attribute": sensitive_attribute,
            "quasi_identifiers": quasi_identifiers,
            "sensitive_distribution": sensitive_distribution
        }
        
        logger.info(f"L-diversity computed: l={l_value}, privacy_level={privacy_level}")
        return result
    
    def information_leakage(
        self, 
        original: Union[str, List[str], pd.DataFrame], 
        anonymized: Union[str, List[str], pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Compute information leakage between original and anonymized data
        
        Args:
            original: Original data (string, list of strings, or DataFrame)
            anonymized: Anonymized data (string, list of strings, or DataFrame)
            
        Returns:
            Dictionary with information leakage analysis
        """
        logger.info("Computing information leakage between original and anonymized data")
        
        # Convert inputs to comparable format
        orig_text = self._extract_text_content(original)
        anon_text = self._extract_text_content(anonymized)
        
        if not orig_text or not anon_text:
            return {
                "character_overlap": 0.0,
                "word_overlap": 0.0,
                "leakage_score": 0.0,
                "privacy_level": "unknown",
                "error": "Empty input data"
            }
        
        # Character-level overlap
        char_overlap = self._compute_character_overlap(orig_text, anon_text)
        
        # Word-level overlap
        word_overlap = self._compute_word_overlap(orig_text, anon_text)
        
        # N-gram overlap (bigrams and trigrams)
        bigram_overlap = self._compute_ngram_overlap(orig_text, anon_text, n=2)
        trigram_overlap = self._compute_ngram_overlap(orig_text, anon_text, n=3)
        
        # Overall leakage score (weighted combination)
        leakage_score = (
            0.3 * char_overlap +
            0.4 * word_overlap +
            0.2 * bigram_overlap +
            0.1 * trigram_overlap
        )
        
        # Classify privacy level based on leakage
        if leakage_score <= 0.1:
            privacy_level = "high"
        elif leakage_score <= 0.3:
            privacy_level = "medium"
        else:
            privacy_level = "low"
        
        # Additional metrics
        compression_ratio = len(anon_text) / len(orig_text) if len(orig_text) > 0 else 0.0
        
        result = {
            "character_overlap": round(char_overlap, 4),
            "word_overlap": round(word_overlap, 4),
            "bigram_overlap": round(bigram_overlap, 4),
            "trigram_overlap": round(trigram_overlap, 4),
            "leakage_score": round(leakage_score, 4),
            "privacy_level": privacy_level,
            "compression_ratio": round(compression_ratio, 4),
            "original_length": len(orig_text),
            "anonymized_length": len(anon_text)
        }
        
        logger.info(f"Information leakage computed: score={leakage_score:.4f}, privacy_level={privacy_level}")
        return result
    
    def differential_privacy_analysis(
        self, 
        epsilon: float, 
        delta: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Analyze differential privacy parameters and return privacy level
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability (smaller = more private)
            
        Returns:
            Dictionary with differential privacy analysis
        """
        logger.info(f"Analyzing differential privacy: ε={epsilon}, δ={delta}")
        
        # Classify privacy level based on epsilon
        if epsilon <= 0.1:
            privacy_level = "high"
            privacy_description = "Very strong privacy protection"
        elif epsilon <= 1.0:
            privacy_level = "medium"
            privacy_description = "Good privacy protection"
        elif epsilon <= 10.0:
            privacy_level = "low"
            privacy_description = "Weak privacy protection"
        else:
            privacy_level = "very_low"
            privacy_description = "Minimal privacy protection"
        
        # Additional analysis
        noise_multiplier = self._estimate_noise_multiplier(epsilon, delta)
        privacy_cost = epsilon  # In composition, privacy costs add up
        
        # Estimate utility impact
        if epsilon < 0.1:
            utility_impact = "high"
        elif epsilon < 1.0:
            utility_impact = "medium"
        else:
            utility_impact = "low"
        
        result = {
            "epsilon": epsilon,
            "delta": delta,
            "privacy_level": privacy_level,
            "privacy_description": privacy_description,
            "utility_impact": utility_impact,
            "noise_multiplier": round(noise_multiplier, 4),
            "privacy_cost": privacy_cost,
            "composition_budget": round(1.0 / epsilon, 2) if epsilon > 0 else float('inf'),
            "recommendations": self._generate_dp_recommendations(epsilon, delta)
        }
        
        logger.info(f"Differential privacy analysis: privacy_level={privacy_level}")
        return result
    
    def calculate_dp_privacy_score(
        self, 
        dataset_metadata: Dict[str, Any],
        use_config: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive DP privacy score from dataset metadata using config
        
        Args:
            dataset_metadata: Metadata from preprocessed dataset
            use_config: Whether to use config epsilon values for comparison
            
        Returns:
            Dictionary with DP privacy score and analysis
        """
        logger.info("Calculating DP privacy score from dataset metadata")
        
        # Extract DP parameters from metadata
        epsilon = dataset_metadata.get('epsilon', 1.0)
        delta = dataset_metadata.get('delta', 1e-5)
        sensitivity = dataset_metadata.get('sensitivity', 1.0)
        noise_dist = dataset_metadata.get('noise_distribution', 'laplace')
        
        # Get DP analysis
        dp_analysis = self.differential_privacy_analysis(epsilon, delta)
        
        # Calculate privacy score (0-100, higher = better privacy)
        # Score based on epsilon: lower epsilon = higher score
        if epsilon <= 0.1:
            base_score = 95
        elif epsilon <= 0.5:
            base_score = 85
        elif epsilon <= 1.0:
            base_score = 75
        elif epsilon <= 2.0:
            base_score = 60
        elif epsilon <= 5.0:
            base_score = 40
        else:
            base_score = 20
        
        # Adjust for delta (smaller delta = better)
        if delta <= 1e-6:
            delta_bonus = 5
        elif delta <= 1e-5:
            delta_bonus = 3
        else:
            delta_bonus = 0
        
        # Adjust for noise distribution (Gaussian slightly better for (ε,δ)-DP)
        dist_bonus = 2 if noise_dist == 'gaussian' else 0
        
        privacy_score = min(100, base_score + delta_bonus + dist_bonus)
        
        # Compare with config epsilon values if requested
        config_comparison = {}
        if use_config and 'epsilon_values' in self.dp_config.get('dp', {}):
            config_epsilons = self.dp_config['dp']['epsilon_values']
            config_comparison = {
                'config_epsilon_values': config_epsilons,
                'current_epsilon': epsilon,
                'is_in_config_range': epsilon in config_epsilons,
                'recommended_epsilon': min(config_epsilons) if config_epsilons else 1.0
            }
        
        result = {
            'privacy_score': privacy_score,
            'privacy_grade': self._score_to_grade(privacy_score),
            'epsilon': epsilon,
            'delta': delta,
            'sensitivity': sensitivity,
            'noise_distribution': noise_dist,
            'privacy_level': dp_analysis['privacy_level'],
            'privacy_description': dp_analysis['privacy_description'],
            'utility_impact': dp_analysis['utility_impact'],
            'config_comparison': config_comparison,
            'recommendations': self._generate_dp_score_recommendations(privacy_score, epsilon)
        }
        
        logger.info(f"DP privacy score calculated: {privacy_score}/100 (Grade: {result['privacy_grade']})")
        return result
    
    def _score_to_grade(self, score: float) -> str:
        """Convert privacy score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        else:
            return 'D'
    
    def _generate_dp_score_recommendations(self, score: float, epsilon: float) -> List[str]:
        """Generate recommendations based on privacy score"""
        recommendations = []
        
        if score < 70:
            recommendations.append(f"Consider reducing epsilon (current: {epsilon}) for better privacy")
        if score >= 90:
            recommendations.append("Excellent privacy protection! Monitor utility impact")
        if epsilon > 2.0:
            recommendations.append("Epsilon is high - privacy guarantees are weak")
        if score < 50:
            recommendations.append("CRITICAL: Privacy protection is insufficient for sensitive data")
        
        return recommendations
    
    def _extract_text_content(self, data: Union[str, List[str], pd.DataFrame]) -> str:
        """Extract text content from various data formats"""
        if isinstance(data, str):
            return data
        elif isinstance(data, list):
            return ' '.join(str(item) for item in data)
        elif isinstance(data, pd.DataFrame):
            # Concatenate all text columns
            text_content = []
            for col in data.columns:
                if data[col].dtype == 'object':  # Likely text column
                    text_content.extend(data[col].astype(str).tolist())
            return ' '.join(text_content)
        else:
            return str(data)
    
    def _compute_character_overlap(self, text1: str, text2: str) -> float:
        """Compute character-level overlap between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Convert to sets of characters
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())
        
        # Jaccard similarity
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """Compute word-level overlap between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Split into words and convert to sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_ngram_overlap(self, text1: str, text2: str, n: int) -> float:
        """Compute n-gram overlap between two texts"""
        if not text1 or not text2:
            return 0.0
        
        def get_ngrams(text: str, n: int) -> set:
            words = text.lower().split()
            if len(words) < n:
                return set()
            return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 and not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def _estimate_noise_multiplier(self, epsilon: float, delta: float) -> float:
        """Estimate noise multiplier for Gaussian mechanism"""
        if epsilon <= 0:
            return float('inf')
        
        # Simplified noise multiplier calculation
        # For Gaussian mechanism: σ ≥ √(2 ln(1.25/δ)) / ε
        if delta <= 0:
            delta = 1e-10
        
        noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        return noise_multiplier
    
    def _generate_dp_recommendations(self, epsilon: float, delta: float) -> List[str]:
        """Generate recommendations for differential privacy parameters"""
        recommendations = []
        
        if epsilon > 10:
            recommendations.append("Consider reducing epsilon for stronger privacy protection")
        
        if epsilon < 0.01:
            recommendations.append("Very low epsilon may severely impact utility")
        
        if delta > 1e-3:
            recommendations.append("Consider reducing delta for better privacy guarantees")
        
        if 0.1 <= epsilon <= 1.0:
            recommendations.append("Good balance between privacy and utility")
        
        if not recommendations:
            recommendations.append("Privacy parameters appear reasonable")
        
        return recommendations
    
    def evaluate_comprehensive_privacy(
        self,
        dataset: Union[pd.DataFrame, List[Dict]],
        quasi_identifiers: List[str],
        sensitive_attribute: str,
        original_data: Optional[Union[str, List[str], pd.DataFrame]] = None,
        epsilon: Optional[float] = None,
        delta: Optional[float] = 1e-5
    ) -> Dict[str, Any]:
        """
        Perform comprehensive privacy evaluation
        
        Args:
            dataset: Anonymized dataset
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attribute: Sensitive attribute column
            original_data: Original data for leakage analysis (optional)
            epsilon: Differential privacy epsilon (optional)
            delta: Differential privacy delta (optional)
            
        Returns:
            Comprehensive privacy evaluation results
        """
        logger.info("Performing comprehensive privacy evaluation")
        
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_info": {
                "total_records": len(dataset),
                "total_columns": len(dataset.columns) if isinstance(dataset, pd.DataFrame) else len(dataset[0]) if dataset else 0
            }
        }
        
        # K-anonymity analysis
        try:
            results["k_anonymity"] = self.compute_k_anonymity(dataset, quasi_identifiers)
        except Exception as e:
            logger.error(f"K-anonymity computation failed: {e}")
            results["k_anonymity"] = {"error": str(e)}
        
        # L-diversity analysis
        try:
            results["l_diversity"] = self.compute_l_diversity(dataset, sensitive_attribute, quasi_identifiers)
        except Exception as e:
            logger.error(f"L-diversity computation failed: {e}")
            results["l_diversity"] = {"error": str(e)}
        
        # Information leakage analysis
        if original_data is not None:
            try:
                results["information_leakage"] = self.information_leakage(original_data, dataset)
            except Exception as e:
                logger.error(f"Information leakage computation failed: {e}")
                results["information_leakage"] = {"error": str(e)}
        
        # Differential privacy analysis
        if epsilon is not None:
            try:
                results["differential_privacy"] = self.differential_privacy_analysis(epsilon, delta)
            except Exception as e:
                logger.error(f"Differential privacy analysis failed: {e}")
                results["differential_privacy"] = {"error": str(e)}
        
        # Overall privacy assessment
        results["overall_assessment"] = self._generate_overall_assessment(results)
        
        self.evaluation_results = results
        logger.info("Comprehensive privacy evaluation completed")
        
        return results
    
    def _generate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall privacy assessment"""
        privacy_scores = []
        privacy_levels = []
        
        # Collect privacy levels from different metrics
        for metric in ["k_anonymity", "l_diversity", "information_leakage", "differential_privacy"]:
            if metric in results and "privacy_level" in results[metric]:
                level = results[metric]["privacy_level"]
                privacy_levels.append(level)
                
                # Convert to numeric score
                if level == "high":
                    privacy_scores.append(3)
                elif level == "medium":
                    privacy_scores.append(2)
                elif level == "low":
                    privacy_scores.append(1)
                else:
                    privacy_scores.append(0)
        
        if not privacy_scores:
            return {"overall_privacy_level": "unknown", "confidence": 0.0}
        
        # Calculate overall score
        avg_score = sum(privacy_scores) / len(privacy_scores)
        
        if avg_score >= 2.5:
            overall_level = "high"
        elif avg_score >= 1.5:
            overall_level = "medium"
        else:
            overall_level = "low"
        
        # Calculate confidence based on consistency
        level_counts = Counter(privacy_levels)
        most_common_count = level_counts.most_common(1)[0][1]
        confidence = most_common_count / len(privacy_levels)
        
        return {
            "overall_privacy_level": overall_level,
            "confidence": round(confidence, 2),
            "metrics_evaluated": len(privacy_scores),
            "privacy_score": round(avg_score, 2),
            "level_distribution": dict(level_counts)
        }
    
    def save_evaluation_results(
        self, 
        results: Optional[Dict[str, Any]] = None,
        output_path: str = "data/results/evaluation.json"
    ) -> str:
        """
        Save privacy evaluation results to JSON file
        
        Args:
            results: Results dictionary (uses self.evaluation_results if None)
            output_path: Path to save the results
            
        Returns:
            Path to the saved file
        """
        if results is None:
            results = self.evaluation_results
        
        if not results:
            logger.warning("No evaluation results to save")
            return ""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing evaluation.json if it exists
        evaluation_data = {}
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    evaluation_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing evaluation file: {e}")
        
        # Add privacy metrics to evaluation data
        evaluation_data["privacy_metrics"] = results
        evaluation_data["metadata"] = {
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            "evaluator_version": "1.0.0"
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Privacy evaluation results saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
            raise


# Convenience functions
def evaluate_dataset_privacy(
    dataset: Union[pd.DataFrame, List[Dict]],
    quasi_identifiers: List[str],
    sensitive_attribute: str,
    original_data: Optional[Union[str, List[str], pd.DataFrame]] = None,
    epsilon: Optional[float] = None,
    delta: float = 1e-5,
    output_path: str = "data/results/evaluation.json"
) -> Dict[str, Any]:
    """
    Convenience function to evaluate dataset privacy
    
    Args:
        dataset: Anonymized dataset
        quasi_identifiers: List of quasi-identifier columns
        sensitive_attribute: Sensitive attribute column
        original_data: Original data for leakage analysis (optional)
        epsilon: Differential privacy epsilon (optional)
        delta: Differential privacy delta
        output_path: Path to save results
        
    Returns:
        Privacy evaluation results
    """
    evaluator = PrivacyEvaluator()
    
    results = evaluator.evaluate_comprehensive_privacy(
        dataset=dataset,
        quasi_identifiers=quasi_identifiers,
        sensitive_attribute=sensitive_attribute,
        original_data=original_data,
        epsilon=epsilon,
        delta=delta
    )
    
    evaluator.save_evaluation_results(results, output_path)
    
    return results


if __name__ == "__main__":
    """Example usage of PrivacyEvaluator"""
    
    print("Privacy Metrics Demo")
    print("=" * 50)
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'age_group': np.random.choice(['18-30', '31-50', '51-65', '65+'], n_samples),
        'gender': np.random.choice(['M', 'F', 'Other'], n_samples),
        'zip_code': np.random.choice(['12345', '23456', '34567', '45678'], n_samples),
        'disease': np.random.choice(['Diabetes', 'Hypertension', 'Heart Disease'], n_samples),
        'income_bracket': np.random.choice(['Low', 'Medium', 'High'], n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize evaluator
    evaluator = PrivacyEvaluator()
    
    # Test k-anonymity
    print("\n1. Testing K-Anonymity:")
    k_anon_result = evaluator.compute_k_anonymity(
        df, 
        quasi_identifiers=['age_group', 'gender', 'zip_code']
    )
    print(f"   K-value: {k_anon_result['k_value']}")
    print(f"   Privacy Level: {k_anon_result['privacy_level']}")
    
    # Test l-diversity
    print("\n2. Testing L-Diversity:")
    l_div_result = evaluator.compute_l_diversity(
        df, 
        sensitive_attribute='disease',
        quasi_identifiers=['age_group', 'gender']
    )
    print(f"   L-value: {l_div_result['l_value']}")
    print(f"   Privacy Level: {l_div_result['privacy_level']}")
    
    # Test information leakage
    print("\n3. Testing Information Leakage:")
    original_text = "John Smith lives at 123 Main St and has diabetes"
    anonymized_text = "Person A lives in zip 12345 and has medical condition X"
    
    leakage_result = evaluator.information_leakage(original_text, anonymized_text)
    print(f"   Leakage Score: {leakage_result['leakage_score']}")
    print(f"   Privacy Level: {leakage_result['privacy_level']}")
    
    # Test differential privacy
    print("\n4. Testing Differential Privacy Analysis:")
    dp_result = evaluator.differential_privacy_analysis(epsilon=1.0, delta=1e-5)
    print(f"   Privacy Level: {dp_result['privacy_level']}")
    print(f"   Description: {dp_result['privacy_description']}")
    
    # Comprehensive evaluation
    print("\n5. Comprehensive Privacy Evaluation:")
    comprehensive_result = evaluator.evaluate_comprehensive_privacy(
        dataset=df,
        quasi_identifiers=['age_group', 'gender', 'zip_code'],
        sensitive_attribute='disease',
        original_data=original_text,
        epsilon=1.0
    )
    
    overall = comprehensive_result['overall_assessment']
    print(f"   Overall Privacy Level: {overall['overall_privacy_level']}")
    print(f"   Confidence: {overall['confidence']}")
    
    print("\n✅ Privacy metrics demo completed!")