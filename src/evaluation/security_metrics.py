#!/usr/bin/env python3
"""
Security Metrics Module for FHE Analysis

This module computes encryption-related security analysis for Fully Homomorphic Encryption (FHE)
systems. It evaluates security parameters, estimates attack resistance, and provides comprehensive
security assessments based on FHE configuration parameters.
"""

import argparse
import json
import logging
import math
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [SecurityMetrics] %(message)s'
)
logger = logging.getLogger(__name__)


class FHESecurityAnalyzer:
    """
    Comprehensive security analysis for FHE systems
    
    This class evaluates various security aspects of FHE configurations including
    key sizes, polynomial degrees, noise levels, and attack resistance estimates.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the FHE Security Analyzer
        
        Args:
            config_path: Path to FHE configuration YAML file
        """
        self.config_path = config_path
        self.config = {}
        self.security_params = {}
        
        if config_path:
            self.load_config(config_path)
        
        logger.info("FHE Security Analyzer initialized")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load FHE configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using default parameters")
                self.config = self._get_default_config()
                return self.config
            
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"✅ Configuration loaded from: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.info("Using default configuration parameters")
            self.config = self._get_default_config()
        
        return self.config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default FHE configuration parameters"""
        return {
            "fhe_parameters": {
                "scheme": "bfv",
                "polynomial_degree": 8192,
                "plaintext_modulus_bits": 20,
                "ciphertext_modulus_bits": 438,
                "security_level": 128,
                "noise_standard_deviation": 3.2,
                "key_switching_decomposition": 60,
                "relinearization_decomposition": 60
            },
            "security_settings": {
                "enable_key_switching": True,
                "enable_relinearization": True,
                "enable_rotation": True,
                "bootstrap_precision": 20
            },
            "performance_settings": {
                "parallelization": True,
                "optimization_level": 2
            }
        }
    
    def analyze_security(self, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive security analysis
        
        Args:
            custom_params: Optional custom parameters to override config
            
        Returns:
            Dictionary containing security analysis results
        """
        logger.info("Starting FHE security analysis...")
        
        # Use custom parameters if provided, otherwise use config
        params = custom_params if custom_params else self.config.get("fhe_parameters", {})
        
        security_metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": params,
            "key_security": self._analyze_key_security(params),
            "polynomial_security": self._analyze_polynomial_security(params),
            "noise_analysis": self._analyze_noise_security(params),
            "attack_resistance": self._calculate_attack_resistance(params),
            "overall_security": {}
        }
        
        # Calculate overall security score
        security_metrics["overall_security"] = self._calculate_overall_security(security_metrics)
        
        logger.info("✅ Security analysis completed")
        return security_metrics
    
    def evaluate_encryption_strength(self) -> Dict[str, Any]:
        """
        Evaluate encryption strength based on FHE config parameters
        
        Returns:
            Dictionary with encryption strength evaluation
        """
        logger.info("Evaluating encryption strength from config...")
        
        fhe_params = self.config.get('fhe_parameters', {})
        
        # Extract key parameters
        poly_degree = fhe_params.get('polynomial_degree', 8192)
        security_level = fhe_params.get('security_level', 128)
        scheme = fhe_params.get('scheme', 'bfv')
        plaintext_modulus = fhe_params.get('plaintext_modulus_bits', 20)
        
        # Evaluate strength
        if security_level >= 256:
            strength_level = "Very High"
            strength_score = 95
        elif security_level >= 192:
            strength_level = "High"
            strength_score = 85
        elif security_level >= 128:
            strength_level = "Medium-High"
            strength_score = 75
        else:
            strength_level = "Medium"
            strength_score = 60
        
        # Adjust for polynomial degree
        if poly_degree >= 16384:
            strength_score += 5
        elif poly_degree < 4096:
            strength_score -= 10
        
        strength_score = min(100, max(0, strength_score))
        
        result = {
            'encryption_strength_score': strength_score,
            'strength_level': strength_level,
            'security_level_bits': security_level,
            'polynomial_degree': poly_degree,
            'scheme': scheme,
            'plaintext_modulus': plaintext_modulus,
            'quantum_resistant': security_level >= 128,
            'recommendations': []
        }
        
        # Add recommendations
        if security_level < 128:
            result['recommendations'].append("Increase security level to at least 128 bits")
        if poly_degree < 8192:
            result['recommendations'].append("Consider increasing polynomial degree for better security")
        if security_level >= 128:
            result['recommendations'].append("Current parameters provide quantum-resistant security")
        
        logger.info(f"Encryption strength: {strength_score}/100 ({strength_level})")
        return result
    
    def _analyze_key_security(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze key-related security metrics
        
        Args:
            params: FHE parameters
            
        Returns:
            Key security analysis
        """
        poly_degree = params.get("polynomial_degree", 8192)
        security_level = params.get("security_level", 128)
        ctxt_modulus_bits = params.get("ciphertext_modulus_bits", 438)
        
        # Calculate key sizes (estimates based on FHE literature)
        public_key_size_bits = poly_degree * ctxt_modulus_bits
        secret_key_size_bits = poly_degree  # Binary/ternary secret key
        
        # Convert to bytes and MB
        public_key_size_bytes = public_key_size_bits // 8
        secret_key_size_bytes = secret_key_size_bits // 8
        
        public_key_size_mb = public_key_size_bytes / (1024 * 1024)
        secret_key_size_mb = secret_key_size_bytes / (1024 * 1024)
        
        # Key security strength estimation
        key_security_strength = min(security_level, self._estimate_key_strength(poly_degree, ctxt_modulus_bits))
        
        return {
            "public_key_size": {
                "bits": public_key_size_bits,
                "bytes": public_key_size_bytes,
                "megabytes": round(public_key_size_mb, 3)
            },
            "secret_key_size": {
                "bits": secret_key_size_bits,
                "bytes": secret_key_size_bytes,
                "megabytes": round(secret_key_size_mb, 6)
            },
            "key_security_strength": key_security_strength,
            "key_generation_complexity": self._estimate_keygen_complexity(poly_degree),
            "key_switching_security": self._analyze_key_switching_security(params)
        }
    
    def _analyze_polynomial_security(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze polynomial-related security metrics
        
        Args:
            params: FHE parameters
            
        Returns:
            Polynomial security analysis
        """
        poly_degree = params.get("polynomial_degree", 8192)
        ctxt_modulus_bits = params.get("ciphertext_modulus_bits", 438)
        
        # Ring-LWE security estimation
        ring_lwe_security = self._estimate_ring_lwe_security(poly_degree, ctxt_modulus_bits)
        
        # Polynomial attack resistance
        lattice_attack_resistance = self._estimate_lattice_attack_resistance(poly_degree, ctxt_modulus_bits)
        
        return {
            "polynomial_degree": poly_degree,
            "ring_dimension": poly_degree,
            "cyclotomic_polynomial": f"X^{poly_degree} + 1",
            "ring_lwe_security_level": ring_lwe_security,
            "lattice_attack_resistance": lattice_attack_resistance,
            "polynomial_factorization_resistance": self._estimate_factorization_resistance(poly_degree),
            "recommended_min_degree": 4096,
            "security_margin": max(0, poly_degree - 4096) / 4096 * 100
        }
    
    def _analyze_noise_security(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze noise-related security metrics
        
        Args:
            params: FHE parameters
            
        Returns:
            Noise security analysis
        """
        noise_std = params.get("noise_standard_deviation", 3.2)
        ctxt_modulus_bits = params.get("ciphertext_modulus_bits", 438)
        plaintext_modulus_bits = params.get("plaintext_modulus_bits", 20)
        
        # Calculate noise budget
        initial_noise_budget = ctxt_modulus_bits - plaintext_modulus_bits - math.log2(noise_std) - 10  # Safety margin
        
        # Noise growth estimation
        noise_growth_per_mult = self._estimate_noise_growth(params)
        max_multiplicative_depth = max(1, int(initial_noise_budget / noise_growth_per_mult))
        
        # Noise flooding resistance
        noise_flooding_resistance = self._estimate_noise_flooding_resistance(noise_std, ctxt_modulus_bits)
        
        return {
            "noise_standard_deviation": noise_std,
            "initial_noise_budget_bits": round(initial_noise_budget, 2),
            "noise_growth_per_multiplication": round(noise_growth_per_mult, 2),
            "estimated_multiplicative_depth": max_multiplicative_depth,
            "noise_flooding_resistance": noise_flooding_resistance,
            "fresh_ciphertext_noise_level": round(math.log2(noise_std), 2),
            "noise_analysis": {
                "low_noise_threshold": 10,  # bits
                "critical_noise_threshold": 5,  # bits
                "current_safety_margin": round(initial_noise_budget - 10, 2)
            }
        }
    
    def _calculate_attack_resistance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate resistance against various cryptographic attacks
        
        Args:
            params: FHE parameters
            
        Returns:
            Attack resistance analysis with scores 0-100
        """
        poly_degree = params.get("polynomial_degree", 8192)
        ctxt_modulus_bits = params.get("ciphertext_modulus_bits", 438)
        security_level = params.get("security_level", 128)
        
        # Calculate individual attack resistances
        attacks = {
            "lattice_reduction_attack": self._calculate_lattice_attack_resistance(poly_degree, ctxt_modulus_bits),
            "lwe_distinguishing_attack": self._calculate_lwe_attack_resistance(poly_degree, ctxt_modulus_bits),
            "algebraic_attack": self._calculate_algebraic_attack_resistance(poly_degree),
            "side_channel_resistance": self._calculate_side_channel_resistance(params),
            "quantum_attack_resistance": self._calculate_quantum_resistance(poly_degree, ctxt_modulus_bits),
            "noise_analysis_attack": self._calculate_noise_attack_resistance(params),
            "key_recovery_attack": self._calculate_key_recovery_resistance(params)
        }
        
        # Calculate overall attack resistance score
        resistance_scores = list(attacks.values())
        overall_score = sum(resistance_scores) / len(resistance_scores)
        
        return {
            "individual_attacks": attacks,
            "overall_attack_resistance_score": round(overall_score, 2),
            "security_classification": self._classify_security_level(overall_score),
            "recommended_improvements": self._generate_security_recommendations(attacks, params)
        }
    
    def _calculate_overall_security(self, security_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall security assessment
        
        Args:
            security_metrics: Complete security analysis
            
        Returns:
            Overall security summary
        """
        # Extract key metrics
        key_strength = security_metrics["key_security"]["key_security_strength"]
        ring_lwe_security = security_metrics["polynomial_security"]["ring_lwe_security_level"]
        noise_budget = security_metrics["noise_analysis"]["initial_noise_budget_bits"]
        attack_resistance = security_metrics["attack_resistance"]["overall_attack_resistance_score"]
        
        # Weighted overall score
        weights = {
            "key_strength": 0.25,
            "ring_lwe_security": 0.30,
            "noise_budget": 0.20,
            "attack_resistance": 0.25
        }
        
        # Normalize scores to 0-100 scale
        normalized_scores = {
            "key_strength": min(100, (key_strength / 128) * 100),
            "ring_lwe_security": min(100, (ring_lwe_security / 128) * 100),
            "noise_budget": min(100, max(0, (noise_budget / 100) * 100)),
            "attack_resistance": attack_resistance
        }
        
        overall_score = sum(weights[k] * normalized_scores[k] for k in weights.keys())
        
        return {
            "overall_security_score": round(overall_score, 2),
            "component_scores": normalized_scores,
            "security_level_classification": self._classify_security_level(overall_score),
            "compliance_status": self._check_compliance(security_metrics),
            "risk_assessment": self._assess_security_risks(overall_score, security_metrics)
        }
    
    # Helper methods for security calculations
    
    def _estimate_key_strength(self, poly_degree: int, modulus_bits: int) -> float:
        """Estimate key security strength based on parameters"""
        # Simplified estimation based on lattice security
        log_q = modulus_bits
        log_n = math.log2(poly_degree)
        
        # Conservative estimate using LWE security formulas
        security_estimate = min(128, (log_q * log_n) / 4)
        return max(80, security_estimate)  # Minimum 80-bit security
    
    def _estimate_ring_lwe_security(self, poly_degree: int, modulus_bits: int) -> float:
        """Estimate Ring-LWE security level"""
        # Based on lattice estimator formulas (simplified)
        n = poly_degree
        log_q = modulus_bits
        
        # Conservative Ring-LWE security estimation
        security_level = min(256, math.sqrt(n) * math.log2(n) + log_q / 8)
        return max(80, security_level)
    
    def _estimate_lattice_attack_resistance(self, poly_degree: int, modulus_bits: int) -> float:
        """Estimate resistance to lattice reduction attacks"""
        # Simplified BKZ complexity estimation
        dimension = poly_degree
        log_q = modulus_bits
        
        # BKZ-β complexity: 2^(0.292β) for β = dimension/2
        beta = dimension // 2
        complexity_bits = 0.292 * beta
        
        return min(128, complexity_bits)
    
    def _estimate_noise_growth(self, params: Dict[str, Any]) -> float:
        """Estimate noise growth per multiplication"""
        poly_degree = params.get("polynomial_degree", 8192)
        
        # Simplified noise growth estimation
        # Actual growth depends on specific FHE scheme implementation
        base_growth = math.log2(poly_degree) + 2
        return base_growth
    
    def _estimate_noise_flooding_resistance(self, noise_std: float, modulus_bits: int) -> float:
        """Estimate resistance to noise flooding attacks"""
        noise_to_modulus_ratio = math.log2(noise_std) / modulus_bits
        
        # Higher ratio means better resistance (up to a point)
        resistance = min(100, (1 - noise_to_modulus_ratio) * 100)
        return max(0, resistance)
    
    def _calculate_lattice_attack_resistance(self, poly_degree: int, modulus_bits: int) -> float:
        """Calculate lattice attack resistance score (0-100)"""
        security_bits = self._estimate_lattice_attack_resistance(poly_degree, modulus_bits)
        return min(100, (security_bits / 128) * 100)
    
    def _calculate_lwe_attack_resistance(self, poly_degree: int, modulus_bits: int) -> float:
        """Calculate LWE distinguishing attack resistance (0-100)"""
        security_bits = self._estimate_ring_lwe_security(poly_degree, modulus_bits)
        return min(100, (security_bits / 128) * 100)
    
    def _calculate_algebraic_attack_resistance(self, poly_degree: int) -> float:
        """Calculate algebraic attack resistance (0-100)"""
        # Resistance increases with polynomial degree
        if poly_degree >= 16384:
            return 95
        elif poly_degree >= 8192:
            return 85
        elif poly_degree >= 4096:
            return 75
        else:
            return 60
    
    def _calculate_side_channel_resistance(self, params: Dict[str, Any]) -> float:
        """Calculate side-channel attack resistance (0-100)"""
        # This would depend on implementation details
        # For now, return a conservative estimate
        return 70  # Moderate resistance assumed
    
    def _calculate_quantum_resistance(self, poly_degree: int, modulus_bits: int) -> float:
        """Calculate quantum attack resistance (0-100)"""
        # FHE is generally quantum-resistant, but strength varies
        security_bits = self._estimate_ring_lwe_security(poly_degree, modulus_bits)
        
        # Quantum algorithms provide square root speedup
        post_quantum_security = security_bits / 2
        
        return min(100, (post_quantum_security / 64) * 100)
    
    def _calculate_noise_attack_resistance(self, params: Dict[str, Any]) -> float:
        """Calculate noise analysis attack resistance (0-100)"""
        noise_std = params.get("noise_standard_deviation", 3.2)
        
        # Higher noise provides better resistance
        if noise_std >= 3.0:
            return 90
        elif noise_std >= 2.0:
            return 75
        else:
            return 60
    
    def _calculate_key_recovery_resistance(self, params: Dict[str, Any]) -> float:
        """Calculate key recovery attack resistance (0-100)"""
        poly_degree = params.get("polynomial_degree", 8192)
        
        # Based on polynomial degree and secret key distribution
        if poly_degree >= 16384:
            return 95
        elif poly_degree >= 8192:
            return 85
        else:
            return 70
    
    def _estimate_keygen_complexity(self, poly_degree: int) -> str:
        """Estimate key generation complexity"""
        complexity_ops = poly_degree * math.log2(poly_degree)
        
        if complexity_ops > 1000000:
            return "High"
        elif complexity_ops > 100000:
            return "Medium"
        else:
            return "Low"
    
    def _analyze_key_switching_security(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze key switching security"""
        decomp_bits = params.get("key_switching_decomposition", 60)
        
        return {
            "decomposition_base_bits": decomp_bits,
            "security_impact": "Medium" if decomp_bits >= 60 else "High",
            "noise_growth_factor": 2 ** (decomp_bits / 10)
        }
    
    def _estimate_factorization_resistance(self, poly_degree: int) -> str:
        """Estimate polynomial factorization resistance"""
        if poly_degree >= 8192:
            return "Very High"
        elif poly_degree >= 4096:
            return "High"
        else:
            return "Medium"
    
    def _classify_security_level(self, score: float) -> str:
        """Classify security level based on score"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Acceptable"
        else:
            return "Needs Improvement"
    
    def _generate_security_recommendations(self, attacks: Dict[str, float], params: Dict[str, Any]) -> List[str]:
        """Generate security improvement recommendations"""
        recommendations = []
        
        poly_degree = params.get("polynomial_degree", 8192)
        
        # Check individual attack resistances
        if attacks.get("lattice_reduction_attack", 0) < 70:
            recommendations.append("Increase polynomial degree for better lattice attack resistance")
        
        if attacks.get("quantum_attack_resistance", 0) < 80:
            recommendations.append("Consider higher security parameters for quantum resistance")
        
        if poly_degree < 8192:
            recommendations.append("Use polynomial degree of at least 8192 for production systems")
        
        if attacks.get("side_channel_resistance", 0) < 80:
            recommendations.append("Implement side-channel attack countermeasures")
        
        if not recommendations:
            recommendations.append("Security parameters appear adequate for current threat model")
        
        return recommendations
    
    def _check_compliance(self, security_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with security standards"""
        overall_score = security_metrics["overall_security"]["overall_security_score"]
        poly_degree = security_metrics["configuration"].get("polynomial_degree", 8192)
        security_level = security_metrics["configuration"].get("security_level", 128)
        
        compliance = {
            "nist_post_quantum": overall_score >= 80 and poly_degree >= 8192,
            "commercial_grade": overall_score >= 70 and security_level >= 128,
            "research_acceptable": overall_score >= 60,
            "production_ready": overall_score >= 85 and poly_degree >= 8192
        }
        
        return compliance
    
    def _assess_security_risks(self, overall_score: float, security_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Assess security risks based on analysis"""
        risks = {}
        
        if overall_score < 60:
            risks["critical"] = "Overall security score too low for production use"
        
        noise_budget = security_metrics["noise_analysis"]["initial_noise_budget_bits"]
        if noise_budget < 20:
            risks["high"] = "Low noise budget may limit computation depth"
        
        poly_degree = security_metrics["configuration"].get("polynomial_degree", 8192)
        if poly_degree < 4096:
            risks["medium"] = "Polynomial degree below recommended minimum"
        
        attack_resistance = security_metrics["attack_resistance"]["overall_attack_resistance_score"]
        if attack_resistance < 70:
            risks["medium"] = "Attack resistance could be improved"
        
        if not risks:
            risks["low"] = "No significant security risks identified"
        
        return risks
    
    def save_security_metrics(self, metrics: Dict[str, Any], output_path: str) -> str:
        """
        Save security metrics to JSON file
        
        Args:
            metrics: Security metrics dictionary
            output_path: Path to save the results
            
        Returns:
            Path to the saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare evaluation.json structure
        evaluation_data = {
            "security_metrics": metrics,
            "metadata": {
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analyzer_version": "1.0.0",
                "config_source": self.config_path or "default"
            }
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Security metrics saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to save security metrics: {e}")
            raise
    
    def print_security_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Print a formatted summary of security metrics
        
        Args:
            metrics: Security metrics dictionary
        """
        print(f"\n{'='*70}")
        print("FHE SECURITY ANALYSIS SUMMARY")
        print(f"{'='*70}")
        
        # Overall security
        overall = metrics["overall_security"]
        print(f"\n🔒 Overall Security Assessment:")
        print(f"   Security Score:        {overall['overall_security_score']:.1f}/100")
        print(f"   Classification:        {overall['security_level_classification']}")
        
        # Key security
        key_sec = metrics["key_security"]
        print(f"\n🔑 Key Security:")
        print(f"   Public Key Size:       {key_sec['public_key_size']['megabytes']:.2f} MB")
        print(f"   Secret Key Size:       {key_sec['secret_key_size']['megabytes']:.6f} MB")
        print(f"   Key Security Strength: {key_sec['key_security_strength']:.0f} bits")
        
        # Polynomial security
        poly_sec = metrics["polynomial_security"]
        print(f"\n🔢 Polynomial Security:")
        print(f"   Polynomial Degree:     {poly_sec['polynomial_degree']}")
        print(f"   Ring-LWE Security:     {poly_sec['ring_lwe_security_level']:.0f} bits")
        print(f"   Security Margin:       {poly_sec['security_margin']:.1f}%")
        
        # Noise analysis
        noise = metrics["noise_analysis"]
        print(f"\n📊 Noise Analysis:")
        print(f"   Initial Noise Budget:  {noise['initial_noise_budget_bits']:.1f} bits")
        print(f"   Max Mult. Depth:       {noise['estimated_multiplicative_depth']}")
        print(f"   Noise Std. Dev:        {noise['noise_standard_deviation']}")
        
        # Attack resistance
        attacks = metrics["attack_resistance"]
        print(f"\n🛡️ Attack Resistance:")
        print(f"   Overall Score:         {attacks['overall_attack_resistance_score']:.1f}/100")
        print(f"   Lattice Attacks:       {attacks['individual_attacks']['lattice_reduction_attack']:.1f}/100")
        print(f"   Quantum Resistance:    {attacks['individual_attacks']['quantum_attack_resistance']:.1f}/100")
        
        # Compliance
        compliance = overall["compliance_status"]
        print(f"\n✅ Compliance Status:")
        for standard, status in compliance.items():
            status_icon = "✅" if status else "❌"
            print(f"   {standard.replace('_', ' ').title():20} {status_icon}")
        
        # Recommendations
        recommendations = attacks["recommended_improvements"]
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print(f"{'='*70}")


def analyze_fhe_security(
    config_path: str,
    output_path: str = "data/evaluation/security_analysis.json",
    print_summary: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to analyze FHE security
    
    Args:
        config_path: Path to FHE configuration YAML file
        output_path: Path to save results
        print_summary: Whether to print summary
        
    Returns:
        Security metrics dictionary
    """
    analyzer = FHESecurityAnalyzer(config_path)
    metrics = analyzer.analyze_security()
    
    if print_summary:
        analyzer.print_security_summary(metrics)
    
    analyzer.save_security_metrics(metrics, output_path)
    
    return metrics


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="FHE Security Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python security_metrics.py --config src/configs/fhe_config.yaml
  python security_metrics.py --config config.yaml --output results/security.json
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/fhe_config.yaml',
        help='Path to FHE configuration YAML file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/evaluation/evaluation.json',
        help='Output path for security metrics JSON'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip printing summary to console'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("Starting FHE security analysis...")
        
        # Analyze security
        metrics = analyze_fhe_security(
            config_path=args.config,
            output_path=args.output,
            print_summary=not args.no_summary
        )
        
        logger.info("✅ FHE security analysis completed successfully!")
        
        return metrics
        
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()