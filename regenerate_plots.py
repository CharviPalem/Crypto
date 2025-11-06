#!/usr/bin/env python3
"""Quick script to regenerate privacy-utility plot without Random Forest"""

from src.evaluation.tradeoff_analysis import PrivacyUtilityAnalyzer

if __name__ == "__main__":
    print("Regenerating privacy-utility plot...")
    analyzer = PrivacyUtilityAnalyzer(results_dir="data/results")
    analyzer.analyze_tradeoff()
    print("Done!")
