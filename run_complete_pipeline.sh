#!/bin/bash
# Complete FHE Project Pipeline Execution Script
# This script runs the entire project from data generation to visualization

set -e  # Exit on error

echo "============================================================"
echo "FHE Project - Complete Pipeline Execution"
echo "============================================================"
echo ""

# Set project root
PROJECT_ROOT="/home/charvi/Documents/Cryptography proj/FHE/fhe_project"
cd "$PROJECT_ROOT"

# Step 1: Generate synthetic data (if needed)
echo "Step 1: Checking/Generating synthetic medical data..."
if [ ! -f "data/raw/synthetic_medical_data.json" ]; then
    echo "  → Generating synthetic data..."
    python src/data_generation/generate_data.py --output data/raw/synthetic_medical_data.json --samples 1000
else
    echo "  ✓ Synthetic data already exists"
fi
echo ""

# Step 2: Preprocess data
echo "Step 2: Preprocessing data..."
python src/preprocessing/preprocess_data.py \
    --input data/raw/synthetic_medical_data.json \
    --output data/processed/preprocessed_data.json
echo ""

# Step 3: Train FHE Logistic Regression (uses PREPROCESSED data)
echo "Step 3: Training FHE Logistic Regression model..."
python src/fhe_models/fhe_logistic_regression.py \
    --train data/processed/preprocessed_data.json \
    --output data/results/fhe_logistic_regression_results.json
echo ""

# Step 4: Train FHE SVM (uses PREPROCESSED data with binary classification)
echo "Step 4: Training FHE SVM model..."
python src/fhe_models/fhe_svm_model.py \
    --train data/processed/preprocessed_data.json \
    --output data/results/fhe_svm_results.json \
    --skip-encryption
echo ""

# Step 5: Evaluate Privacy Metrics
echo "Step 5: Evaluating privacy metrics..."
python src/evaluation/privacy_metrics.py \
    --input data/raw/synthetic_medical_data.json \
    --output data/results/privacy_evaluation.json
echo ""

# Step 6: Evaluate Security Metrics
echo "Step 6: Evaluating security metrics..."
python src/evaluation/security_metrics.py \
    --output data/results/security_evaluation.json
echo ""

# Step 7: Generate Privacy-Utility Tradeoff Analysis
echo "Step 7: Generating privacy-utility tradeoff analysis..."
python src/evaluation/tradeoff_analysis.py \
    --output data/results
echo ""

# Step 8: Generate all visualizations
echo "Step 8: Generating visualizations..."
python src/visualization/plot_results.py \
    --output data/results
echo ""

echo "============================================================"
echo "Pipeline Execution Complete!"
echo "============================================================"
echo ""
echo "Results saved in: data/results/"
echo ""
echo "Generated files:"
echo "  - fhe_logistic_regression_results.json"
echo "  - fhe_svm_results.json"
echo "  - privacy_evaluation.json"
echo "  - security_evaluation.json"
echo "  - privacy_utility_plot.png"
echo "  - model_comparison.png"
echo "  - performance_metrics.png"
echo "  - privacy_scores.png"
echo "  - security_analysis.png"
echo ""
