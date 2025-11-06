# Fully Homomorphic Encryption (FHE) NLP Pipeline

## Project Overview
This project implements a privacy-preserving NLP pipeline using Fully Homomorphic Encryption (FHE) and Differential Privacy (DP) techniques. The system allows for secure computation on encrypted medical text data while preserving patient privacy.

## Key Features
- **FHE Models**: Logistic Regression and SVM implementations using Concrete ML
- **Differential Privacy**: Configurable privacy parameters (ε, δ) via `dp_config.yaml`
- **Pipeline Orchestration**: End-to-end execution via `run_pipeline.py`
- **Evaluation**: Performance, privacy, and security metrics
- **Visualization**: Results plotting and analysis

## Project Structure
```
fhe_project/
├── src/                  # Core implementation
│   ├── configs/          # Configuration files
│   ├── fhe_models/       # FHE model implementations
│   ├── pipeline/         # Pipeline orchestration
│   ├── evaluation/       # Metric evaluation
│   └── visualization/    # Results plotting
├── data/                 # Data storage
│   ├── raw/              # Raw input data
│   ├── processed/        # Preprocessed data
│   └── results/          # Output results
├── notebooks/            # Demo notebooks
└── requirements.txt      # Python dependencies
```

## Getting Started
1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure parameters**:
Edit `src/configs/dp_config.yaml` for privacy settings

3. **Run pipeline**:
```bash
python src/pipeline/run_pipeline.py
```

## Configuration
Key configurable parameters in `dp_config.yaml`:
- Privacy budget (ε, δ)
- Noise mechanisms (Laplace/Gaussian)
- Sensitivity settings
- Evaluation metrics

## Demo Notebooks
- `data_generation_demo.ipynb`: Data preprocessing
- `fhe_training_demo.ipynb`: Model training
- `evaluation_demo.ipynb`: Results analysis

## License
[Specify license here]