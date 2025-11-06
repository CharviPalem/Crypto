#!/usr/bin/env python3
"""
Medical Data Preprocessing Script

This script performs data cleaning, anonymization, and differential privacy
on synthetic medical datasets for FHE NLP processing.
"""

import argparse
import json
import logging
import hashlib
import re
import string
from typing import Dict, List, Any
import numpy as np
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def hash_identifier(identifier: str, salt: str = "medical_fhe_2024") -> str:
    """
    Hash sensitive identifiers using SHA-256
    
    Args:
        identifier: The identifier to hash
        salt: Salt for hashing (default: "medical_fhe_2024")
    
    Returns:
        Hashed identifier as hexadecimal string
    """
    if not identifier or identifier == "":
        return "UNKNOWN"
    
    # Combine identifier with salt and hash
    combined = f"{identifier}{salt}"
    hash_object = hashlib.sha256(combined.encode('utf-8'))
    return hash_object.hexdigest()[:12]  # Return first 12 characters


def add_dp_noise(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
    """
    Add Laplacian noise for differential privacy
    
    Args:
        value: Original numeric value
        epsilon: Privacy parameter (smaller = more privacy)
        sensitivity: Sensitivity of the query (default: 1.0)
    
    Returns:
        Value with added Laplacian noise
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")
    
    # Scale parameter for Laplacian distribution
    scale = sensitivity / epsilon
    
    # Generate Laplacian noise
    noise = np.random.laplace(0, scale)
    
    return max(0, value + noise)  # Ensure non-negative values


def clean_text(text: str) -> str:
    """
    Clean and normalize text data
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned and normalized text
    """
    if not text or text == "":
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation but keep medical abbreviations
    # Keep periods in abbreviations like "dr.", "mg.", etc.
    text = re.sub(r'[^\w\s\.]', ' ', text)
    
    # Remove standalone periods
    text = re.sub(r'\s+\.\s+', ' ', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def generalize_age(age: int) -> str:
    """
    Generalize age into age groups for privacy
    
    Args:
        age: Original age
    
    Returns:
        Age group as string
    """
    if age < 18:
        return "under_18"
    elif age < 30:
        return "18_29"
    elif age < 50:
        return "30_49"
    elif age < 65:
        return "50_64"
    else:
        return "65_plus"


def anonymize_record(record: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
    """
    Anonymize a single medical record
    
    Args:
        record: Original medical record
        epsilon: Privacy parameter for differential privacy
    
    Returns:
        Anonymized medical record
    """
    anonymized = {}
    
    # Keep patient_id but hash it
    anonymized['patient_id'] = f"ANON_{hash_identifier(record.get('patient_id', ''))}"
    
    # Replace PII with hashed/generalized versions
    anonymized['name_hash'] = hash_identifier(record.get('name', ''))
    anonymized['phone_hash'] = hash_identifier(record.get('phone', ''))
    anonymized['ssn_hash'] = hash_identifier(record.get('ssn', ''))
    anonymized['email_domain'] = record.get('email', '').split('@')[-1] if '@' in record.get('email', '') else 'unknown.com'
    
    # Generalize age instead of adding noise (more practical for age groups)
    original_age = record.get('age', 0)
    anonymized['age_group'] = generalize_age(original_age)
    anonymized['age_dp'] = int(add_dp_noise(original_age, epsilon, sensitivity=5))  # Age sensitivity = 5 years
    
    # Keep gender as is (already categorical)
    anonymized['gender'] = record.get('gender', 'Unknown')
    
    # Keep disease information (needed for analysis)
    anonymized['disease'] = record.get('disease', 'Unknown')
    
    # Clean and normalize medical notes
    original_notes = record.get('medical_notes', '')
    anonymized['medical_notes_clean'] = clean_text(original_notes)
    
    # Add noise to any numeric fields that might be present
    # (In this case, we don't have weight, but we'll prepare for it)
    if 'weight' in record:
        anonymized['weight_dp'] = add_dp_noise(record['weight'], epsilon, sensitivity=2)
    
    return anonymized


def load_dataset(input_path: str) -> Dict[str, Any]:
    """
    Load dataset from JSON file
    
    Args:
        input_path: Path to input JSON file
    
    Returns:
        Loaded dataset dictionary
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded dataset from: {input_path}")
        
        if 'data' not in data:
            raise ValueError("Dataset must contain 'data' field")
        
        logger.info(f"Dataset contains {len(data['data'])} records")
        return data
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)


def save_dataset(dataset: Dict[str, Any], output_path: str) -> None:
    """
    Save processed dataset to JSON file
    
    Args:
        dataset: Processed dataset dictionary
        output_path: Path to output JSON file
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed dataset saved to: {output_file}")
        logger.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        sys.exit(1)


def print_sample_comparison(original_record: Dict, anonymized_record: Dict) -> None:
    """
    Print before/after comparison of a sample record
    
    Args:
        original_record: Original record
        anonymized_record: Anonymized record
    """
    print(f"\n{'='*60}")
    print("SAMPLE RECORD COMPARISON")
    print(f"{'='*60}")
    
    print("\n--- BEFORE (Original) ---")
    for key, value in original_record.items():
        if key == 'medical_notes':
            print(f"{key}: {str(value)[:100]}...")
        else:
            print(f"{key}: {value}")
    
    print("\n--- AFTER (Anonymized) ---")
    for key, value in anonymized_record.items():
        if key == 'medical_notes_clean':
            print(f"{key}: {str(value)[:100]}...")
        else:
            print(f"{key}: {value}")
    
    print(f"{'='*60}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Preprocess medical data with anonymization and differential privacy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_data.py --input data/raw/medical_data.json --output data/processed/clean_data.json --epsilon 1.0
  python preprocess_data.py -i raw_data.json -o processed_data.json -e 0.5 --verbose
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input JSON file path containing raw medical data'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSON file path for processed data'
    )
    
    parser.add_argument(
        '--epsilon', '-e',
        type=float,
        default=1.0,
        help='Epsilon parameter for differential privacy (default: 1.0, smaller = more privacy)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--show-sample',
        action='store_true',
        help='Show before/after comparison of first record'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.seed:
        np.random.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")
    
    # Validate arguments
    if args.epsilon <= 0:
        logger.error("Epsilon must be positive")
        sys.exit(1)
    
    logger.info("Starting medical data preprocessing")
    logger.info(f"Parameters: input={args.input}, output={args.output}, epsilon={args.epsilon}")
    
    try:
        # Load dataset
        dataset = load_dataset(args.input)
        original_records = dataset['data']
        
        if not original_records:
            logger.error("No records found in dataset")
            sys.exit(1)
        
        # Process records
        logger.info(f"Processing {len(original_records)} records...")
        anonymized_records = []
        
        for i, record in enumerate(original_records):
            if (i + 1) % 100 == 0 or i == 0:
                logger.info(f"Processed {i + 1}/{len(original_records)} records")
            
            anonymized_record = anonymize_record(record, args.epsilon)
            anonymized_records.append(anonymized_record)
        
        # Create processed dataset
        processed_dataset = {
            "metadata": {
                "total_records": len(anonymized_records),
                "description": "Anonymized and differentially private medical dataset",
                "epsilon": args.epsilon,
                "processing_steps": [
                    "PII hashing/generalization",
                    "Age group generalization",
                    "Text cleaning and normalization",
                    "Differential privacy noise injection"
                ],
                "fields": list(anonymized_records[0].keys()) if anonymized_records else []
            },
            "data": anonymized_records
        }
        
        # Save processed dataset
        save_dataset(processed_dataset, args.output)
        
        # Show sample comparison if requested
        if args.show_sample and original_records and anonymized_records:
            print_sample_comparison(original_records[0], anonymized_records[0])
        
        logger.info("Data preprocessing completed successfully!")
        
        # Print summary
        print(f"\n{'='*50}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Input file: {args.input}")
        print(f"Output file: {args.output}")
        print(f"Records processed: {len(anonymized_records)}")
        print(f"Epsilon (privacy): {args.epsilon}")
        print(f"Privacy techniques applied:")
        print(f"  - PII hashing (SHA-256)")
        print(f"  - Age generalization")
        print(f"  - Differential privacy (Laplacian noise)")
        print(f"  - Text normalization")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()