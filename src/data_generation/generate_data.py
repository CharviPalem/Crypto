#!/usr/bin/env python3
"""
Synthetic Medical Text Dataset Generator

This script generates synthetic medical records using the Faker library.
Each record includes patient information and medical notes for FHE NLP processing.
"""

import argparse
import json
import logging
import random
from typing import Dict, List
from faker import Faker
from faker.providers import BaseProvider
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalProvider(BaseProvider):
    """Custom Faker provider for medical data"""
    
    diseases = [
        "Hypertension", "Type 2 Diabetes", "Coronary Artery Disease", 
        "Chronic Obstructive Pulmonary Disease", "Asthma", "Arthritis",
        "Depression", "Anxiety Disorder", "Migraine", "Osteoporosis",
        "Gastroesophageal Reflux Disease", "Chronic Kidney Disease",
        "Atrial Fibrillation", "Hypothyroidism", "Hyperlipidemia"
    ]
    
    treatments = [
        "prescribed medication", "recommended physical therapy", 
        "scheduled follow-up appointment", "ordered blood tests",
        "referred to specialist", "recommended lifestyle changes",
        "prescribed dietary modifications", "ordered imaging studies",
        "initiated monitoring protocol", "adjusted medication dosage"
    ]
    
    symptoms = [
        "chest pain", "shortness of breath", "fatigue", "dizziness",
        "headache", "joint pain", "muscle weakness", "nausea",
        "difficulty sleeping", "mood changes", "memory issues",
        "digestive problems", "skin rash", "vision changes"
    ]
    
    def disease(self) -> str:
        return self.random_element(self.diseases)
    
    def treatment(self) -> str:
        return self.random_element(self.treatments)
    
    def symptom(self) -> str:
        return self.random_element(self.symptoms)


def generate_medical_notes(fake: Faker) -> str:
    """Generate realistic medical notes"""
    templates = [
        "Patient presents with {symptom1} and {symptom2}. Diagnosed with {disease}. {treatment} and scheduled for follow-up in 2 weeks.",
        "Follow-up visit for {disease}. Patient reports improvement in {symptom1}. {treatment} to continue current regimen.",
        "New patient consultation. Chief complaint: {symptom1}. Physical examination reveals {symptom2}. Preliminary diagnosis: {disease}. {treatment}.",
        "Routine check-up. Patient with history of {disease} doing well. Mild {symptom1} reported. {treatment} and continue monitoring.",
        "Emergency visit due to acute {symptom1}. Workup suggests {disease}. {treatment} and admitted for observation.",
        "Specialist referral for {disease}. Patient experiencing {symptom1} and {symptom2}. {treatment} pending further evaluation."
    ]
    
    template = random.choice(templates)
    return template.format(
        symptom1=fake.symptom(),
        symptom2=fake.symptom(),
        disease=fake.disease(),
        treatment=fake.treatment().capitalize()
    )


def generate_patient_record(fake: Faker, patient_id: int) -> Dict:
    """Generate a single patient record"""
    return {
        "patient_id": f"PAT-{patient_id:06d}",
        "name": fake.name(),
        "age": random.randint(18, 95),
        "gender": random.choice(["Male", "Female", "Other"]),
        "disease": fake.disease(),
        "phone": fake.phone_number(),
        "ssn": fake.ssn(),
        "email": fake.email(),
        "medical_notes": generate_medical_notes(fake)
    }


def generate_dataset(num_samples: int) -> List[Dict]:
    """Generate the complete synthetic dataset"""
    fake = Faker()
    fake.add_provider(MedicalProvider)
    
    logger.info(f"Generating {num_samples} synthetic medical records...")
    
    dataset = []
    for i in range(num_samples):
        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"Generated {i + 1}/{num_samples} records")
        
        record = generate_patient_record(fake, i + 1)
        dataset.append(record)
    
    logger.info(f"Successfully generated {len(dataset)} records")
    return dataset


def save_dataset(dataset: List[Dict], output_path: str) -> None:
    """Save dataset to JSON file"""
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_records": len(dataset),
                    "description": "Synthetic medical text dataset for FHE NLP processing",
                    "fields": ["patient_id", "name", "age", "gender", "disease", "phone", "ssn", "email", "medical_notes"]
                },
                "data": dataset
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to: {output_file}")
        logger.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        sys.exit(1)


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic medical text dataset for FHE NLP processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_data.py --samples 1000 --output data/raw/medical_data.json
  python generate_data.py -s 500 -o /tmp/test_data.json
        """
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=1000,
        help='Number of synthetic records to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/raw/synthetic_medical_data.json',
        help='Output file path for the generated dataset (default: data/raw/synthetic_medical_data.json)'
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
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.seed:
        random.seed(args.seed)
        Faker.seed(args.seed)
        logger.info(f"Using random seed: {args.seed}")
    
    # Validate arguments
    if args.samples <= 0:
        logger.error("Number of samples must be positive")
        sys.exit(1)
    
    logger.info("Starting synthetic medical dataset generation")
    logger.info(f"Parameters: samples={args.samples}, output={args.output}")
    
    try:
        # Generate dataset
        dataset = generate_dataset(args.samples)
        
        # Save to file
        save_dataset(dataset, args.output)
        
        logger.info("Dataset generation completed successfully!")
        
        # Print summary
        print(f"\n{'='*50}")
        print("DATASET GENERATION SUMMARY")
        print(f"{'='*50}")
        print(f"Records generated: {len(dataset)}")
        print(f"Output file: {args.output}")
        print(f"Sample record fields: {list(dataset[0].keys()) if dataset else 'None'}")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()