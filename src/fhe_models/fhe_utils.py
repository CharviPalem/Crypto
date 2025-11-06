#!/usr/bin/env python3
"""
FHE Utilities Module

This module provides helper functions for Fully Homomorphic Encryption (FHE) operations
supporting both Pyfhel and Concrete-ML libraries. It includes functions for:
- FHE context initialization with BFV scheme
- Batch encryption and decryption operations
- Key management (save/load public and secret keys)
- Logging of ciphertext sizes and noise budget monitoring

Compatible with both fhe_svm_model.py and fhe_logistic_regression.py
"""

import logging
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [FHE_Utils] %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing FHE libraries
try:
    from Pyfhel import Pyfhel, PyCtxt
    PYFHEL_AVAILABLE = True
    logger.info("✅ Pyfhel library available")
except ImportError as e:
    PYFHEL_AVAILABLE = False
    logger.warning(f"⚠️ Pyfhel not available: {e}")

try:
    import concrete.ml
    CONCRETE_ML_AVAILABLE = True
    logger.info("✅ Concrete-ML library available")
except ImportError as e:
    CONCRETE_ML_AVAILABLE = False
    logger.warning(f"⚠️ Concrete-ML not available: {e}")


class FHEContextManager:
    """
    Manages FHE context for both Pyfhel and Concrete-ML libraries
    """
    
    def __init__(self, library: str = "auto"):
        """
        Initialize FHE context manager
        
        Args:
            library: FHE library to use ("pyfhel", "concrete", or "auto")
        """
        self.library = self._select_library(library)
        self.context = None
        self.public_key = None
        self.secret_key = None
        self.relin_key = None
        self.rotate_key = None
        self.context_params = {}
        
    def _select_library(self, library: str) -> str:
        """Select appropriate FHE library"""
        if library == "auto":
            if PYFHEL_AVAILABLE:
                return "pyfhel"
            elif CONCRETE_ML_AVAILABLE:
                return "concrete"
            else:
                raise RuntimeError("No FHE library available. Install Pyfhel or Concrete-ML")
        elif library == "pyfhel" and not PYFHEL_AVAILABLE:
            raise RuntimeError("Pyfhel not available")
        elif library == "concrete" and not CONCRETE_ML_AVAILABLE:
            raise RuntimeError("Concrete-ML not available")
        return library


def initialize_fhe_context(
    scheme: str = "bfv",
    poly_degree: int = 8192,
    plaintext_modulus_bits: int = 20,
    security_level: int = 128,
    library: str = "auto"
) -> FHEContextManager:
    """
    Initialize FHE context with BFV scheme and specified parameters
    
    Args:
        scheme: Encryption scheme ("bfv" or "ckks")
        poly_degree: Polynomial degree (power of 2, typically 4096, 8192, 16384)
        plaintext_modulus_bits: Bits for plaintext modulus
        security_level: Security level in bits (128, 192, or 256)
        library: FHE library to use ("pyfhel", "concrete", or "auto")
    
    Returns:
        FHEContextManager instance with initialized context
    """
    logger.info(f"Initializing FHE context with {scheme.upper()} scheme")
    logger.info(f"Parameters: poly_degree={poly_degree}, t_bits={plaintext_modulus_bits}, sec={security_level}")
    
    context_manager = FHEContextManager(library)
    
    if context_manager.library == "pyfhel":
        context_manager = _initialize_pyfhel_context(
            context_manager, scheme, poly_degree, plaintext_modulus_bits, security_level
        )
    elif context_manager.library == "concrete":
        context_manager = _initialize_concrete_context(
            context_manager, scheme, poly_degree, plaintext_modulus_bits, security_level
        )
    
    # Store context parameters
    context_manager.context_params = {
        "scheme": scheme,
        "poly_degree": poly_degree,
        "plaintext_modulus_bits": plaintext_modulus_bits,
        "security_level": security_level,
        "library": context_manager.library
    }
    
    logger.info(f"✅ FHE context initialized successfully using {context_manager.library}")
    return context_manager


def _initialize_pyfhel_context(
    context_manager: FHEContextManager,
    scheme: str,
    poly_degree: int,
    plaintext_modulus_bits: int,
    security_level: int
) -> FHEContextManager:
    """Initialize Pyfhel context"""
    try:
        he = Pyfhel()
        
        # Generate context
        he.contextGen(
            scheme=scheme,
            n=poly_degree,
            t_bits=plaintext_modulus_bits,
            sec=security_level
        )
        
        # Generate keys
        logger.info("Generating encryption keys...")
        he.keyGen()
        he.relinKeyGen()
        he.rotateKeyGen()
        
        context_manager.context = he
        logger.info("Pyfhel context and keys generated successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Pyfhel context: {e}")
        raise RuntimeError(f"Pyfhel initialization failed: {e}")
    
    return context_manager


def _initialize_concrete_context(
    context_manager: FHEContextManager,
    scheme: str,
    poly_degree: int,
    plaintext_modulus_bits: int,
    security_level: int
) -> FHEContextManager:
    """Initialize Concrete-ML context (placeholder for future implementation)"""
    logger.info("Concrete-ML context initialization (using default parameters)")
    # Concrete-ML handles context internally, so we just store parameters
    context_manager.context = "concrete_ml_context"
    return context_manager


def encrypt_batch(
    data: Union[List[int], List[float], np.ndarray],
    context_manager: FHEContextManager,
    scale_factor: float = 1000.0
) -> List[Any]:
    """
    Encrypt a batch of data
    
    Args:
        data: Data to encrypt (list or numpy array)
        context_manager: FHE context manager
        scale_factor: Scaling factor for float to int conversion
    
    Returns:
        List of encrypted ciphertexts
    """
    if context_manager.context is None:
        raise RuntimeError("FHE context not initialized")
    
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    logger.info(f"Encrypting batch of {len(data)} values...")
    start_time = time.time()
    
    encrypted_data = []
    
    if context_manager.library == "pyfhel":
        encrypted_data = _encrypt_batch_pyfhel(data, context_manager, scale_factor)
    elif context_manager.library == "concrete":
        encrypted_data = _encrypt_batch_concrete(data, context_manager, scale_factor)
    
    encryption_time = time.time() - start_time
    logger.info(f"Batch encryption completed in {encryption_time:.4f} seconds")
    
    # Log ciphertext information
    _log_ciphertext_info(encrypted_data, context_manager)
    
    return encrypted_data


def _encrypt_batch_pyfhel(
    data: np.ndarray,
    context_manager: FHEContextManager,
    scale_factor: float
) -> List[PyCtxt]:
    """Encrypt batch using Pyfhel"""
    he = context_manager.context
    encrypted_data = []
    
    for i, value in enumerate(data):
        if (i + 1) % 100 == 0 or i == 0:
            logger.debug(f"Encrypted {i + 1}/{len(data)} values")
        
        # Scale and convert to integer
        if isinstance(value, (int, np.integer)):
            int_value = int(value)
        else:
            int_value = int(value * scale_factor)
        
        # Encrypt
        encrypted_value = he.encryptInt(int_value)
        encrypted_data.append(encrypted_value)
    
    return encrypted_data


def _encrypt_batch_concrete(
    data: np.ndarray,
    context_manager: FHEContextManager,
    scale_factor: float
) -> List[Any]:
    """Encrypt batch using Concrete-ML (placeholder)"""
    logger.info("Concrete-ML batch encryption (placeholder implementation)")
    # For now, return the data as-is since Concrete-ML handles encryption internally
    return data.tolist()


def decrypt_batch(
    encrypted_data: List[Any],
    context_manager: FHEContextManager,
    scale_factor: float = 1000.0
) -> np.ndarray:
    """
    Decrypt a batch of encrypted data
    
    Args:
        encrypted_data: List of encrypted ciphertexts
        context_manager: FHE context manager
        scale_factor: Scaling factor for int to float conversion
    
    Returns:
        Decrypted data as numpy array
    """
    if context_manager.context is None:
        raise RuntimeError("FHE context not initialized")
    
    logger.info(f"Decrypting batch of {len(encrypted_data)} ciphertexts...")
    start_time = time.time()
    
    if context_manager.library == "pyfhel":
        decrypted_data = _decrypt_batch_pyfhel(encrypted_data, context_manager, scale_factor)
    elif context_manager.library == "concrete":
        decrypted_data = _decrypt_batch_concrete(encrypted_data, context_manager, scale_factor)
    
    decryption_time = time.time() - start_time
    logger.info(f"Batch decryption completed in {decryption_time:.4f} seconds")
    
    return np.array(decrypted_data)


def _decrypt_batch_pyfhel(
    encrypted_data: List[PyCtxt],
    context_manager: FHEContextManager,
    scale_factor: float
) -> List[float]:
    """Decrypt batch using Pyfhel"""
    he = context_manager.context
    decrypted_data = []
    
    for i, encrypted_value in enumerate(encrypted_data):
        if (i + 1) % 100 == 0 or i == 0:
            logger.debug(f"Decrypted {i + 1}/{len(encrypted_data)} values")
        
        # Decrypt and scale back
        decrypted_int = he.decryptInt(encrypted_value)
        decrypted_float = decrypted_int / scale_factor
        decrypted_data.append(decrypted_float)
    
    return decrypted_data


def _decrypt_batch_concrete(
    encrypted_data: List[Any],
    context_manager: FHEContextManager,
    scale_factor: float
) -> List[float]:
    """Decrypt batch using Concrete-ML (placeholder)"""
    logger.info("Concrete-ML batch decryption (placeholder implementation)")
    # For now, return the data as-is since Concrete-ML handles decryption internally
    return encrypted_data


def save_keys(
    context_manager: FHEContextManager,
    keys_dir: str,
    key_prefix: str = "fhe_keys"
) -> Dict[str, str]:
    """
    Save public and secret keys to files
    
    Args:
        context_manager: FHE context manager
        keys_dir: Directory to save keys
        key_prefix: Prefix for key filenames
    
    Returns:
        Dictionary with paths to saved key files
    """
    if context_manager.context is None:
        raise RuntimeError("FHE context not initialized")
    
    # Create keys directory
    keys_path = Path(keys_dir)
    keys_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving FHE keys to {keys_path}")
    
    saved_files = {}
    
    if context_manager.library == "pyfhel":
        saved_files = _save_keys_pyfhel(context_manager, keys_path, key_prefix)
    elif context_manager.library == "concrete":
        saved_files = _save_keys_concrete(context_manager, keys_path, key_prefix)
    
    # Save context parameters
    params_file = keys_path / f"{key_prefix}_params.json"
    import json
    with open(params_file, 'w') as f:
        json.dump(context_manager.context_params, f, indent=2)
    saved_files["params"] = str(params_file)
    
    logger.info(f"✅ Keys saved successfully: {list(saved_files.keys())}")
    return saved_files


def _save_keys_pyfhel(
    context_manager: FHEContextManager,
    keys_path: Path,
    key_prefix: str
) -> Dict[str, str]:
    """Save Pyfhel keys"""
    he = context_manager.context
    saved_files = {}
    
    try:
        # Save context
        context_file = keys_path / f"{key_prefix}_context.pkl"
        with open(context_file, 'wb') as f:
            pickle.dump(he.to_bytes_context(), f)
        saved_files["context"] = str(context_file)
        
        # Save public key
        public_key_file = keys_path / f"{key_prefix}_public.pkl"
        with open(public_key_file, 'wb') as f:
            pickle.dump(he.to_bytes_public_key(), f)
        saved_files["public_key"] = str(public_key_file)
        
        # Save secret key
        secret_key_file = keys_path / f"{key_prefix}_secret.pkl"
        with open(secret_key_file, 'wb') as f:
            pickle.dump(he.to_bytes_secret_key(), f)
        saved_files["secret_key"] = str(secret_key_file)
        
        # Save relinearization key
        relin_key_file = keys_path / f"{key_prefix}_relin.pkl"
        with open(relin_key_file, 'wb') as f:
            pickle.dump(he.to_bytes_relin_key(), f)
        saved_files["relin_key"] = str(relin_key_file)
        
        # Save rotation keys
        rotate_key_file = keys_path / f"{key_prefix}_rotate.pkl"
        with open(rotate_key_file, 'wb') as f:
            pickle.dump(he.to_bytes_rotate_key(), f)
        saved_files["rotate_key"] = str(rotate_key_file)
        
    except Exception as e:
        logger.error(f"Error saving Pyfhel keys: {e}")
        raise
    
    return saved_files


def _save_keys_concrete(
    context_manager: FHEContextManager,
    keys_path: Path,
    key_prefix: str
) -> Dict[str, str]:
    """Save Concrete-ML keys (placeholder)"""
    logger.info("Concrete-ML key saving (placeholder implementation)")
    return {"concrete_keys": "placeholder"}


def load_keys(
    keys_dir: str,
    key_prefix: str = "fhe_keys",
    library: str = "auto"
) -> FHEContextManager:
    """
    Load public and secret keys from files
    
    Args:
        keys_dir: Directory containing key files
        key_prefix: Prefix for key filenames
        library: FHE library to use
    
    Returns:
        FHEContextManager with loaded keys
    """
    keys_path = Path(keys_dir)
    if not keys_path.exists():
        raise FileNotFoundError(f"Keys directory not found: {keys_path}")
    
    logger.info(f"Loading FHE keys from {keys_path}")
    
    # Load context parameters
    params_file = keys_path / f"{key_prefix}_params.json"
    if params_file.exists():
        import json
        with open(params_file, 'r') as f:
            params = json.load(f)
        library = params.get("library", library)
    
    context_manager = FHEContextManager(library)
    
    if context_manager.library == "pyfhel":
        context_manager = _load_keys_pyfhel(context_manager, keys_path, key_prefix)
    elif context_manager.library == "concrete":
        context_manager = _load_keys_concrete(context_manager, keys_path, key_prefix)
    
    logger.info(f"✅ Keys loaded successfully using {context_manager.library}")
    return context_manager


def _load_keys_pyfhel(
    context_manager: FHEContextManager,
    keys_path: Path,
    key_prefix: str
) -> FHEContextManager:
    """Load Pyfhel keys"""
    try:
        he = Pyfhel()
        
        # Load context
        context_file = keys_path / f"{key_prefix}_context.pkl"
        with open(context_file, 'rb') as f:
            context_bytes = pickle.load(f)
        he.from_bytes_context(context_bytes)
        
        # Load public key
        public_key_file = keys_path / f"{key_prefix}_public.pkl"
        with open(public_key_file, 'rb') as f:
            public_key_bytes = pickle.load(f)
        he.from_bytes_public_key(public_key_bytes)
        
        # Load secret key
        secret_key_file = keys_path / f"{key_prefix}_secret.pkl"
        with open(secret_key_file, 'rb') as f:
            secret_key_bytes = pickle.load(f)
        he.from_bytes_secret_key(secret_key_bytes)
        
        # Load relinearization key
        relin_key_file = keys_path / f"{key_prefix}_relin.pkl"
        if relin_key_file.exists():
            with open(relin_key_file, 'rb') as f:
                relin_key_bytes = pickle.load(f)
            he.from_bytes_relin_key(relin_key_bytes)
        
        # Load rotation keys
        rotate_key_file = keys_path / f"{key_prefix}_rotate.pkl"
        if rotate_key_file.exists():
            with open(rotate_key_file, 'rb') as f:
                rotate_key_bytes = pickle.load(f)
            he.from_bytes_rotate_key(rotate_key_bytes)
        
        context_manager.context = he
        
    except Exception as e:
        logger.error(f"Error loading Pyfhel keys: {e}")
        raise
    
    return context_manager


def _load_keys_concrete(
    context_manager: FHEContextManager,
    keys_path: Path,
    key_prefix: str
) -> FHEContextManager:
    """Load Concrete-ML keys (placeholder)"""
    logger.info("Concrete-ML key loading (placeholder implementation)")
    context_manager.context = "concrete_ml_context"
    return context_manager


def _log_ciphertext_info(encrypted_data: List[Any], context_manager: FHEContextManager) -> None:
    """
    Log information about ciphertext sizes and noise budget
    
    Args:
        encrypted_data: List of encrypted ciphertexts
        context_manager: FHE context manager
    """
    if not encrypted_data:
        return
    
    logger.info(f"Ciphertext Information:")
    logger.info(f"  Number of ciphertexts: {len(encrypted_data)}")
    
    if context_manager.library == "pyfhel":
        _log_pyfhel_ciphertext_info(encrypted_data, context_manager)
    elif context_manager.library == "concrete":
        _log_concrete_ciphertext_info(encrypted_data, context_manager)


def _log_pyfhel_ciphertext_info(encrypted_data: List[PyCtxt], context_manager: FHEContextManager) -> None:
    """Log Pyfhel ciphertext information"""
    he = context_manager.context
    
    if encrypted_data:
        sample_ctxt = encrypted_data[0]
        
        # Get ciphertext size
        ctxt_size = len(sample_ctxt.to_bytes())
        total_size = ctxt_size * len(encrypted_data)
        
        logger.info(f"  Ciphertext size (sample): {ctxt_size:,} bytes")
        logger.info(f"  Total encrypted data size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
        
        # Get noise budget (if available)
        try:
            noise_budget = he.noise_level(sample_ctxt)
            logger.info(f"  Noise budget (sample): {noise_budget:.2f} bits")
            
            if noise_budget < 10:
                logger.warning("⚠️ Low noise budget detected! Consider using fresh ciphertexts.")
            elif noise_budget < 5:
                logger.error("❌ Critical noise budget! Decryption may fail.")
        except Exception as e:
            logger.debug(f"Could not get noise budget: {e}")


def _log_concrete_ciphertext_info(encrypted_data: List[Any], context_manager: FHEContextManager) -> None:
    """Log Concrete-ML ciphertext information (placeholder)"""
    logger.info("  Concrete-ML ciphertext info (placeholder)")


def get_context_info(context_manager: FHEContextManager) -> Dict[str, Any]:
    """
    Get information about the FHE context
    
    Args:
        context_manager: FHE context manager
    
    Returns:
        Dictionary with context information
    """
    if context_manager.context is None:
        return {"status": "not_initialized"}
    
    info = {
        "library": context_manager.library,
        "status": "initialized",
        "parameters": context_manager.context_params
    }
    
    if context_manager.library == "pyfhel":
        he = context_manager.context
        info.update({
            "scheme": he.scheme,
            "poly_degree": he.n,
            "plaintext_modulus": he.t,
            "security_level": he.sec,
            "key_switching_available": he.relinKeyGen_status,
            "rotation_available": he.rotateKeyGen_status
        })
    
    return info


# Convenience functions for backward compatibility
def create_fhe_context(**kwargs) -> FHEContextManager:
    """Alias for initialize_fhe_context"""
    return initialize_fhe_context(**kwargs)


def encrypt_data(data, context_manager, **kwargs) -> List[Any]:
    """Alias for encrypt_batch"""
    return encrypt_batch(data, context_manager, **kwargs)


def decrypt_data(encrypted_data, context_manager, **kwargs) -> np.ndarray:
    """Alias for decrypt_batch"""
    return decrypt_batch(encrypted_data, context_manager, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    """Example usage of FHE utilities"""
    
    print("FHE Utilities Demo")
    print("=" * 50)
    
    try:
        # Initialize FHE context
        print("\n1. Initializing FHE context...")
        context_manager = initialize_fhe_context(
            scheme="bfv",
            poly_degree=4096,
            plaintext_modulus_bits=20,
            security_level=128
        )
        
        # Print context info
        print("\n2. Context information:")
        info = get_context_info(context_manager)
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test encryption/decryption
        print("\n3. Testing encryption/decryption...")
        test_data = [1, 2, 3, 4, 5, 10, 100, 1000]
        print(f"   Original data: {test_data}")
        
        encrypted_data = encrypt_batch(test_data, context_manager)
        print(f"   Encrypted {len(encrypted_data)} values")
        
        decrypted_data = decrypt_batch(encrypted_data, context_manager)
        print(f"   Decrypted data: {decrypted_data.tolist()}")
        
        # Check accuracy
        max_error = np.max(np.abs(np.array(test_data) - decrypted_data))
        print(f"   Maximum error: {max_error}")
        
        # Test key saving/loading
        print("\n4. Testing key save/load...")
        keys_dir = "/tmp/fhe_test_keys"
        saved_files = save_keys(context_manager, keys_dir)
        print(f"   Saved keys: {list(saved_files.keys())}")
        
        # Load keys
        loaded_context = load_keys(keys_dir)
        print(f"   Loaded context using: {loaded_context.library}")
        
        print("\n✅ FHE utilities demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()