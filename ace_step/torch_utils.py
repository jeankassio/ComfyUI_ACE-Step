"""
PyTorch utilities with Windows compatibility.

This module provides utilities to safely use PyTorch features
that may have compatibility issues on Windows or specific configurations.
"""

import platform
import torch
import warnings


def safe_torch_compile(model, enable_compile=True):
    """
    Safely apply torch.compile with fallback for incompatible systems.
    
    torch.compile() can fail or be unavailable on:
    - Windows with certain CUDA configurations
    - Systems without triton support
    - Older PyTorch versions
    - bfloat16 on Windows (common with triton issues)
    
    Args:
        model: PyTorch model to compile
        enable_compile (bool): Whether to attempt compilation (can be overridden by system)
    
    Returns:
        model: The compiled model if successful, otherwise the original model
    """
    
    if not enable_compile:
        return model
    
    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, 'compile'):
            warnings.warn("torch.compile not available in this PyTorch version", UserWarning)
            return model
        
        # Check for known problematic configurations on Windows
        if platform.system() == "Windows":
            # bfloat16 with triton on Windows is known to cause issues
            if hasattr(model, 'dtype') and model.dtype == torch.bfloat16:
                warnings.warn(
                    "torch.compile with bfloat16 on Windows can cause issues with triton. "
                    "Skipping compilation for stability.",
                    UserWarning
                )
                return model
        
        # Try to compile
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print(f"[SafeTorchCompile] Model compiled successfully")
            return compiled_model
        except RuntimeError as e:
            # torch.compile failed, likely due to triton or platform incompatibility
            error_str = str(e).lower()
            
            if "triton" in error_str or "cuda" in error_str or "windows" in error_str or "int too large" in error_str:
                warnings.warn(
                    f"torch.compile failed (likely triton/CUDA/platform issue on {platform.system()}): {e}\n"
                    f"Continuing without compilation.",
                    UserWarning
                )
                return model
            else:
                raise
                
    except Exception as e:
        warnings.warn(
            f"Unexpected error during torch.compile: {e}\n"
            f"Continuing without compilation.",
            UserWarning
        )
        return model


def safe_cuda_empty_cache():
    """
    Safely empty CUDA cache if available.
    
    This is safe to call even if CUDA is not available.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        warnings.warn(f"Failed to empty CUDA cache: {e}", UserWarning)


def safe_cuda_synchronize():
    """
    Safely synchronize CUDA if available.
    
    This is safe to call even if CUDA is not available.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        warnings.warn(f"Failed to synchronize CUDA: {e}", UserWarning)


def get_optimal_device():
    """
    Get the optimal compute device for the current system.
    
    Returns:
        torch.device: The best available device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_optimal_dtype():
    """
    Get the optimal data type for the current device.
    
    Returns:
        torch.dtype: The recommended dtype for the device
    """
    device = get_optimal_device()
    
    if device.type == "cuda":
        # Check if GPU supports bfloat16
        if torch.cuda.get_device_capability(device)[0] >= 8:  # Ampere or newer
            return torch.bfloat16
        else:
            return torch.float16
    elif device.type == "mps":
        return torch.float16
    else:
        return torch.float32


def setup_torch_backends():
    """
    Setup PyTorch backends with appropriate settings for the current system.
    
    This should be called early in the application initialization.
    """
    try:
        # Disable cudnn benchmark for deterministic results
        torch.backends.cudnn.benchmark = False
        
        # Enable cudnn deterministic mode if available
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
    except Exception as e:
        warnings.warn(f"Failed to configure torch backends: {e}", UserWarning)
    
    # Set high precision for float32 matmul
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        # This might not be available in older PyTorch versions
        pass
    
    # Try to enable TF32 for CUDA (can improve performance)
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
