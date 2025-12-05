"""
Audio utilities with Windows compatibility fallback.

This module provides audio loading/saving utilities with fallback mechanisms
for systems where torchcodec is not compatible (e.g., Windows with CUDA).
"""

import os
import platform
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import warnings


def load_audio_safe(audio_path: str, sr: int = None, mono: bool = False):
    """
    Load audio file with Windows/CUDA compatibility fallback.
    
    This function attempts to load audio using torchaudio.load() first.
    If that fails (common issue with torchcodec on Windows), it falls back
    to librosa.load() which is more compatible.
    
    Args:
        audio_path (str): Path to the audio file
        sr (int, optional): Target sample rate. If None, keeps original sr
        mono (bool): Whether to convert to mono (False = keep stereo/original channels)
    
    Returns:
        tuple: (audio_tensor, sample_rate)
            audio_tensor: torch.Tensor of shape (channels, samples)
            sample_rate: int, the sample rate of the audio
    
    Raises:
        RuntimeError: If both torchaudio.load and librosa.load fail
    """
    
    # Attempt 1: Try torchaudio.load (preferred method)
    try:
        audio, sample_rate = torchaudio.load(audio_path)
        
        # Handle mono/stereo conversion if needed
        if mono and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        elif not mono and audio.shape[0] == 1:
            # Optional: convert mono to stereo by duplicating
            pass  # Keep as is for now
            
        # Resample if needed
        if sr is not None and sample_rate != sr:
            resampler = torchaudio.transforms.Resample(sample_rate, sr)
            audio = resampler(audio)
            sample_rate = sr
            
        return audio, sample_rate
        
    except Exception as e:
        print(f"[AudioUtils] torchaudio.load failed: {e}")
        print(f"[AudioUtils] Falling back to librosa for Windows compatibility...")
        
        # Attempt 2: Fallback to librosa
        try:
            # librosa loads mono by default, set mono=False to preserve channels
            audio_np, sample_rate = librosa.load(
                audio_path, 
                sr=sr,
                mono=mono
            )
            
            # Convert numpy array to torch tensor
            # librosa returns (samples,) for mono or (samples,) for mono=False single channel
            if len(audio_np.shape) == 1:
                # Mono audio
                audio = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, samples)
            else:
                # Multi-channel audio
                audio = torch.from_numpy(audio_np).float()  # (channels, samples)
            
            return audio, sample_rate
            
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load audio from {audio_path} using both torchaudio and librosa. "
                f"torchaudio error: {e}. librosa error: {e2}"
            )


def load_audio_safe_stereo(audio_path: str, sr: int = None):
    """
    Load audio file ensuring stereo output (convert mono to stereo if needed).
    
    Args:
        audio_path (str): Path to the audio file
        sr (int, optional): Target sample rate
    
    Returns:
        tuple: (audio_tensor, sample_rate)
            audio_tensor: torch.Tensor of shape (2, samples) - always stereo
            sample_rate: int, the sample rate of the audio
    """
    audio, sample_rate = load_audio_safe(audio_path, sr=sr, mono=False)
    
    # Ensure stereo (convert mono to stereo if needed)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        # If more than 2 channels, take first 2
        audio = audio[:2, :]
    
    return audio, sample_rate


def load_audio_safe_mono(audio_path: str, sr: int = None):
    """
    Load audio file as mono.
    
    Args:
        audio_path (str): Path to the audio file
        sr (int, optional): Target sample rate
    
    Returns:
        tuple: (audio_tensor, sample_rate)
            audio_tensor: torch.Tensor of shape (1, samples) - mono
            sample_rate: int, the sample rate of the audio
    """
    return load_audio_safe(audio_path, sr=sr, mono=True)


def get_audio_info(audio_path: str):
    """
    Get audio file information without loading the full audio.
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        dict: Dictionary with 'sample_rate' and 'num_frames' keys
    """
    try:
        # Try using torchaudio.info first
        try:
            info = torchaudio.info(audio_path)
            return {
                'sample_rate': info.sample_rate,
                'num_frames': info.num_frames,
                'num_channels': info.num_channels
            }
        except Exception as e:
            print(f"[AudioUtils] torchaudio.info failed: {e}")
            
            # Fallback to librosa
            y, sr = librosa.load(audio_path, sr=None, mono=False)
            return {
                'sample_rate': sr,
                'num_frames': len(y) if len(y.shape) == 1 else y.shape[1],
                'num_channels': 1 if len(y.shape) == 1 else y.shape[0]
            }
    except Exception as e:
        raise RuntimeError(f"Failed to get audio info for {audio_path}: {e}")


def save_audio_safe(audio_path: str, audio_tensor: torch.Tensor, sample_rate: int, format: str = None):
    """
    Save audio file with Windows/CUDA compatibility fallback.
    
    This function attempts to save audio using torchaudio.save() first.
    If that fails (common issue with torchcodec on Windows), it falls back
    to soundfile.write() or librosa methods which are more compatible.
    
    Args:
        audio_path (str): Path where to save the audio file
        audio_tensor (torch.Tensor): Audio tensor of shape (channels, samples) or (samples,)
        sample_rate (int): Sample rate of the audio
        format (str, optional): Format to save in (e.g., 'wav', 'mp3', 'flac'). 
                               If None, inferred from file extension.
    
    Returns:
        bool: True if save was successful
    
    Raises:
        RuntimeError: If all save methods fail
    """
    
    # Ensure tensor is on CPU and float32
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()
    
    if audio_tensor.dtype != torch.float32:
        audio_tensor = audio_tensor.float()
    
    # Attempt 1: Try torchaudio.save (preferred method)
    try:
        torchaudio.save(audio_path, audio_tensor, sample_rate)
        print(f"[AudioUtils] Successfully saved audio to {audio_path} using torchaudio")
        return True
    except Exception as e:
        print(f"[AudioUtils] torchaudio.save failed: {e}")
        print(f"[AudioUtils] Trying fallback methods for Windows compatibility...")
        
        # Attempt 2: Try soundfile (most compatible)
        try:
            # Convert tensor to numpy
            audio_np = audio_tensor.detach().cpu().numpy()
            
            # Soundfile expects (samples, channels) not (channels, samples)
            if len(audio_np.shape) == 2:
                audio_np = audio_np.T  # Transpose to (samples, channels)
            
            sf.write(audio_path, audio_np, sample_rate)
            print(f"[AudioUtils] Successfully saved audio to {audio_path} using soundfile")
            return True
        except Exception as e2:
            print(f"[AudioUtils] soundfile.write failed: {e2}")
            
            # Attempt 3: Try librosa.output.write_wav (fallback)
            try:
                audio_np = audio_tensor.detach().cpu().numpy()
                
                # Librosa expects (samples,) for mono or specific format
                if len(audio_np.shape) == 2:
                    if audio_np.shape[0] == 1:
                        audio_np = audio_np[0]  # Mono case
                    else:
                        # Average channels for mono if needed
                        audio_np = np.mean(audio_np, axis=0)
                
                librosa.output.write_wav(audio_path, audio_np, sr=sample_rate)
                print(f"[AudioUtils] Successfully saved audio to {audio_path} using librosa")
                return True
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to save audio to {audio_path} using all methods. "
                    f"torchaudio error: {e}. soundfile error: {e2}. librosa error: {e3}"
                )


def save_audio_safe_batch(audio_paths: list, audio_tensors: list, sample_rates: list):
    """
    Save multiple audio files with fallback.
    
    Args:
        audio_paths (list): List of output file paths
        audio_tensors (list): List of audio tensors
        sample_rates (list or int): Sample rate(s) - can be single int or list
    
    Returns:
        list: List of booleans indicating success for each file
    """
    if not isinstance(sample_rates, list):
        sample_rates = [sample_rates] * len(audio_paths)
    
    results = []
    for path, tensor, sr in zip(audio_paths, audio_tensors, sample_rates):
        try:
            result = save_audio_safe(path, tensor, sr)
            results.append(result)
        except Exception as e:
            warnings.warn(f"Failed to save {path}: {e}", UserWarning)
            results.append(False)
    
    return results
