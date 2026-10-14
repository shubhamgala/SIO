"""Speaker embedding extraction and matching with SpeechBrain ECAPA-TDNN."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier  # Import SpeechBrain classifier
from speechbrain.utils.fetching import LocalStrategy  # Import LocalStrategy for safe Windows file copying

from .config import AuthConfig  # Import biometric configuration
from .storage import load_embedding, save_embedding  # Import embedding storage

_SAMPLE_RATE = 16_000  # Target sample rate in Hz
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "spkrec-ecapa-voxceleb"  # Model path

@lru_cache(maxsize=1)  # Cache model in memory
def _speaker_classifier() -> EncoderClassifier:  # Define model loader
    """Load and cache the pretrained speaker encoder on first use."""
    return EncoderClassifier.from_hparams(  # Load model
        source="speechbrain/spkrec-ecapa-voxceleb",  # Pretrained model name
        savedir=str(_MODEL_DIR),  # Local save directory
        local_strategy=LocalStrategy.COPY,  # Use file copy instead of Windows symlink
    )  # End of loader


import soundfile as sf  # Import soundfile for reliable cross-platform audio reading

def extract_voice_embedding(audio_path: str) -> np.ndarray:  # Define voice embedding extractor function
    """Load audio and return its ECAPA speaker embedding as a NumPy array."""
    data, sample_rate = sf.read(audio_path, dtype='float32')  # Read WAV audio samples using soundfile
    waveform = torch.from_numpy(data)  # Convert float32 numpy array to PyTorch tensor
    if waveform.ndim == 1:  # Check if audio is single-channel mono
        waveform = waveform.unsqueeze(0)  # Add channel dimension to make tensor 2D
    else:  # If multi-channel audio
        waveform = waveform.transpose(0, 1)  # Transpose shape to (channels, time)
    if waveform.shape[0] == 0 or waveform.shape[1] == 0:  # Validate that audio is non-empty
        raise ValueError(f"Audio file contains no usable samples: {audio_path}")  # Raise error
    duration_seconds = waveform.shape[1] / sample_rate  # Calculate duration in seconds
    if duration_seconds < 1.0:  # Ensure speech sample is at least 1 second long
        raise ValueError(f"Audio must be at least 1 second long; got {duration_seconds:.2f} seconds")  # Raise error
    waveform = waveform.mean(dim=0, keepdim=True)  # Average channels to mono
    if sample_rate != _SAMPLE_RATE:  # Check if sample rate differs from target 16kHz
        waveform = torchaudio.functional.resample(waveform, sample_rate, _SAMPLE_RATE)  # Resample to 16kHz
    with torch.inference_mode():  # Disable autograd for fast inference
        embedding = _speaker_classifier().encode_batch(waveform)  # Compute 192-d speaker vector
    return embedding.squeeze().detach().cpu().numpy().astype(np.float32)  # Return numpy float32 embedding


def enroll_voice(user_id: str, audio_paths: list[str]) -> bool:
    """Average voice embeddings from the provided samples and save them."""
    if not audio_paths:
        return False
    embeddings = [extract_voice_embedding(path) for path in audio_paths]
    average_embedding = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    save_embedding(user_id, "voice", average_embedding)
    return True


def verify_voice(user_id: str, audio_path: str) -> tuple[bool, float]:
    """Compare an audio sample with the user's enrolled voice embedding."""
    embedding = np.asarray(extract_voice_embedding(audio_path), dtype=np.float32).reshape(-1)
    enrolled_embedding = np.asarray(
        load_embedding(user_id, "voice"), dtype=np.float32
    ).reshape(-1)
    if enrolled_embedding.shape != embedding.shape:
        raise ValueError("Enrolled and candidate voice embeddings have different shapes")

    denominator = float(np.linalg.norm(enrolled_embedding) * np.linalg.norm(embedding))
    similarity = (
        float(np.dot(enrolled_embedding, embedding) / denominator)
        if denominator > 0
        else 0.0
    )
    return similarity >= AuthConfig().voice_similarity_threshold, similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll or verify a voice embedding.")
    commands = parser.add_subparsers(dest="command", required=True)
    enroll_parser = commands.add_parser("enroll", help="Enroll audio samples for a user")
    enroll_parser.add_argument("user_id")
    enroll_parser.add_argument("audio_paths", nargs="+")
    verify_parser = commands.add_parser("verify", help="Verify a voice sample for a user")
    verify_parser.add_argument("user_id")
    verify_parser.add_argument("audio_path")
    args = parser.parse_args()
    if args.command == "enroll":
        print(enroll_voice(args.user_id, args.audio_paths))
    else:
        matched, similarity = verify_voice(args.user_id, args.audio_path)
        print(f"matched={matched}, similarity={similarity:.4f}")
