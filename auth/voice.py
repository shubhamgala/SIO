"""Speaker embedding extraction and matching with SpeechBrain ECAPA-TDNN."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

from .config import AuthConfig
from .storage import load_embedding, save_embedding


_SAMPLE_RATE = 16_000
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "spkrec-ecapa-voxceleb"


@lru_cache(maxsize=1)
def _speaker_classifier() -> EncoderClassifier:
    """Load and cache the pretrained speaker encoder on first use."""
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(_MODEL_DIR),
    )


def extract_voice_embedding(audio_path: str) -> np.ndarray:
    """Load audio and return its ECAPA speaker embedding as a NumPy array."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim != 2 or waveform.shape[0] == 0 or waveform.shape[1] == 0:
        raise ValueError(f"Audio file contains no usable samples: {audio_path}")

    duration_seconds = waveform.shape[1] / sample_rate
    if duration_seconds < 1.0:
        raise ValueError(
            f"Audio must be at least 1 second long; got {duration_seconds:.2f} seconds"
        )

    # The speaker model expects mono audio. Average channels before resampling.
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != _SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, _SAMPLE_RATE)

    with torch.inference_mode():
        embedding = _speaker_classifier().encode_batch(waveform)
    return embedding.squeeze().detach().cpu().numpy().astype(np.float32)


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
