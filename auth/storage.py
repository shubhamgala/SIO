"""File-based storage for biometric embeddings."""

from pathlib import Path

import numpy as np


_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"
_VALID_KINDS = {"face", "voice"}


def _embedding_path(user_id: str, kind: str) -> Path:
    """Build the path for an embedding, validating its kind and user ID."""
    if kind not in _VALID_KINDS:
        raise ValueError("kind must be 'face' or 'voice'")
    if not user_id or Path(user_id).name != user_id or user_id in {".", ".."}:
        raise ValueError("user_id must be a non-empty filename-safe identifier")
    return _EMBEDDINGS_DIR / f"{user_id}_{kind}.npy"


def save_embedding(user_id: str, kind: str, embedding: np.ndarray) -> None:
    """Save a user's face or voice embedding as a NumPy .npy file."""
    path = _embedding_path(user_id, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding)


def load_embedding(user_id: str, kind: str) -> np.ndarray:
    """Load and return a user's saved face or voice embedding."""
    return np.load(_embedding_path(user_id, kind), allow_pickle=False)
