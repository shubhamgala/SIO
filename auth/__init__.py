"""Biometric authentication package scaffold."""

from .config import AuthConfig
from .storage import load_embedding, save_embedding

__all__ = ["AuthConfig", "load_embedding", "save_embedding"]
