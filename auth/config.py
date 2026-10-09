"""Configuration for biometric authentication."""

from dataclasses import dataclass


@dataclass
class AuthConfig:
    """Thresholds and timing settings used by authentication components."""

    face_similarity_threshold: float = 0.6
    """Minimum face similarity score accepted as a match."""

    voice_similarity_threshold: float = 0.75
    """Minimum voice similarity score accepted as a match."""

    grace_period_seconds: int = 5
    """Time in seconds allowed for a transient authentication interruption."""

    face_poll_interval_seconds: float = 0.75
    """Interval in seconds between face authentication polling attempts."""
