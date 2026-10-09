"""Manual webcam check: run with ``python -m auth.test_monitor_manual``."""

import time

from .face import FaceSessionMonitor


def main() -> None:
    monitor = FaceSessionMonitor(
        user_id="user1",
        source=0,
        on_face_lost=lambda: print("FACE LOST", flush=True),
    )
    monitor.start()
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
