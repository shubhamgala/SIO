"""Face detection and embedding extraction using OpenCV's pretrained models."""

import argparse
import threading
import time
from pathlib import Path
from collections.abc import Callable

import cv2
import numpy as np

from .config import AuthConfig
from .storage import load_embedding, save_embedding


_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_YUNET_MODEL = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_SFACE_MODEL = _MODELS_DIR / "face_recognition_sface_2021dec.onnx"
_CAFFE_CONFIG = _MODELS_DIR / "deploy.prototxt"
_CAFFE_WEIGHTS = _MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"


def _detect_faces(image: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray | None]]:
    """Return face boxes and optional landmarks using YuNet or the Caffe SSD."""
    height, width = image.shape[:2]

    # Download face_detection_yunet_2023mar.onnx from
    # https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    # and place it in the project's models/ folder. If unavailable, use the
    # standard Caffe SSD detector files described below.
    if hasattr(cv2, "FaceDetectorYN") and _YUNET_MODEL.is_file():
        detector = cv2.FaceDetectorYN.create(
            str(_YUNET_MODEL), "", (width, height), 0.9, 0.3, 5000
        )
        _, faces = detector.detect(image)
        if faces is None:
            return [], []
        boxes = []
        landmarks = []
        for face in faces:
            x, y, box_width, box_height = (int(round(value)) for value in face[:4])
            boxes.append((x, y, box_width, box_height))
            landmarks.append(np.asarray(face[4:14], dtype=np.float32).reshape(5, 2))
        return boxes, landmarks

    # Download deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel
    # from the OpenCV opencv_3rdparty dnn_samples_face_detector_20170830
    # model files, then place both files in the project's models/ folder.
    if not (_CAFFE_CONFIG.is_file() and _CAFFE_WEIGHTS.is_file()):
        raise FileNotFoundError(
            "Face detector model files are missing. Put the YuNet ONNX model, "
            "or both Caffe SSD files, in the project's models/ folder."
        )
    detector = cv2.dnn.readNetFromCaffe(str(_CAFFE_CONFIG), str(_CAFFE_WEIGHTS))
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    detector.setInput(blob)
    detections = detector.forward()
    boxes = []
    for detection in detections[0, 0]:
        if float(detection[2]) < 0.5:
            continue
        x1, y1, x2, y2 = (detection[3:7] * np.array([width, height, width, height])).astype(int)
        boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    return boxes, [None] * len(boxes)


def _extract_face_embedding_from_frame(image: np.ndarray) -> np.ndarray | None:
    """Detect the largest face in a BGR frame and return its SFace embedding."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a non-empty BGR image with three channels")

    boxes, _ = _detect_faces(image)
    height, width = image.shape[:2]
    valid_boxes = []
    for x, y, box_width, box_height in boxes:
        left = max(0, x)
        top = max(0, y)
        right = min(width, x + box_width)
        bottom = min(height, y + box_height)
        if right > left and bottom > top:
            valid_boxes.append((left, top, right, bottom))
    if not valid_boxes:
        return None

    left, top, right, bottom = max(
        valid_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])
    )
    face_crop = cv2.resize(image[top:bottom, left:right], (160, 160))

    if not _SFACE_MODEL.is_file():
        raise FileNotFoundError(
            f"SFace model not found at {_SFACE_MODEL}. Download the model and "
            "place it in the project's models/ folder."
        )
    recognizer = cv2.FaceRecognizerSF.create(str(_SFACE_MODEL), "")
    embedding = recognizer.feature(face_crop).reshape(-1)
    return cv2.normalize(embedding, None, norm_type=cv2.NORM_L2).reshape(-1)


def extract_face_embedding(image_path: str) -> np.ndarray | None:
    """Detect the largest face and return its normalized SFace embedding.

    Download face_recognition_sface_2021dec.onnx from
    https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
    and place it in the project's models/ folder. Face detection uses YuNet
    when its model is present, or falls back to the standard Caffe SSD files.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return _extract_face_embedding_from_frame(image)


def enroll_face(user_id: str, image_paths: list[str]) -> bool:
    """Average all valid face embeddings and save the result for a user."""
    embeddings = [
        embedding
        for image_path in image_paths
        if (embedding := extract_face_embedding(image_path)) is not None
    ]
    if not embeddings:
        return False
    average_embedding = np.mean(np.stack(embeddings), axis=0)
    average_embedding = cv2.normalize(
        average_embedding.astype(np.float32), None, norm_type=cv2.NORM_L2
    ).reshape(-1)
    save_embedding(user_id, "face", average_embedding)
    return True


def verify_face(
    user_id: str,
    image_path_or_frame: str | np.ndarray,
    config: AuthConfig | None = None,
) -> tuple[bool, float]:
    """Compare an image or BGR frame with the user's enrolled face embedding."""
    if isinstance(image_path_or_frame, (str, Path)):
        embedding = extract_face_embedding(str(image_path_or_frame))
    elif isinstance(image_path_or_frame, np.ndarray):
        embedding = _extract_face_embedding_from_frame(image_path_or_frame)
    else:
        raise TypeError("image_path_or_frame must be a file path or a NumPy BGR frame")

    if embedding is None:
        return False, 0.0

    enrolled_embedding = np.asarray(load_embedding(user_id, "face"), dtype=np.float32).reshape(-1)
    candidate_embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if enrolled_embedding.shape != candidate_embedding.shape:
        raise ValueError("Enrolled and candidate face embeddings have different shapes")
    denominator = float(np.linalg.norm(enrolled_embedding) * np.linalg.norm(candidate_embedding))
    similarity = (
        float(np.dot(enrolled_embedding, candidate_embedding) / denominator)
        if denominator > 0
        else 0.0
    )
    active_config = config or AuthConfig()
    return similarity >= active_config.face_similarity_threshold, similarity


class FaceSessionMonitor:
    """Monitor a video source and report when an enrolled face is lost."""

    def __init__(
        self,
        user_id: str,
        source: int | cv2.VideoCapture = 0,
        config: AuthConfig | None = None,
        on_face_lost: Callable[[], None] = lambda: None,
    ) -> None:
        self.user_id = user_id
        self.capture = source if isinstance(source, cv2.VideoCapture) else cv2.VideoCapture(source)
        self.config = config or AuthConfig()
        self.on_face_lost = on_face_lost
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start monitoring in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor, name=f"FaceSessionMonitor-{self.user_id}", daemon=True
        )
        self._thread.start()

    def _monitor(self) -> None:
        last_verified_at = time.monotonic()
        loss_reported = False
        poll_interval = max(0.0, self.config.face_poll_interval_seconds)
        while not self._stop_event.is_set():
            success, frame = self.capture.read()
            if success and frame is not None:
                verified, _ = verify_face(self.user_id, frame, self.config)
                if verified:
                    last_verified_at = time.monotonic()
                    loss_reported = False

            if (
                not loss_reported
                and time.monotonic() - last_verified_at > self.config.grace_period_seconds
            ):
                loss_reported = True
                self.on_face_lost()

            if self._stop_event.wait(poll_interval):
                break

    def stop(self) -> None:
        """Signal the monitor to stop, join it, and release its video source."""
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        self.capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll or verify a face embedding.")
    commands = parser.add_subparsers(dest="command", required=True)
    enroll_parser = commands.add_parser("enroll", help="Enroll face images for a user")
    enroll_parser.add_argument("user_id")
    enroll_parser.add_argument("image_paths", nargs="+")
    verify_parser = commands.add_parser("verify", help="Verify a face image for a user")
    verify_parser.add_argument("user_id")
    verify_parser.add_argument("image_path")
    args = parser.parse_args()
    if args.command == "enroll":
        print(enroll_face(args.user_id, args.image_paths))
    else:
        matched, similarity = verify_face(args.user_id, args.image_path)
        print(f"matched={matched}, similarity={similarity:.4f}")
