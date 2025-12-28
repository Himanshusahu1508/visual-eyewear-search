import os
import uuid
from PIL import Image
import cv2


TMP_DIR = "data/tmp_crops"
os.makedirs(TMP_DIR, exist_ok=True)


def smart_crop_eyewear(image_path: str) -> str:
    """
    Attempts to detect face and crop eyewear region.
    If MediaPipe is unavailable or detection fails,
    returns original image path safely.
    """
    try:
        # Lazy import (CRITICAL)
        import mediapipe as mp

        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

        image = cv2.imread(image_path)
        if image is None:
            return image_path

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = detector.process(image_rgb)
        if not results.detections:
            return image_path

        detection = results.detections[0]
        box = detection.location_data.relative_bounding_box

        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        # Upper-face crop (glasses region)
        y2 = y1 + int(0.6 * bh)
        x2 = x1 + bw

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image_path

        crop_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.jpg")
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(crop_path)

        return crop_path

    except Exception:
        # ANY failure → fallback safely
        return image_path
