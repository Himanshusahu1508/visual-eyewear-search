import cv2
import numpy as np
from PIL import Image


def extract_color_histogram(image_path, bins=(8, 8, 8)):
    """
    Extract normalized HSV color histogram from image path
    """
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None,
        bins,
        [0, 180, 0, 256, 0, 256]
    )

    cv2.normalize(hist, hist)
    return hist.flatten()


def compute_color_similarity(hist1, hist2):
    """
    Compare two color histograms using cosine similarity
    """
    hist1 = hist1 / np.linalg.norm(hist1)
    hist2 = hist2 / np.linalg.norm(hist2)

    return float(np.dot(hist1, hist2))
