from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def process_image(img: Image.Image) -> Image.Image:
    # 🔧 Your processing logic here
    # Example: convert to grayscale
    return img.convert("L")


def image_decomposition(img_path) -> Image.Image:

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu threshold, lines become 0 (black), empty areas 255 (white)
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=4
    )

    h, w = binary.shape
    closed_mask = np.zeros_like(binary, dtype=np.uint8)

    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue

        # touches image border? -> not closed
        if (ys.min() == 0 or ys.max() == h-1 or
            xs.min() == 0 or xs.max() == w-1):
            continue  # open to the outside

        # this is a closed area
        closed_mask[labels == label] = 255

    color_img = np.zeros((h, w, 3), np.uint8)
    random.seed(42)  # for reproducibility
    for label in range(1, num_labels):
        if np.any(closed_mask[labels == label]):
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            color_img[labels == label] = color
    
    pil_img = Image.fromarray(color_img)

    return pil_img