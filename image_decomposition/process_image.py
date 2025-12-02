from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

def process_image(img: Image.Image) -> Image.Image:
    # 🔧 Your processing logic here
    # Example: convert to grayscale
    return img.convert("L")

def contour_to_svg_path(contour):
    """
    contour: numpy array of shape (N, 1, 2) or (N, 2)
    returns: SVG path string like "M x0 y0 L x1 y1 ... Z"
    """
    pts = contour.reshape(-1, 2)

    # Start at first point
    x0, y0 = pts[0]
    commands = [f"M {float(x0):.2f} {float(y0):.2f}"]

    # Line to each subsequent point
    for x, y in pts[1:]:
        commands.append(f"L {float(x):.2f} {float(y):.2f}")

    # Close the path
    commands.append("Z")
    return " ".join(commands)

def image_decomposition(img) -> Image.Image:

    # img = cv2.imread(img_path)
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

def image_to_svgs(img_path, output_dir):

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
    min_cc_area = 10
    area_count = 1
    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue

        # touches image border? -> not closed
        if (ys.min() == 0 or ys.max() == h-1 or
            xs.min() == 0 or xs.max() == w-1):
            continue  # open to the outside
        
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_cc_area:
            continue

        # this is a closed area
        closed_mask[labels == label] = 255
    
    os.makedirs(output_dir, exist_ok=True)
    
    for label in range(1, num_labels):
        # Check if this label is part of a closed area at all
        if not np.any(closed_mask[labels == label]):
            continue

        # Make a mask for just this label
        mask = np.zeros_like(binary, dtype=np.uint8)
        mask[labels == label] = 255

        # Find contours of this region
        # (Only external contour, we don't expect holes inside these areas)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        # Choose the biggest contour (just in case there are tiny fragments)
        cnt = max(contours, key=cv2.contourArea)

        # Optional: simplify the contour to fewer points
        epsilon = 0.5  # tweak for more/less detail
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        path_d = contour_to_svg_path(approx)

        # Build SVG text in the same style as cow_separate_9.svg
        svg_content = f'''<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="{path_d}" fill="white"/>
</svg>
    '''

        out_path = os.path.join(output_dir, f"area_{area_count}.svg")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        area_count += 1