import cv2
import numpy as np


def load_image(image_path, target_width=2000):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = img.shape
    if w != target_width:
        scale = target_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (target_width, new_h))

    return img


def deskew(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  #

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    coords = np.column_stack(np.where(dilated > 0))
    if len(coords) == 0:
        return img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if 0.5 < abs(angle) < 15:
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)  #

    return img


def get_lines(img):
    # split image into text lines
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )  # there was a error with installation here - --- IGNORE ---

    # horizontal projection
    row_sums = np.sum(binary, axis=1).astype(np.float32)
    row_sums = cv2.GaussianBlur(row_sums.reshape(-1, 1), (1, 15), 0).flatten()

    threshold = 0.05 * np.percentile(row_sums, 90)
    is_text = row_sums > threshold

    lines = []
    start = None
    for i, val in enumerate(is_text):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start > 15:
                lines.append(img[start:i, :])
            start = None

    if start is not None and len(is_text) - start > 15:
        lines.append(img[start : len(is_text), :])

    return lines


def split_line(line, max_width=1000):
    # split wide line in middle
    h, w = line.shape
    if w <= max_width:
        return [line]

    mid_start = int(w * 0.4)
    mid_end = int(w * 0.6)
    middle = line[:, mid_start:mid_end]

    col_sums = np.sum(middle, axis=0)
    split_pos = mid_start + np.argmax(col_sums)

    return [line[:, :split_pos], line[:, split_pos:]]


def prepare_line(line, height=64):
    h, w = line.shape
    if h < 5 or w < 5:
        return None

    scale = height / h
    new_w = int(w * scale)
    line = cv2.resize(line, (new_w, height))
    line = cv2.normalize(line, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
    line = cv2.copyMakeBorder(line, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    return line
