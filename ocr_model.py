import numpy as np
import cv2
import pytesseract


class OCRModel:
    def __init__(self):
        print("loading the model")

    def predict(self, line_image):
        config = "--psm 7"  # single line model
        text = pytesseract.image_to_string(line_image, config=config)
        return text.strip()
