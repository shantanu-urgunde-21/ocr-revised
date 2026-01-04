import cv2
import numpy as np
from pathlib import Path
import ocr_preprocessing as prep
from ocr_model import OCRModel


class OCRService:

    def __init__(self):
        self.model = OCRModel()

    def process_image(self, image_path):

        img = prep.load_image(image_path)
        img = prep.deskew(img)

        # Get text lines
        lines = prep.get_lines(img)
        text_lines = []

        for line in lines:
            # split if too wide
            chunks = prep.split_line(line)
            line_parts = []

            for chunk in chunks:
                prepared = prep.prepare_line(chunk)
                if prepared is None:
                    continue

                try:
                    text = self.model.predict(prepared)
                    if text:
                        line_parts.append(text)
                except:
                    pass

            if line_parts:
                text_lines.append(" ".join(line_parts))

        return {
            "filename": Path(image_path).name,
            "text": "\n".join(text_lines),
            "lines": text_lines,
        }

    def process_multiple(self, image_paths):
        # process multiple images
        results = []
        all_lines = []

        for idx, path in enumerate(image_paths):
            try:
                result = self.process_image(path)
                results.append(
                    {
                        "file_index": idx,
                        "filename": result["filename"],
                        "text": result["text"],
                        "lines": result["lines"],
                    }
                )
                all_lines.extend(result["lines"])
            except Exception as e:
                print(f"Error: {path} - {e}")
                results.append(
                    {
                        "file_index": idx,
                        "filename": Path(path).name,
                        "text": "",
                        "lines": [],
                    }
                )

        return {
            "num_files": len(image_paths),
            "results": results,
            "concatenated": {"text": "\n".join(all_lines), "lines": all_lines},
        }

    def process_bytes(self, image_bytes, filename):
        # Process image from bytes
        # don't worry about this function, it's correct (hopefully)
        arr = np.frombuffer(image_bytes, np.uint8)  # type: ignore
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)  # type: ignore
        if img is None:
            raise ValueError("Cannot decode image")

        # Resize
        h, w = img.shape
        if w != 2000:
            scale = 2000 / w
            img = cv2.resize(img, (2000, int(h * scale)))

        # Fix rotation
        img = prep.deskew(img)

        # Get text lines
        lines = prep.get_lines(img)
        text_lines = []

        for line in lines:
            chunks = prep.split_line(line)
            line_parts = []

            for chunk in chunks:
                prepared = prep.prepare_line(chunk)
                if prepared is None:
                    continue

                try:
                    text = self.model.predict(prepared)
                    if text:
                        line_parts.append(text)
                except:
                    pass

            if line_parts:
                text_lines.append(" ".join(line_parts))

        return {
            "filename": filename,
            "text": "\n".join(text_lines),
            "lines": text_lines,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_service.py image1.jpg image2.png ...")
        sys.exit(1)

    service = OCRService()
    result = service.process_multiple(sys.argv[1:])

    print(f"\nProcessed {result['num_files']} images")
    print("\nExtracted text:")
    print(result["concatenated"]["text"])
