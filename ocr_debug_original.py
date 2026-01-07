import cv2
import numpy as np
from pathlib import Path
import ocr_preprocessing as prep
from ocr_model import OCRModel
import json
from datetime import datetime


class OCRDebugService:
    """Debug version of your ORIGINAL OCR service (horizontal projection)"""

    def __init__(self, debug_dir="debug_output_original"):
        self.model = OCRModel()
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        print(f"Debug output: {self.debug_dir}")

    def process_image_debug(self, image_path, skip_deskew=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = Path(image_path).stem
        session_dir = self.debug_dir / f"{img_name}_{timestamp}"
        session_dir.mkdir(exist_ok=True)

        debug_log = {
            "input_file": str(image_path),
            "timestamp": timestamp,
            "skip_deskew": skip_deskew,
            "pipeline_steps": [],
        }

        print(f"\n{'='*60}")
        print(f"DEBUGGING ORIGINAL SYSTEM: {image_path}")
        print(f"Output: {session_dir}")
        print(f"{'='*60}\n")

        # Step 1: Load & Resize
        print("Step 1: Loading and resizing...")
        img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h_orig, w_orig = img_original.shape
        cv2.imwrite(str(session_dir / "01_original.png"), img_original)

        img = prep.load_image(image_path, target_width=2000)
        h, w = img.shape
        cv2.imwrite(str(session_dir / "02_resized.png"), img)
        print(f"  ✓ {w_orig}x{h_orig} → {w}x{h}")

        debug_log["pipeline_steps"].append(
            {
                "step": 1,
                "name": "load_resize",
                "original": {"w": w_orig, "h": h_orig},
                "resized": {"w": w, "h": h},
            }
        )

        # Step 2: Deskew (optional)
        print(f"\nStep 2: Deskewing (skip={skip_deskew})...")
        img_before_deskew = img.copy()

        if skip_deskew:
            img_deskewed = img.copy()
            print("  ⊘ SKIPPED")
            cv2.imwrite(str(session_dir / "03_deskewed_SKIPPED.png"), img_deskewed)
            diff_score = 0
        else:
            img_deskewed = prep.deskew(img)
            diff = cv2.absdiff(img_before_deskew, img_deskewed)
            diff_score = np.mean(diff)

            cv2.imwrite(str(session_dir / "03_deskewed.png"), img_deskewed)
            cv2.imwrite(
                str(session_dir / "03_deskew_diff.png"), diff * 10
            )  # amplify for visibility

            print(f"  ✓ Done (diff: {diff_score:.2f})")
            if diff_score > 10:
                print(f"  ⚠️  WARNING: Large change! May have degraded quality")

        debug_log["pipeline_steps"].append(
            {
                "step": 2,
                "name": "deskew",
                "skipped": skip_deskew,
                "diff_score": float(diff_score),
            }
        )

        img = img_deskewed

        # Step 3: Line Detection (Horizontal Projection)
        print("\nStep 3: Detecting lines (horizontal projection)...")

        # Show the thresholding
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(str(session_dir / "04_binary.png"), binary)

        # Get projection
        row_sums = np.sum(binary, axis=1).astype(np.float32)
        threshold = 0.1 * np.percentile(row_sums, 90)

        # Visualize projection
        proj_img = np.zeros((img.shape[0], 500, 3), dtype=np.uint8)
        max_sum = np.max(row_sums)
        for i, val in enumerate(row_sums):
            bar_len = int((val / max_sum) * 450)
            color = (0, 255, 0) if val > threshold else (100, 100, 100)
            cv2.line(proj_img, (0, i), (bar_len, i), color, 1)

        cv2.line(
            proj_img,
            (int((threshold / max_sum) * 450), 0),
            (int((threshold / max_sum) * 450), img.shape[0]),
            (0, 0, 255),
            2,
        )
        cv2.imwrite(str(session_dir / "05_projection.png"), proj_img)

        lines = prep.get_lines(img)
        print(f"  ✓ Detected {len(lines)} lines")
        print(f"    Threshold: {threshold:.2f}, Max projection: {max_sum:.2f}")

        debug_log["pipeline_steps"].append(
            {
                "step": 3,
                "name": "line_detection_projection",
                "num_lines": len(lines),
                "threshold": float(threshold),
                "max_projection": float(max_sum),
            }
        )

        # Save detected lines with visualization
        lines_dir = session_dir / "06_detected_lines"
        lines_dir.mkdir(exist_ok=True)

        # Create visualization on original image
        img_vis = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

        line_details = []
        y_pos = 0
        for idx, line in enumerate(lines):
            h_line, w_line = line.shape

            # Draw rectangle on visualization
            cv2.rectangle(img_vis, (0, y_pos), (w_line, y_pos + h_line), (0, 255, 0), 2)
            cv2.putText(
                img_vis,
                f"L{idx}",
                (10, y_pos + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Save individual line
            line_path = lines_dir / f"line_{idx:03d}_{w_line}x{h_line}.png"
            cv2.imwrite(str(line_path), line)

            detail = {
                "idx": idx,
                "width": w_line,
                "height": h_line,
                "y_position": y_pos,
            }
            line_details.append(detail)
            print(f"    Line {idx}: {w_line}x{h_line} at y={y_pos}")

            y_pos += h_line

        cv2.imwrite(str(session_dir / "06_lines_visualization.png"), img_vis)
        debug_log["pipeline_steps"].append(
            {"step": 3.5, "name": "line_details", "lines": line_details}
        )

        # Step 4: Line Splitting
        print("\nStep 4: Splitting wide lines...")
        all_chunks = []
        chunks_dir = session_dir / "07_split_chunks"
        chunks_dir.mkdir(exist_ok=True)

        chunk_idx = 0
        for line_idx, line in enumerate(lines):
            chunks = prep.split_line(line, max_width=1000)

            if len(chunks) > 1:
                print(f"    Line {line_idx} → {len(chunks)} chunks")

            for chunk_num, chunk in enumerate(chunks):
                h_c, w_c = chunk.shape
                chunk_path = (
                    chunks_dir / f"chunk_{chunk_idx:03d}_L{line_idx}_C{chunk_num}.png"
                )
                cv2.imwrite(str(chunk_path), chunk)

                all_chunks.append(
                    {
                        "image": chunk,
                        "line_idx": line_idx,
                        "chunk_num": chunk_num,
                        "path": chunk_path,
                        "size": {"w": w_c, "h": h_c},
                    }
                )
                chunk_idx += 1

        print(f"  ✓ Total chunks: {len(all_chunks)}")

        # Step 5: Prepare for OCR
        print("\nStep 5: Preparing for Tesseract...")
        prepared_dir = session_dir / "08_prepared"
        prepared_dir.mkdir(exist_ok=True)

        prepared_chunks = []
        for chunk_data in all_chunks:
            prepared = prep.prepare_line(chunk_data["image"], height=64)
            if prepared is not None:
                prep_path = prepared_dir / f"prep_{len(prepared_chunks):03d}.png"
                cv2.imwrite(str(prep_path), prepared)

                prepared_chunks.append(
                    {**chunk_data, "prepared": prepared, "prep_path": prep_path}
                )

        print(f"  ✓ Prepared {len(prepared_chunks)} chunks")

        # Step 6: OCR
        print("\nStep 6: Running Tesseract OCR...")
        text_lines = []
        ocr_results = []

        current_line_parts = []
        current_line_idx = prepared_chunks[0]["line_idx"] if prepared_chunks else -1

        for prep_data in prepared_chunks:
            line_idx = prep_data["line_idx"]

            if line_idx != current_line_idx:
                if current_line_parts:
                    text_lines.append(" ".join(current_line_parts))
                current_line_parts = []
                current_line_idx = line_idx

            try:
                text = self.model.predict(prep_data["prepared"])

                if text:
                    current_line_parts.append(text)

                ocr_result = {
                    "chunk": str(prep_data["path"].name),
                    "line_idx": line_idx,
                    "text": text,
                    "length": len(text),
                }
                ocr_results.append(ocr_result)

                status = "✓" if text else "✗"
                preview = text[:60] + "..." if len(text) > 60 else text
                print(f"    {status} L{line_idx}: '{preview}'")

            except Exception as e:
                print(f"    ✗ L{line_idx}: ERROR - {e}")
                ocr_results.append(
                    {
                        "chunk": str(prep_data["path"].name),
                        "line_idx": line_idx,
                        "error": str(e),
                    }
                )

        if current_line_parts:
            text_lines.append(" ".join(current_line_parts))

        # Save results
        final_text = "\n".join(text_lines)

        results_file = session_dir / "09_final_text.txt"
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(final_text)

        ocr_file = session_dir / "09_ocr_results.json"
        with open(ocr_file, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, indent=2)

        debug_log["final_results"] = {
            "num_lines": len(text_lines),
            "total_chars": len(final_text),
            "lines": text_lines,
        }

        log_file = session_dir / "debug_log.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(debug_log, f, indent=2)

        print(f"\n{'='*60}")
        print(f"COMPLETE!")
        print(f"{'='*60}")
        print(f"Output: {session_dir}")
        print(f"  - {len(lines)} detected lines")
        print(f"  - {len(all_chunks)} chunks")
        print(f"  - {len(prepared_chunks)} prepared")
        print(f"  - {len(text_lines)} final text lines")
        print(f"  - {len(final_text)} total characters")
        print(f"{'='*60}\n")

        return {
            "filename": Path(image_path).name,
            "text": final_text,
            "lines": text_lines,
            "debug_info": {
                "session_dir": str(session_dir),
                "num_lines_detected": len(lines),
                "num_chunks": len(all_chunks),
                "skip_deskew": skip_deskew,
                "deskew_diff": float(diff_score),
            },
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_debug_original.py image.jpg [--skip-deskew]")
        print("\nDebugs your ORIGINAL system (horizontal projection line detection)")
        print("\nOptions:")
        print("  --skip-deskew    Skip deskewing step")
        sys.exit(1)

    image_path = sys.argv[1]
    skip_deskew = "--skip-deskew" in sys.argv

    service = OCRDebugService()
    result = service.process_image_debug(image_path, skip_deskew=skip_deskew)

    print("\n=== EXTRACTED TEXT ===")
    print(result["text"])
    print("\n=== SUMMARY ===")
    print(f"Lines detected: {result['debug_info']['num_lines_detected']}")
    print(f"Lines output: {len(result['lines'])}")
    print(f"Deskew diff: {result['debug_info']['deskew_diff']:.2f}")
    print(f"Output dir: {result['debug_info']['session_dir']}")
