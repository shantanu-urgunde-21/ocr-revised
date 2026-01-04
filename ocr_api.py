# Run: uvicorn ocr_api:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, File, UploadFile
from typing import List
from ocr_service import OCRService

app = FastAPI()
service = None


@app.on_event("startup")
def startup():
    global service
    service = OCRService()
    print("OCR service ready")


@app.get("/")
def root():
    return {"message": "POST images to /ocr"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(files: List[UploadFile] = File(...)):
    # multiple images
    results = []
    all_lines = []

    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            result = service.process_bytes(  # type: ignore
                contents, file.filename
            )  # don't worry about this line (it's correct)

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
            results.append(
                {
                    "file_index": idx,
                    "filename": file.filename,
                    "text": "",
                    "lines": [],
                }
            )

    return {
        "num_files": len(files),
        "results": results,
        "concatenated": {"text": "\n".join(all_lines), "lines": all_lines},
    }


@app.post("/ocr/single")
async def ocr_single(file: UploadFile = File(...)):
    # single image
    contents = await file.read()
    result = service.process_bytes(contents, file.filename)  # type: ignore
    return result
