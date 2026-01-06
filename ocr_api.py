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
    results = []
    concatenated_lines = []

    for idx, file in enumerate(files):
        try:
            contents = await file.read()

            # {
            #   "filename": str,
            #   "text": str,
            #   "lines": List[str]
            # }

            result = service.process_bytes(contents, file.filename)  # type: ignore

            file_result = {
                "file_index": idx,
                "filename": result["filename"],
                "text": result["text"],
                "lines": result["lines"],
            }

        except Exception:
            file_result = {
                "file_index": idx,
                "filename": file.filename,
                "text": "",
                "lines": [],
            }

        results.append(file_result)
        concatenated_lines.extend(file_result["lines"])

    return {
        "num_files": len(files),
        "results": results,
        "concatenated": {
            "text": "\n".join(concatenated_lines),
            "lines": concatenated_lines,
        },
    }


@app.post("/ocr/single")
async def ocr_single(file: UploadFile = File(...)):
    # single image
    contents = await file.read()
    result = service.process_bytes(contents, file.filename)  # type: ignore
    return result
