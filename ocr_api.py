# Run: uvicorn ocr_api:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, File, UploadFile, HTTPException
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
async def ocr(file: List[UploadFile] = File(...)):
    """
    Accepts one or multiple files under field name 'file' and returns:
    - structured OCR output per file
    - a concatenated text across all files
    """
    files = file  # support single form field named 'file' (may contain multiple files)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    concatenated_lines = []

    for idx, f in enumerate(files):
        try:
            contents = await f.read()
            if not contents:
                raise ValueError("Empty file")

            result = service.process_bytes(contents, f.filename)  # type: ignore

            file_result = {
                "file_index": idx,
                "filename": result["filename"],
                "text": result["text"],
                "lines": result["lines"],
            }
            concatenated_lines.extend(result["lines"])

        except Exception as e:
            file_result = {
                "file_index": idx,
                "filename": f.filename,
                "text": "",
                "lines": [],
                "error": str(e),
            }

        results.append(file_result)

    return {
        "num_files": len(files),
        "results": results,
        "concatenated": {
            "text": "\n".join(concatenated_lines),
            "lines": concatenated_lines,
        },
    }
