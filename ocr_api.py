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
async def ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        result = service.process_bytes(contents, file.filename)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/multiple")
async def ocr_multiple(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    concatenated_lines = []

    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            if not contents:
                raise ValueError("Empty file")

            result = service.process_bytes(contents, file.filename)

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
                "filename": file.filename,
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
