pip install -r requirements.txt

uvicorn ocr_api:app --host 0.0.0.0 --port 8000

Response Format
json
{
"num_files": 2,
"results": [
{
"file_index": 0,
"filename": "image1.jpg",
"text": "Line 1\nLine 2",
"lines": ["Line 1", "Line 2"]
}
],
"concatenated": {
"text": "All text combined",
"lines": ["Line 1", "Line 2", "Line 3"]
}
}

Command Line
python ocr_service.py image1.jpg image2.png
