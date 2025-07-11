from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional

router = APIRouter(prefix="/scan", tags=["Scan"])

@router.post("/email")
async def scan_email(
    subject: str = Form(...),
    body: str = Form(...),
    urls: Optional[str] = Form(None),
    from_address: Optional[str] = Form(None),
    reply_to: Optional[str] = Form(None),
    return_path: Optional[str] = Form(None)
):
    # Dummy logic for now (you can replace with ML model or phishing rules)
    red_flags = []
    suspicious_keywords = ["verify", "password", "update", "login", "urgent"]

    for keyword in suspicious_keywords:
        if keyword in body.lower() or keyword in subject.lower():
            red_flags.append(keyword)

    # Dummy confidence score
    confidence = 96 if not red_flags else 42

    return {
        "status": "success",
        "result": "Safe" if not red_flags else "Suspicious",
        "confidence": confidence,
        "red_flags": red_flags,
        "suspicious_urls": urls.split(",") if urls else []
    }

@router.post("/ocr")
async def scan_ocr_text(
    extracted_text: str = Form(...)
):
    # Basic scanning of text from OCR
    red_flags = []
    keywords = ["verify", "click", "urgent", "reset", "compromised"]
    for word in keywords:
        if word in extracted_text.lower():
            red_flags.append(word)

    return {
        "result": "Suspicious" if red_flags else "Clean",
        "red_flags": red_flags,
        "text_snippet": extracted_text[:100]
    }

@router.post("/upload")
async def scan_file(
    files: List[UploadFile] = File(...)
):
    file_names = [file.filename for file in files]
    return {
        "message": "Files received",
        "files": file_names
    }
