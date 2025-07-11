from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import joblib
import io
import os
import spacy
import requests
import base64
import time
from transformers import pipeline

app = FastAPI()

# ✅ Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# ✅ Load Hugging Face Transformer classifier
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

# 🌐 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📩 Input Schema
class EmailScanRequest(BaseModel):
    subject: str
    body: str
    from_email: str
    urls: str = ""
    reply_to: str = ""
    return_path: str = ""

# 🔐 VirusTotal URL Cache
url_cache = {}

# ✅ VirusTotal Integration
def check_url_reputation(url):
    if url in url_cache:
        return url_cache[url]

    try:
        api_key = os.getenv("VT_API_KEY")  # ✅ Fixed line
        headers = {"x-apikey": api_key}
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")

        response = requests.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}",
            headers=headers
        )

        if response.status_code == 429:
            time.sleep(15)
            response = requests.get(
                f"https://www.virustotal.com/api/v3/urls/{url_id}",
                headers=headers
            )

        if response.status_code == 200:
            data = response.json()
            stats = data["data"]["attributes"]["last_analysis_stats"]
            verdict = "clean"
            if stats["malicious"] > 0:
                verdict = "malicious"
            elif stats["suspicious"] > 0:
                verdict = "suspicious"

            result = {
                "url": url,
                "verdict": verdict,
                "detections": stats
            }
            url_cache[url] = result
            return result
        else:
            return {"url": url, "verdict": "unknown", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"url": url, "verdict": "error", "error": str(e)}

# ✅ Named Entity Recognition
def extract_named_entities(text):
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "EMAIL"]]))

# 🔍 Scan Email API
@app.post("/scan-email/")
async def scan_email(data: EmailScanRequest):
    combined_text = f"{data.subject} {data.body} {data.from_email} {data.urls} {data.reply_to} {data.return_path}"

    # ⚡ BERT model prediction
    result = classifier(combined_text)[0]
    status = "phishing" if result['label'].lower() == "spam" else "safe"

    # ⚠️ Red flag keywords
    red_flags = [w for w in ["verify", "password", "update", "click", "login"] if w in data.body.lower()]
    entities = extract_named_entities(data.body)

    # 🌐 URL check
    suspicious_urls = []
    if data.urls:
        for url in data.urls.split():
            rep = check_url_reputation(url)
            if rep["verdict"] in ["suspicious", "malicious"]:
                suspicious_urls.append(rep)

    return {
        "status": status,
        "confidence": f"{result['score'] * 100:.2f}%",
        "red_flags": red_flags,
        "named_entities": entities,
        "suspicious_urls": suspicious_urls
    }

# 📄 Upload Text File
@app.post("/upload-text/")
async def upload_text(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        return {"text": text.strip()}
    except Exception:
        return {"error": "❌ Failed to read text file"}

# 🖼 Upload Image (OCR)
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        extracted_text = pytesseract.image_to_string(image)
        return {"text": extracted_text.strip()}
    except Exception:
        return {"error": "❌ OCR failed on image"}

# 📑 Upload PDF (OCR)
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = ""
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return {"text": text.strip()}
    except Exception:
        return {"error": "❌ Failed to extract text from PDF"}

# 🔄 Universal File Extractor
@app.post("/extract-file/")
async def extract_file(file: UploadFile = File(...)):
    try:
        content_type = file.content_type
        file_data = await file.read()

        if content_type == "application/pdf":
            with fitz.open(stream=file_data, filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
        elif content_type.startswith("image/"):
            image = Image.open(io.BytesIO(file_data))
            text = pytesseract.image_to_string(image)
        elif content_type == "text/plain":
            text = file_data.decode("utf-8")
        else:
            return {"error": "❌ Unsupported file type"}

        return {"status": "success", "extracted_text": text.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 📸 Final OCR Endpoint
@app.post("/extract-text/")
async def extract_text(image: UploadFile = File(...)):
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(img)
        return {"status": "success", "extracted_text": text.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ✅ Health Check Route (Important for Render)
@app.get("/")
async def root():
    return {"message": "🚀 PhishScan backend is live!"}

# ✅ Log App Startup
@app.on_event("startup")
async def startup_event():
    print("✅ App started successfully.")
