from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
import re

from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import spacy
from PyPDF2 import PdfReader
from docx import Document

# Additional imports for OCR
from pdf2image import convert_from_path
import pytesseract

app = FastAPI(title="Decentralized AI Legal Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize models
summarizer_en = pipeline("summarization", model="facebook/bart-large-cnn")
try:
    summarizer_odia = pipeline("summarization", model="ai4bharat/indic-bert-odia")
except Exception:
    summarizer_odia = summarizer_en

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

semantic_corpus = [
    "The plaintiff filed a petition for enforcement of contract.",
    "Arbitration award dated March 15, 2023, is sought to be enforced.",
    "Breach of agreement led to monetary damages.",
    "Defendant failed to comply with settlement terms.",
    "The court held in favor of respondent citing previous case law."
]
corpus_embeddings = semantic_model.encode(semantic_corpus, convert_to_numpy=True)
faiss_index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
faiss_index.add(corpus_embeddings)


def extract_text_pdf(file_path: str) -> str:
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception:
        # Fallback to OCR for scanned PDFs
        text = extract_text_ocr_pdf(file_path)
    if not text.strip():
        # Still empty, try OCR
        text = extract_text_ocr_pdf(file_path)
    return text


def extract_text_ocr_pdf(file_path: str) -> str:
    # Convert PDF pages to images and extract text via OCR
    text = ""
    try:
        pages = convert_from_path(file_path)
        for page in pages:
            text += pytesseract.image_to_string(page, lang='eng') + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR extraction failed: {str(e)}")
    return text


def extract_text_docx(file_path: str) -> str:
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Docx read failed: {str(e)}")
    return text


def extract_text_txt(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text read failed: {str(e)}")
    return text


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_pdf(file_path)
    elif ext == ".docx":
        return extract_text_docx(file_path)
    elif ext == ".txt":
        return extract_text_txt(file_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type {ext}")


def summarize_text(text: str, language: str = "en", max_len: int = 130, min_len: int = 30) -> str:
    if not text.strip():
        return "No text found to summarize."
    if language.lower() in ['or', 'odia']:
        try:
            summ = summarizer_odia(text, max_length=max_len, min_length=min_len, do_sample=False)
        except Exception:
            summ = summarizer_en(text, max_length=max_len, min_length=min_len, do_sample=False)
    else:
        summ = summarizer_en(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summ[0]['summary_text']


def semantic_search(query: str, top_k: int = 3) -> List[str]:
    q_vec = semantic_model.encode(query, convert_to_numpy=True)
    distances, indices = faiss_index.search(np.array([q_vec]), top_k)
    return [semantic_corpus[idx] for idx in indices[0]]


def extract_entities_and_citations(text: str):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    citations = re.findall(r'\b([A-Z][a-z]+ v\. [A-Z][a-z]+,? \d{4})\b', text)
    return {"entities": entities, "citations": citations}


def format_petition(petitioner: str, respondent: str, grounds: str, prayer: str) -> str:
    template = (f"IN THE COURT OF XYZ\n\n"
                f"Petitioner: {petitioner}\n"
                f"Respondent: {respondent}\n\n"
                f"GROUNDS:\n{grounds}\n\n"
                f"PRAYER:\n{prayer}\n\n"
                f"Place: _______      Date: _______\n"
                f"Signature: ________________")
    return template


# Models for requests
class PetitionRequest(BaseModel):
    petitioner: str
    respondent: str
    grounds: str
    prayer: str


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload_summarize/")
async def upload_and_summarize(file: UploadFile = File(...), language: str = Form("en")):
    # Save uploaded file temporarily
    filename = file.filename
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(filepath)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the document.")
    summary = summarize_text(text, language=language)
    os.remove(filepath)
    return {"summary": summary}


@app.post("/semantic_search/")
def semantic_search_api(request: SearchRequest):
    results = semantic_search(request.query, request.top_k)
    return {"results": results}


@app.post("/extract_entities/")
async def extract_entities_api(file: UploadFile = File(...)):
    filename = file.filename
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(filepath)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text found in document.")
    data = extract_entities_and_citations(text)
    os.remove(filepath)
    return data


@app.post("/format_petition/")
def format_petition_api(petition: PetitionRequest):
    formatted = format_petition(petition.petitioner, petition.respondent, petition.grounds, petition.prayer)
    return {"petition": formatted}


@app.get("/")
def read_root():
    return {"message": "Decentralized AI Legal Assistant API is running."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("document_ai:app", host="0.0.0.0", port=8000, reload=True)
