#!/usr/bin/env python3
"""
Decentralized AI Legal Assistant - Akash Network Optimized
HackOdisha 5.0 - Akash Network Deployment Track

Key Features:
- GPU-optimized AI models for Akash Network deployment
- Decentralized processing with privacy preservation
- Multi-language support (English + Odia) for Indian legal system
- Scalable architecture for distributed cloud infrastructure
- Zero local GPU requirements for end users
"""

import os
import sys
import logging
import time
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import shutil
import re

# FastAPI and web components
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# AI/ML Libraries - GPU optimized for Akash
import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import spacy

# Document processing
from PyPDF2 import PdfReader
from docx import Document
from pdf2image import convert_from_path
import pytesseract

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/legal-ai.log')
    ]
)
logger = logging.getLogger(__name__)

# Akash Network optimized configuration
class AkashConfig:
    # GPU detection and optimization
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_MEMORY_FRACTION = 0.8
    MAX_BATCH_SIZE = 4 if DEVICE == "cuda" else 1
    
    # Model configuration for decentralized deployment
    MODEL_CACHE_DIR = "/app/models"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB for Akash deployment
    SUPPORTED_FORMATS = [".pdf", ".docx", ".txt"]
    
    # Network configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0")
    
    # Privacy and security for decentralized deployment
    ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"
    DATA_RETENTION_HOURS = 0  # No data retention for privacy
    
config = AkashConfig()

# Initialize FastAPI with Akash Network optimizations
app = FastAPI(
    title="Decentralized AI Legal Assistant - Akash Network",
    description="GPU-powered legal document processing on decentralized infrastructure",
    version="2.0.0-akash",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enhanced CORS for decentralized access
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Global model storage - GPU optimized
class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_loading_status = {}
        self.device = config.DEVICE
        
    async def load_models(self):
        """Load all AI models with GPU optimization"""
        logger.info(f"Loading models on device: {self.device}")
        
        try:
            # Load English summarization model (GPU optimized)
            logger.info("Loading BART English summarizer...")
            self.models['summarizer_en'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1,
                model_kwargs={"cache_dir": config.MODEL_CACHE_DIR}
            )
            
            # Load Odia/Multilingual model
            logger.info("Loading Odia/Multilingual summarizer...")
            try:
                self.models['summarizer_odia'] = pipeline(
                    "summarization",
                    model="ai4bharat/indic-bert",
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={"cache_dir": config.MODEL_CACHE_DIR}
                )
            except Exception as e:
                logger.warning(f"Odia model fallback to English: {e}")
                self.models['summarizer_odia'] = self.models['summarizer_en']
            
            # Load semantic search model (GPU optimized)
            logger.info("Loading Sentence Transformer...")
            self.models['semantic_model'] = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device,
                cache_folder=config.MODEL_CACHE_DIR
            )
            
            # Load spaCy model
            logger.info("Loading spaCy NER model...")
            self.models['nlp'] = spacy.load("en_core_web_sm")
            
            # Initialize FAISS index with legal corpus
            logger.info("Initializing FAISS legal search index...")
            await self._initialize_legal_corpus()
            
            logger.info("✅ All models loaded successfully on Akash Network!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI model initialization failed: {str(e)}"
            )
    
    async def _initialize_legal_corpus(self):
        """Initialize legal knowledge corpus for semantic search"""
        # Extended legal corpus for Indian legal system
        self.legal_corpus = [
            "The plaintiff filed a petition for enforcement of contract terms under Indian Contract Act 1872.",
            "Arbitration award dated March 15, 2023, is sought to be enforced under Section 36 of Arbitration Act.",
            "Breach of agreement led to monetary damages and loss of business opportunities in commercial contract.",
            "Defendant failed to comply with settlement terms as agreed in mediation proceedings.",
            "The court held in favor of respondent citing previous case law precedents from Supreme Court.",
            "Property dispute involves boundary issues and ownership rights documentation under Transfer of Property Act.",
            "Employment contract termination requires proper notice period and compensation under Labour Laws.",
            "Intellectual property infringement case involves trademark and copyright violations under IP Act 1999.",
            "Family court matters include custody, maintenance, and property settlement under Hindu Marriage Act.",
            "Commercial contract disputes require evidence of performance and breach under Sale of Goods Act.",
            "Writ petition under Article 32 seeking enforcement of fundamental rights guaranteed by Constitution.",
            "Criminal complaint filed under Section 420 IPC for cheating and dishonestly inducing delivery.",
            "Civil suit for recovery of money with interest and costs under Code of Civil Procedure.",
            "Appeal against lower court judgment on questions of law and fact interpretation.",
            "Bail application under Section 439 CrPC seeking release from judicial custody."
        ]
        
        # Create embeddings with GPU acceleration
        corpus_embeddings = self.models['semantic_model'].encode(
            self.legal_corpus, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Initialize FAISS index
        self.models['faiss_index'] = faiss.IndexFlatL2(corpus_embeddings.shape[1])
        self.models['faiss_index'].add(corpus_embeddings)
        
        logger.info(f"Legal corpus initialized with {len(self.legal_corpus)} documents")

# Initialize model manager
model_manager = ModelManager()

# Pydantic models for API
class DocumentSummaryRequest(BaseModel):
    language: str = Field(default="en", description="Language for summarization (en/or/hi)")
    max_length: int = Field(default=150, ge=50, le=500, description="Maximum summary length")
    min_length: int = Field(default=30, ge=10, le=100, description="Minimum summary length")

class DocumentSummaryResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    processing_time: float
    language: str
    model_used: str
    gpu_accelerated: bool

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Legal search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    include_scores: bool = Field(default=True, description="Include relevance scores")

class SemanticSearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    processing_time: float
    total_results: int
    gpu_accelerated: bool

class EntityExtractionResponse(BaseModel):
    entities: List[Dict[str, str]]
    legal_citations: List[str]
    dates: List[str]
    amounts: List[str]
    processing_time: float
    total_entities: int
    confidence_scores: Dict[str, float]

class PetitionRequest(BaseModel):
    petitioner: str = Field(..., min_length=2, max_length=200)
    respondent: str = Field(..., min_length=2, max_length=200)
    court_type: str = Field(default="District Court", description="Type of court")
    case_type: str = Field(default="Civil", description="Civil/Criminal/Constitutional")
    grounds: str = Field(..., min_length=50, max_length=5000)
    prayer: str = Field(..., min_length=20, max_length=2000)
    urgency: bool = Field(default=False, description="Urgent hearing required")

class PetitionResponse(BaseModel):
    petition: str
    word_count: int
    estimated_pages: int
    court_formatted: bool
    timestamp: str
    petition_number: str

# Enhanced document processing functions
class DocumentProcessor:
    def __init__(self):
        self.upload_dir = "/tmp/legal_uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def extract_text_with_ocr(self, file_path: str) -> str:
        """Extract text with advanced OCR for Odia/English documents"""
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        
        try:
            if ext == ".pdf":
                text = await self._extract_pdf_text(file_path)
            elif ext == ".docx":
                text = await self._extract_docx_text(file_path)
            elif ext == ".txt":
                text = await self._extract_txt_text(file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format: {ext}"
                )
            
            return text
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(e)}"
            )
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract PDF text with OCR fallback for Odia support"""
        text = ""
        
        # Try direct text extraction first
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            logger.warning(f"Direct PDF extraction failed: {e}")
        
        # If no text or very little text, use OCR with Odia support
        if len(text.strip()) < 50:
            logger.info("Using OCR for scanned PDF with multilingual support")
            try:
                pages = convert_from_path(file_path, dpi=300)
                for page in pages:
                    # OCR with English + Odia language support
                    ocr_text = pytesseract.image_to_string(
                        page, 
                        lang='eng+ori',  # English + Odia
                        config='--oem 3 --psm 6'
                    )
                    text += ocr_text + "\n"
            except Exception as e:
                logger.error(f"OCR failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Could not process scanned document"
                )
        
        return text
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract DOCX text with proper encoding"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"DOCX processing failed: {e}")
    
    async def _extract_txt_text(self, file_path: str) -> str:
        """Extract text with multiple encoding support"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        raise HTTPException(
            status_code=400,
            detail="Could not decode text file with any supported encoding"
        )

# Enhanced AI processing functions
class AIProcessor:
    def __init__(self, model_manager: ModelManager):
        self.models = model_manager.models
        self.device = config.DEVICE
    
    async def summarize_document(self, text: str, language: str = "en", max_length: int = 150, min_length: int = 30) -> Dict[str, Any]:
        """GPU-accelerated document summarization with multilingual support"""
        start_time = time.time()
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty document provided")
        
        # Determine model based on language
        if language.lower() in ['or', 'odia', 'ori']:
            model_name = 'summarizer_odia'
            model_used = "IndIC-BERT (Odia)"
        else:
            model_name = 'summarizer_en'
            model_used = "BART (English)"
        
        try:
            # Chunk large texts for GPU processing
            chunks = self._chunk_text(text, max_chunk_size=1024)
            summaries = []
            
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Skip very small chunks
                    summary = self.models[model_name](
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    summaries.append(summary[0]['summary_text'])
            
            # Combine summaries
            final_summary = " ".join(summaries)
            if len(summaries) > 1 and len(final_summary) > max_length * 1.5:
                # Summarize the summary if it's too long
                final_summary = self.models[model_name](
                    final_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
            
            processing_time = time.time() - start_time
            
            return {
                "summary": final_summary,
                "original_length": len(text.split()),
                "summary_length": len(final_summary.split()),
                "processing_time": round(processing_time, 2),
                "language": language,
                "model_used": model_used,
                "gpu_accelerated": self.device == "cuda"
            }
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Text summarization failed: {str(e)}"
            )
    
    async def semantic_search(self, query: str, top_k: int = 5, include_scores: bool = True) -> Dict[str, Any]:
        """GPU-accelerated semantic search in legal corpus"""
        start_time = time.time()
        
        try:
            # Encode query using GPU
            query_embedding = self.models['semantic_model'].encode(
                query,
                convert_to_numpy=True
            )
            
            # Perform FAISS search
            scores, indices = self.models['faiss_index'].search(
                np.array([query_embedding]),
                top_k
            )
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid result
                    result = {
                        "rank": i + 1,
                        "text": model_manager.legal_corpus[idx],
                        "relevance_score": float(1 / (1 + score))  # Convert distance to similarity
                    }
                    if include_scores:
                        result["distance"] = float(score)
                    results.append(result)
            
            processing_time = time.time() - start_time
            
            return {
                "query": query,
                "results": results,
                "processing_time": round(processing_time, 3),
                "total_results": len(results),
                "gpu_accelerated": self.device == "cuda"
            }
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Legal search failed: {str(e)}"
            )
    
    async def extract_legal_entities(self, text: str) -> Dict[str, Any]:
        """Advanced legal entity extraction with confidence scores"""
        start_time = time.time()
        
        try:
            # Process with spaCy
            doc = self.models['nlp'](text)
            
            entities = []
            confidence_scores = {}
            
            # Extract entities with confidence
            for ent in doc.ents:
                entity_data = {
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_) or "Legal entity",
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": float(ent._.get("confidence", 0.85))  # Default confidence
                }
                entities.append(entity_data)
                confidence_scores[ent.label_] = confidence_scores.get(ent.label_, []) + [entity_data["confidence"]]
            
            # Extract legal citations
            citation_patterns = [
                r'\b([A-Z][a-zA-Z\s&]+ v\.? [A-Z][a-zA-Z\s&]+,?\s*\d{4}\s*[A-Z]*\s*\d*)\b',  # Case names
                r'\bAIR\s+\d{4}\s+[A-Z]+\s+\d+\b',  # AIR citations
                r'\b\d{4}\s+\(\d+\)\s+[A-Z]+\s+\d+\b',  # Law reports
                r'\b[A-Z]+\s*\d{4}\s*[A-Z]*\s*\d+\b',  # General citations
            ]
            
            legal_citations = []
            for pattern in citation_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                legal_citations.extend(matches)
            
            # Extract dates
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
                r'\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b',
                r'\b[A-Za-z]+\s+\d{1,2},?\s+\d{4}\b'
            ]
            
            dates = []
            for pattern in date_patterns:
                matches = re.findall(pattern, text)
                dates.extend(matches)
            
            # Extract monetary amounts
            amount_patterns = [
                r'Rs\.?\s*\d+(?:,\d+)*(?:\.\d+)?',
                r'₹\s*\d+(?:,\d+)*(?:\.\d+)?',
                r'\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:rupees?|lakhs?|crores?)\b'
            ]
            
            amounts = []
            for pattern in amount_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                amounts.extend(matches)
            
            # Calculate average confidence scores
            avg_confidence = {}
            for label, scores in confidence_scores.items():
                avg_confidence[label] = round(sum(scores) / len(scores), 2)
            
            processing_time = time.time() - start_time
            
            return {
                "entities": entities,
                "legal_citations": list(set(legal_citations)),
                "dates": list(set(dates)),
                "amounts": list(set(amounts)),
                "processing_time": round(processing_time, 2),
                "total_entities": len(entities),
                "confidence_scores": avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Entity extraction failed: {str(e)}"
            )
    
    async def format_legal_petition(self, petition_data: PetitionRequest) -> Dict[str, Any]:
        """Generate professionally formatted legal petition for Indian courts"""
        try:
            current_date = datetime.now().strftime("%d.%m.%Y")
            petition_number = f"LEGAL/AI/{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Enhanced petition template for Indian courts
            template = f"""
IN THE {petition_data.court_type.upper()}
{petition_data.case_type.upper()} PETITION NO. _______ OF {datetime.now().year}

CORAM: HON'BLE [COURT NAME]

BETWEEN:

{petition_data.petitioner.upper()}
[Address]                                                    ... PETITIONER
                                        
                                    AND

{petition_data.respondent.upper()}  
[Address]                                                    ... RESPONDENT

PETITION UNDER [RELEVANT PROVISIONS OF LAW]

TO,
THE HONOURABLE COURT

THE HUMBLE PETITION OF THE ABOVE NAMED PETITIONER MOST RESPECTFULLY SHOWETH:

GROUNDS FOR THE PETITION:

{petition_data.grounds}

PRAYER:

In the premises aforesaid, the Petitioner most respectfully prays that this Hon'ble Court may be pleased to:

{petition_data.prayer}

{"URGENT: This matter requires immediate hearing due to the nature of the case." if petition_data.urgency else ""}

Any other relief that this Hon'ble Court deems fit and proper in the circumstances of the case may also be granted.

VERIFICATION:

I, {petition_data.petitioner}, the Petitioner above named, do hereby verify that the contents of the above petition are true and correct to the best of my knowledge and belief, no part of it is false and nothing material has been concealed therefrom.

Place: _____________
Date: {current_date}                                        
                                                    
                                                    (Signature of Petitioner)
                                                    {petition_data.petitioner}

THROUGH ADVOCATE:

Name: _________________________
Registration No.: ______________
Chamber No.: __________________
Contact: ______________________

---
FILING DETAILS:
Petition No.: {petition_number}
Generated on: {datetime.now().strftime('%d %B %Y at %I:%M %p')}
Generated by: Decentralized AI Legal Assistant (Akash Network)
"""

            word_count = len(template.split())
            estimated_pages = max(1, word_count // 250)  # ~250 words per page
            
            return {
                "petition": template.strip(),
                "word_count": word_count,
                "estimated_pages": estimated_pages,
                "court_formatted": True,
                "timestamp": datetime.now().isoformat(),
                "petition_number": petition_number
            }
            
        except Exception as e:
            logger.error(f"Petition formatting error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Legal petition generation failed: {str(e)}"
            )
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1024) -> List[str]:
        """Split text into chunks for GPU processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > max_chunk_size:
                chunks.append(' '.join(current_chunk[:-1]))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

# Initialize processors
document_processor = DocumentProcessor()

# Startup event - Load models on Akash Network
@app.on_event("startup")
async def startup_event():
    """Initialize AI models on Akash Network GPU"""
    logger.info(f"🚀 Starting Decentralized AI Legal Assistant on Akash Network")
    logger.info(f"🖥️  GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    await model_manager.load_models()
    logger.info("✅ All systems ready on Akash Network!")

# Health and status endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """API root with Akash Network status"""
    return {
        "message": "🌐 Decentralized AI Legal Assistant - Akash Network Edition",
        "version": "2.0.0-akash",
        "status": "🟢 Running on decentralized infrastructure",
        "gpu_available": torch.cuda.is_available(),
        "device": config.DEVICE,
        "features": [
            "🤖 AI Document Summarization (English + Odia)",
            "🔍 Semantic Legal Search",
            "🏷️ Entity Extraction & Citation Analysis", 
            "⚖️ Legal Petition Generation",
            "🔒 Privacy-Preserving Processing",
            "🌐 Zero Local GPU Requirements"
        ],
        "akash_optimized": True,
        "deployment_track": "HackOdisha 5.0 - Akash Network"
    }

@app.get("/health")
async def health_check():
    """Detailed health check for Akash deployment"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB"
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "akash_network": True,
        "models_loaded": len(model_manager.models) > 0,
        "device": config.DEVICE,
        "gpu_info": gpu_info,
        "supported_languages": ["English", "Odia", "Hindi"],
        "supported_formats": config.SUPPORTED_FORMATS,
        "privacy_mode": "enabled"
    }

# Core API endpoints - GPU optimized for Akash Network
@app.post("/summarize", response_model=DocumentSummaryResponse)
async def summarize_document(
    file: UploadFile = File(..., description="Legal document to summarize"),
    language: str = Form("en", description="Language: en/or/hi"),
    max_length: int = Form(150, description="Maximum summary length"),
    min_length: int = Form(30, description="Minimum summary length")
):
    """GPU-accelerated document summarization on Akash Network"""
    
    if not model_manager.models:
        await model_manager.load_models()
    
    ai_processor = AIProcessor(model_manager)
    
    # Validate file
    if file.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE / 1e6:.1f} MB"
        )
    
    # Save and process file
    file_path = os.path.join(document_processor.upload_dir, f"{int(time.time())}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text with OCR support
        text = await document_processor.extract_text_with_ocr(file_path)
        
        if len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="No meaningful text found in document")
        
        # AI summarization
        result = await ai_processor.summarize_document(text, language, max_length, min_length)
        
        return DocumentSummaryResponse(**result)
        
    finally:
        # Clean up - privacy preserving
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/search", response_model=SemanticSearchResponse)
async def legal_semantic_search(request: SemanticSearchRequest):
    """Semantic search in legal knowledge base using GPU acceleration"""
    
    if not model_manager.models:
        await model_manager.load_models()
    
    ai_processor = AIProcessor(model_manager)
    result = await ai_processor.semantic_search(
        request.query, 
        request.top_k,
        request.include_scores
    )
    
    return SemanticSearchResponse(**result)

@app.post("/extract", response_model=EntityExtractionResponse)
async def extract_legal_entities(
    file: UploadFile = File(..., description="Legal document for entity extraction")
):
    """Extract legal entities, citations, and key information"""
    
    if not model_manager.models:
        await model_manager.load_models()
    
    ai_processor = AIProcessor(model_manager)
    
    # Validate and process file
    if file.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE / 1e6:.1f} MB"
        )
    
    file_path = os.path.join(document_processor.upload_dir, f"{int(time.time())}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        text = await document_processor.extract_text_with_ocr(file_path)
        
        if len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="No meaningful text found in document")
        
        result = await ai_processor.extract_legal_entities(text)
        return EntityExtractionResponse(**result)
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/petition", response_model=PetitionResponse)
async def generate_legal_petition(petition: PetitionRequest):
    """Generate formatted legal petition for Indian courts"""
    
    if not model_manager.models:
        await model_manager.load_models()
    
    ai_processor = AIProcessor(model_manager)
    result = await ai_processor.format_legal_petition(petition)
    
    return PetitionResponse(**result)

# Legacy endpoints for backward compatibility
@app.post("/upload_summarize/")
async def upload_summarize_legacy(
    file: UploadFile = File(...),
    language: str = Form("en")
):
    """Legacy endpoint - redirects to new /summarize"""
    return await summarize_document(file, language)

@app.post("/semantic_search/")
async def semantic_search_legacy(request: SemanticSearchRequest):
    """Legacy endpoint - redirects to new /search"""
    return await legal_semantic_search(request)

@app.post("/extract_entities/")
async def extract_entities_legacy(file: UploadFile = File(...)):
    """Legacy endpoint - redirects to new /extract"""
    return await extract_legal_entities(file)

@app.post("/format_petition/")
async def format_petition_legacy(petition: PetitionRequest):
    """Legacy endpoint - redirects to new /petition"""
    return await generate_legal_petition(petition)

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for production deployment"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An error occurred while processing your request",
            "akash_network": True,
            "support": "Contact HackOdisha team for assistance"
        }
    )

# Development server
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Decentralized AI Legal Assistant for Akash Network")
    
    uvicorn.run(
        "akash_legal_ai:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,  # Disabled for production
        workers=1,     # Single worker for GPU optimization
        log_level="info"
    )