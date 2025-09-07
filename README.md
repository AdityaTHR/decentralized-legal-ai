# Decentralized AI Legal Assistant - HackOdisha 5.0

## Overview
The Decentralized AI Legal Assistant is an AI-powered application designed to simplify legal document processing in India. It supports multilingual summarization, semantic search, entity extraction, and petition generation with special focus on Odia language support. The backend is optimized for deployment on Akash Network’s decentralized GPU cloud infrastructure, allowing users to access advanced AI without any local resource requirements.

## Features
- **Document Summarization**: AI-powered summaries of legal documents in English and Odia.
- **Semantic Search**: Natural language search for relevant legal precedents.
- **Entity Extraction**: Extract legal entities, citations, dates, and amounts.
- **Petition Generation**: Auto-generate professionally formatted legal petitions.
- **Privacy-Preserving**: Documents processed on decentralized infrastructure without retention.
- **Universal Access**: Works on any device, no GPU or installation required.
- **Akash Network Deployment**: Leveraging permissionless GPU cloud for cost-effective AI processing.

## Tech Stack
- Backend: FastAPI, PyTorch (GPU-accelerated), Transformers, SentenceTransformers, spaCy
- Frontend: Modern responsive HTML, CSS, JavaScript
- Deployment: Docker, Akash Network (SDL configuration provided)
- OCR Support: pdf2image, pytesseract for scanned legal documents

## Getting Started

### Prerequisites
- Python 3.8+
- Docker (for containerized deployment)
- Git
- Akash Network account for deployment (optional)

### Installation & Running Locally
Clone the repository:
git clone https://github.com/AdityaTHR/decentralized-legal-ai.git
cd decentralized-legal-ai
python -m venv venv


Activate venv
Windows:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn document_ai:app --reload --host 0.0.0.0 --port 8000


Linux/Mac:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
uvicorn document_ai:app --reload --host 0.0.0.0 --port 8000


Access API at `http://localhost:8000/` and frontend via `index.html`.

### Deploying on Akash Network
- Build and push Docker image using included `Dockerfile.akash`.
- Use `akash-deploy.yml` SDL file to deploy on Akash Network GPU cloud.
- Refer to [AKASH_DEPLOYMENT.md](docs/AKASH_DEPLOYMENT.md) for full deployment steps.

## Usage
- Upload legal documents (PDF, DOCX, TXT) via API or frontend.
- Choose language for summarization (English or Odia).
- Perform semantic search queries to find relevant legal cases.
- Extract entities and citations from uploaded documents.
- Generate formatted legal petitions for court filing.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to enhance features, fix bugs, or improve documentation.

## License
This project is licensed under the MIT License.

## Acknowledgments
- HackOdisha 5.0 organizers and mentors
- Akash Network for decentralized GPU infrastructure
- Hugging Face for AI models and transformers
- SpaCy community for NLP tools
- The Odia tech and legal community inspiring regional support

---

*This project aims to democratize access to AI-powered legal assistance across India, using decentralized infrastructure for privacy, cost-effectiveness, and universal accessibility.*
