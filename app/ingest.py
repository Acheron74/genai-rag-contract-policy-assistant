import os
import glob
import pdfplumber
import chromadb
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.embeddings import embed_texts
from app.pii import mask_pii
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
COLLECTION_NAME = "compliance_contract_docs"

# Expanded keywords based on CUAD master_clauses.csv
CLAUSE_KEYWORDS = {
    "termination": [
        "terminate", "termination", "cancel", "cancellation", "rescind", 
        "end of agreement", "convenience", "breach", "notice period"
    ],
    "effective_date": [
        "effective date", "commencement date", "start date", "dated as of", 
        "made as of", "initial term", "expiration date"
    ],
    "renewal": [
        "renew", "renewal", "extension", "extend", "automatic renewal", 
        "successive term"
    ],
    "parties": [
        "parties", "between", "among", "entered into by", "buyer", "seller", 
        "provider", "customer", "licensor", "licensee", "grantor", "grantee",
        "landlord", "tenant", "contractor"
    ],
    "governing_law": [
        "governing law", "jurisdiction", "laws of", "venue", "courts of", 
        "construed in accordance", "dispute resolution", "arbitration"
    ],
    "confidentiality": [
        "confidential", "confidentiality", "non-disclosure", "secrecy", 
        "proprietary information", "trade secret"
    ],
    "payment_terms": [
        "payment", "fees", "invoice", "due date", "remittance", "pricing", 
        "compensation", "royalties", "minimum commitment"
    ],
    "liability": [
        "liability", "indemnification", "damages", "limitation of liability", 
        "hold harmless", "cap on liability", "liquidated damages", "warranty"
    ],
    "license_ip": [
        "license grant", "intellectual property", "ownership", "work for hire", 
        "assignment", "patent", "trademark", "copyright", "source code"
    ],
    "restrictive_covenants": [
        "non-compete", "exclusivity", "non-solicit", "competitive restriction", 
        "most favored nation", "territory"
    ]
}

def detect_clause_types(text: str) -> str:
    """
    Returns a comma-separated string of detected clause types based on keywords.
    """
    text_lower = text.lower()
    detected = []
    for clause, keywords in CLAUSE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(clause)
    return ",".join(detected) if detected else "general"

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
    return text

def ingest_documents():
    """
    Main ingestion function.
    1. Scans docs/policies and docs/contracts for PDFs.
    2. Extracts text using pdfplumber.
    3. Chunks text using RecursiveCharacterTextSplitter.
    4. Masks PII in each chunk.
    5. Generates embeddings using MPNet.
    6. Tags chunks with clause types (for contracts).
    7. Stores everything in local ChromaDB.
    """
    # Initialize Chroma Client
    logger.info(f"Initializing ChromaDB at {VECTOR_STORE_DIR}")
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    # Find PDFs
    policy_files = glob.glob(os.path.join(DOCS_DIR, "policies", "*.pdf"))
    contract_files = glob.glob(os.path.join(DOCS_DIR, "contracts", "*.pdf"))
    all_files = policy_files + contract_files

    logger.info(f"Found {len(all_files)} documents to ingest.")

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        doc_type = "policy" if "policies" in file_path else "contract"
        
        logger.info(f"Processing {file_name} ({doc_type})...")
        
        raw_text = extract_text_from_pdf(file_path)
        if not raw_text:
            logger.warning(f"No text extracted from {file_name}")
            continue

        # Chunk first to avoid spaCy length limits
        raw_chunks = text_splitter.split_text(raw_text)
        
        if not raw_chunks:
            continue

        # Mask PII in each chunk
        # logger.info("Masking PII...")
        chunks = [mask_pii(chunk) for chunk in raw_chunks]

        # Embed
        # logger.info(f"Embedding {len(chunks)} chunks...")
        embeddings = embed_texts(chunks)

        # Prepare data for Chroma
        ids = [f"{file_name}_{i}" for i in range(len(chunks))]
        metadatas = []
        for i, chunk in enumerate(chunks):
            clause_tags = detect_clause_types(chunk)
            metadatas.append({
                "source": file_name, 
                "type": doc_type, 
                "chunk_index": i,
                "clause_types": clause_tags
            })
        
        # Add to collection
        # Upsert to avoid duplicates if run multiple times
        collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(chunks)} chunks from {file_name}")

    logger.info("Ingestion complete.")

if __name__ == "__main__":
    ingest_documents()
