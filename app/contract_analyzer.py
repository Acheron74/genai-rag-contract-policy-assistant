import json
import logging
from typing import Dict, Any, Optional, List
from app.rag_service import RAGService
from app.schemas import ContractSchema, PaymentTerms
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContractAnalyzer:
    """
    Analyzes contract documents to extract structured information using an LLM.
    Uses 'Smart Context' building to select relevant chunks based on clause types
    to fit within the LLM's context window and avoid OOM errors.
    """
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    def analyze(self, file_name: str) -> ContractSchema:
        """
        Analyzes a specific contract file.
        1. Retrieves all chunks for the file from ChromaDB.
        2. Filters and organizes chunks into relevant sections (Parties, Dates, etc.).
        3. Constructs a prompt with this 'Smart Context'.
        4. Generates a JSON response using the LLM.
        5. Validates and cleans the output to match ContractSchema.
        """
        logger.info(f"Analyzing contract: {file_name}")
        
        # 1. Retrieve ALL chunks for the file
        # We fetch all chunks first and then filter in memory because Chroma's 
        # metadata filtering is exact match, and our tags are comma-separated strings.
        results = self.rag_service.collection.get(
            where={"source": file_name},
            include=["documents", "metadatas"]
        )
        
        documents = results['documents']
        metadatas = results['metadatas']
        
        if not documents:
            logger.warning(f"No documents found for {file_name}")
            return ContractSchema(doc_id=file_name, notes="No content found.")

        # 2. Build "Smart Context"
        # Organize chunks into buckets based on their clause_type tags.
        relevant_sections = {
            "parties": [],
            "effective_date": [],
            "termination": [],
            "governing_law": [],
            "confidentiality": [],
            "payment_terms": [],
            "liability": [], # Added from CUAD for risk analysis
            "general": [] 
        }

        for doc, meta in zip(documents, metadatas):
            tags = meta.get("clause_types", "general")
            
            # Distribute doc to relevant buckets
            if "parties" in tags: relevant_sections["parties"].append(doc)
            if "effective_date" in tags or "renewal" in tags: relevant_sections["effective_date"].append(doc)
            if "termination" in tags: relevant_sections["termination"].append(doc)
            if "governing_law" in tags: relevant_sections["governing_law"].append(doc)
            if "confidentiality" in tags: relevant_sections["confidentiality"].append(doc)
            if "payment_terms" in tags: relevant_sections["payment_terms"].append(doc)
            
            # Collect high-risk sections for risk scoring
            if "liability" in tags or "restrictive_covenants" in tags or "license_ip" in tags:
                relevant_sections["liability"].append(doc)
            
            # Keep first 3 chunks as "Intro/General" context (usually contains parties/dates)
            if meta.get("chunk_index", 999) < 3:
                relevant_sections["general"].append(doc)

        def get_top_chunks(chunks: List[str], limit: int = 2) -> str:
            """Helper to join top N chunks from a list."""
            return "\n...\n".join(chunks[:limit])

        # Construct the context string by prioritizing specific sections
        context_parts = [
            f"--- INTRO ---\n{get_top_chunks(relevant_sections['general'], 3)}",
            f"--- PARTIES ---\n{get_top_chunks(relevant_sections['parties'])}",
            f"--- DATES ---\n{get_top_chunks(relevant_sections['effective_date'])}",
            f"--- TERMINATION ---\n{get_top_chunks(relevant_sections['termination'])}",
            f"--- LAW ---\n{get_top_chunks(relevant_sections['governing_law'])}",
            f"--- PAYMENT ---\n{get_top_chunks(relevant_sections['payment_terms'])}",
            f"--- RISK FACTORS (Liability/IP/Restrictions) ---\n{get_top_chunks(relevant_sections['liability'], 3)}"
        ]
        
        full_context = "\n\n".join(context_parts)
        
        # Hard limit to ~6000 chars to prevent OOM on 8GB VRAM GPUs
        context_text = full_context[:6000]
        
        # 3. Build Prompt
        prompt = f"""<|system|>
You are a legal expert. Analyze the following contract text segments and extract the required fields in STRICT JSON format.
Do not include any markdown formatting or explanation. Just the JSON.
Fields required:
- doc_id: "{file_name}"
- parties: Who are the parties?
- effective_date: When does it start?
- termination_clause: Summary of termination rights.
- confidentiality_clause: Summary of confidentiality obligations.
- governing_law: Which jurisdiction?
- payment_terms: {{ "description": "...", "due_date": "..." }}
- risk_score: Integer 0-100 based on risk (high liability, strict termination = high risk).
- notes: Any other key observations.

If a field is not found, use null.
<|end|>
<|user|>
Contract Segments:
{context_text}
<|end|>
<|assistant|>"""

        # 4. Generate
        if self.rag_service.pipe:
            # Reduced max_new_tokens to save memory
            output = self.rag_service.pipe(prompt, max_new_tokens=500, do_sample=False)
            generated_text = output[0]['generated_text']
            response = generated_text.split("<|assistant|>")[-1].strip()
            
            # Clean up markdown code blocks if present
            response = response.replace("```json", "").replace("```", "").strip()
            
            try:
                data = json.loads(response)
                
                # --- FIX: Handle Lists AND Dictionaries for String Fields ---
                # LLMs sometimes return lists or dicts for fields defined as strings.
                # We convert them to strings to satisfy Pydantic validation.
                string_fields = ["parties", "effective_date", "termination_clause", "confidentiality_clause", "governing_law", "notes"]
                for field in string_fields:
                    val = data.get(field)
                    if val:
                        if isinstance(val, list):
                            # Convert list to string
                            data[field] = ", ".join([str(item) for item in val])
                        elif isinstance(val, dict):
                            # Convert dict to string (e.g. "key: value, key2: value2")
                            data[field] = ", ".join([f"{k}: {v}" for k, v in val.items()])
                
                # Fix for payment_terms (Schema expects object, LLM might return string or list)
                pt = data.get("payment_terms")
                if pt:
                    if isinstance(pt, str):
                         data["payment_terms"] = {"description": pt}
                    elif isinstance(pt, list):
                        # Join list items into a single description string
                        description = ", ".join([str(item) for item in pt])
                        data["payment_terms"] = {"description": description}
                
                contract_data = ContractSchema(**data)
                return contract_data
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON: {response}")
                return ContractSchema(doc_id=file_name, notes="Failed to parse analysis result. Raw output: " + response[:100])
            except Exception as e:
                logger.error(f"Validation error: {e}")
                return ContractSchema(doc_id=file_name, notes=f"Validation error: {str(e)}")
        else:
            return ContractSchema(doc_id=file_name, notes="LLM not loaded.")

