from pydantic import BaseModel, Field
from typing import List, Optional

class PaymentTerms(BaseModel):
    description: Optional[str] = Field(None, description="Description of payment terms")
    due_date: Optional[str] = Field(None, description="Payment due date or period")

class ContractSchema(BaseModel):
    doc_id: Optional[str] = Field(None, description="Document ID or Filename")
    parties: Optional[str] = Field(None, description="Parties involved in the contract")
    effective_date: Optional[str] = Field(None, description="Effective date of the contract")
    termination_clause: Optional[str] = Field(None, description="Termination clause details")
    confidentiality_clause: Optional[str] = Field(None, description="Confidentiality clause details")
    governing_law: Optional[str] = Field(None, description="Governing law jurisdiction")
    payment_terms: Optional[PaymentTerms] = Field(None, description="Payment terms details")
    risk_score: Optional[int] = Field(None, description="Risk score from 0 to 100")
    notes: Optional[str] = Field(None, description="Additional notes or observations")

class QueryRequest(BaseModel):
    question: str
    collection_name: str = "compliance_contract_docs"

class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    similarity_scores: List[float]

class AnalyzeRequest(BaseModel):
    file_name: str

class HealthResponse(BaseModel):
    status: str
