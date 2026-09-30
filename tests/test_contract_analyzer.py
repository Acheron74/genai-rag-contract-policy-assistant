
import sys
from unittest.mock import MagicMock

# 1. First, create mock classes that act like Pydantic models
class MockBaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            # If v is a dict and we are expecting a sub-model, we should ideally wrap it.
            # But for simplicity, we'll just check what the test expects.
            setattr(self, k, v)
    def dict(self):
        return self.__dict__

class PaymentTerms(MockBaseModel):
    pass

class ContractSchema(MockBaseModel):
    def __init__(self, **data):
        super().__init__(**data)
        if hasattr(self, 'payment_terms') and isinstance(self.payment_terms, dict):
            self.payment_terms = PaymentTerms(**self.payment_terms)

# 2. Inject our mock classes into sys.modules BEFORE anything imports app.schemas
mock_schemas = MagicMock()
mock_schemas.PaymentTerms = PaymentTerms
mock_schemas.ContractSchema = ContractSchema

sys.modules['app.schemas'] = mock_schemas
sys.modules['pydantic'] = MagicMock()
sys.modules['chromadb'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['app.embeddings'] = MagicMock()

import json
import pytest
from app.contract_analyzer import ContractAnalyzer

@pytest.fixture
def mock_rag_service():
    mock = MagicMock()
    mock.collection.get.return_value = {
        'documents': ['doc1'],
        'metadatas': [{'clause_types': 'general', 'chunk_index': 0}]
    }
    return mock

def test_payment_terms_string_conversion(mock_rag_service):
    analyzer = ContractAnalyzer(mock_rag_service)
    llm_output = {
        "doc_id": "test.pdf",
        "payment_terms": "Net 30 days",
        "risk_score": 10
    }
    mock_rag_service.pipe.return_value = [{'generated_text': f"<|assistant|>{json.dumps(llm_output)}"}]
    result = analyzer.analyze("test.pdf")
    assert result.payment_terms.description == "Net 30 days"

def test_payment_terms_list_conversion(mock_rag_service):
    analyzer = ContractAnalyzer(mock_rag_service)
    llm_output = {
        "doc_id": "test.pdf",
        "payment_terms": ["Net 30", "Late fee 5%"],
        "risk_score": 10
    }
    mock_rag_service.pipe.return_value = [{'generated_text': f"<|assistant|>{json.dumps(llm_output)}"}]
    result = analyzer.analyze("test.pdf")
    assert result.payment_terms.description == "Net 30, Late fee 5%"

def test_payment_terms_dict_normalization(mock_rag_service):
    analyzer = ContractAnalyzer(mock_rag_service)
    llm_output = {
        "doc_id": "test.pdf",
        "payment_terms": {"description": ["Net 30"], "due_date": 30},
        "risk_score": 20
    }
    mock_rag_service.pipe.return_value = [{'generated_text': f"<|assistant|>{json.dumps(llm_output)}"}]
    result = analyzer.analyze("test.pdf")
    assert result.payment_terms.description == "Net 30"
    assert result.payment_terms.due_date == "30"

def test_payment_terms_malformed_dict(mock_rag_service):
    analyzer = ContractAnalyzer(mock_rag_service)
    # Dict with no expected keys should be flattened to description
    llm_output = {
        "doc_id": "test.pdf",
        "payment_terms": {"term1": "30 days", "term2": "none"},
        "risk_score": 20
    }
    mock_rag_service.pipe.return_value = [{'generated_text': f"<|assistant|>{json.dumps(llm_output)}"}]
    result = analyzer.analyze("test.pdf")
    assert "term1: 30 days" in result.payment_terms.description
    assert "term2: none" in result.payment_terms.description

def test_risk_score_casting_valid_string(mock_rag_service):
    analyzer = ContractAnalyzer(mock_rag_service)
    llm_output = {
        "doc_id": "test.pdf",
        "risk_score": "42"
    }
    mock_rag_service.pipe.return_value = [{'generated_text': f"<|assistant|>{json.dumps(llm_output)}"}]
    result = analyzer.analyze("test.pdf")
    assert result.risk_score == 42

def test_risk_score_casting_invalid(mock_rag_service):
    analyzer = ContractAnalyzer(mock_rag_service)
    llm_output = {
        "doc_id": "test.pdf",
        "risk_score": "high"
    }
    mock_rag_service.pipe.return_value = [{'generated_text': f"<|assistant|>{json.dumps(llm_output)}"}]
    result = analyzer.analyze("test.pdf")
    assert result.risk_score == 0

def test_risk_score_none(mock_rag_service):
    analyzer = ContractAnalyzer(mock_rag_service)
    llm_output = {
        "doc_id": "test.pdf",
        "risk_score": None
    }
    mock_rag_service.pipe.return_value = [{'generated_text': f"<|assistant|>{json.dumps(llm_output)}"}]
    result = analyzer.analyze("test.pdf")
    assert result.risk_score == 0
