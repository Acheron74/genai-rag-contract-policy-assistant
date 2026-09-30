import sys
from unittest.mock import MagicMock
import json

# Mock dependencies
sys.modules['app.rag_service'] = MagicMock()
sys.modules['app.rag_service'].RAGService = MagicMock()

# Mock Pydantic
class MockBaseModel:
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)

def mock_field(*args, **kwargs):
    return None

pydantic_mock = MagicMock()
pydantic_mock.BaseModel = MockBaseModel
pydantic_mock.Field = mock_field
sys.modules['pydantic'] = pydantic_mock

# Now import the components we want to test
from app.contract_analyzer import ContractAnalyzer
mock_schemas = MagicMock()
mock_schemas.ContractSchema = MockBaseModel
mock_schemas.PaymentTerms = MockBaseModel
sys.modules['app.schemas'] = mock_schemas

from app.contract_analyzer import ContractAnalyzer

def test_coercion():
    rag_service = MagicMock()
    rag_service.collection.get.return_value = {
        'documents': ['test doc'],
        'metadatas': [{'clause_types': 'general', 'chunk_index': 0}]
    }
    rag_service.pipe = MagicMock()

    analyzer = ContractAnalyzer(rag_service)

    test_cases = [
        {
            "name": "Lists in string fields",
            "llm_output": {
                "doc_id": "test1.pdf",
                "parties": ["Company A", "Company B"],
                "notes": ["Note 1", "Note 2"]
            },
            "expected_parties": "Company A, Company B",
            "expected_notes": "Note 1, Note 2"
        },
        {
            "name": "Dicts in string fields",
            "llm_output": {
                "doc_id": "test2.pdf",
                "effective_date": {"year": 2023, "month": "January"},
                "governing_law": {"state": "NY", "country": "USA"}
            },
            "expected_effective_date": "year: 2023, month: January",
            "expected_governing_law": "state: NY, country: USA"
        },
        {
            "name": "Primitives in string fields",
            "llm_output": {
                "doc_id": "test3.pdf",
                "notes": 12345
            },
            "expected_notes": "12345"
        },
        {
            "name": "Malformed payment_terms (string)",
            "llm_output": {
                "doc_id": "test4.pdf",
                "payment_terms": "Net 30"
            },
            "expected_payment": {"description": "Net 30"}
        },
        {
            "name": "Malformed payment_terms (list)",
            "llm_output": {
                "doc_id": "test5.pdf",
                "payment_terms": ["Net 30", "2% discount"]
            },
            "expected_payment": {"description": "Net 30, 2% discount"}
        },
        {
            "name": "Malformed payment_terms (dict with non-string subfields)",
            "llm_output": {
                "doc_id": "test6.pdf",
                "payment_terms": {
                    "description": ["Net 30", "2% discount"],
                    "due_date": 30
                }
            },
            "expected_payment": {"description": "Net 30, 2% discount", "due_date": "30"}
        },
        {
            "name": "Malformed payment_terms (dict without schema keys)",
            "llm_output": {
                "doc_id": "test7.pdf",
                "payment_terms": {
                    "amount": 1000,
                    "currency": "USD"
                }
            },
            "expected_payment": {"description": "amount: 1000, currency: USD"}
        },
        {
            "name": "Non-integer risk_score",
            "llm_output": {
                "doc_id": "test8.pdf",
                "risk_score": "45"
            },
            "expected_risk_score": 45
        },
        {
            "name": "Invalid risk_score",
            "llm_output": {
                "doc_id": "test9.pdf",
                "risk_score": "high"
            },
            "expected_risk_score": 0
        }
    ]

    for case in test_cases:
        print(f"Running test case: {case['name']}")
        rag_service.pipe.return_value = [{'generated_text': f"<|assistant|>{json.dumps(case['llm_output'])}"}]

        result = analyzer.analyze(case['llm_output']['doc_id'])

        if "expected_parties" in case:
            actual = getattr(result, 'parties', None)
            assert actual == case['expected_parties'], f"Parties mismatch: expected {case['expected_parties']} ({type(case['expected_parties'])}), got {actual} ({type(actual)})"
        if "expected_notes" in case:
            actual = getattr(result, 'notes', None)
            assert actual == case['expected_notes'], f"Notes mismatch: expected {case['expected_notes']} ({type(case['expected_notes'])}), got {actual} ({type(actual)})"
        if "expected_effective_date" in case:
            actual = getattr(result, 'effective_date', None)
            assert actual == case['expected_effective_date'], f"Effective date mismatch: expected {case['expected_effective_date']} ({type(case['expected_effective_date'])}), got {actual} ({type(actual)})"
        if "expected_governing_law" in case:
            actual = getattr(result, 'governing_law', None)
            assert actual == case['expected_governing_law'], f"Governing law mismatch: expected {case['expected_governing_law']} ({type(case['expected_governing_law'])}), got {actual} ({type(actual)})"

        if "expected_payment" in case:
            pt = result.payment_terms
            expected = case['expected_payment']
            if isinstance(pt, dict):
                actual_pt = pt
            else:
                actual_pt = {k: v for k, v in pt.__dict__.items() if not k.startswith('_')}

            for k, v in expected.items():
                actual_val = actual_pt.get(k)
                assert actual_val == v, f"Payment term {k} mismatch: expected {v} ({type(v)}), got {actual_val} ({type(actual_val)})"

        if "expected_risk_score" in case:
            actual = getattr(result, 'risk_score', None)
            assert actual == case['expected_risk_score'], f"Risk score mismatch: expected {case['expected_risk_score']} ({type(case['expected_risk_score'])}), got {actual} ({type(actual)})"

        print(f"Case {case['name']} PASSED")

if __name__ == "__main__":
    try:
        test_coercion()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
