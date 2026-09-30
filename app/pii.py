import spacy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model. Ensure 'en_core_web_sm' is installed.
# python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading en_core_web_sm model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def mask_pii(text: str) -> str:
    """
    Masks PII in the text using spaCy NER.
    Entities masked: PERSON, ORG, GPE.
    
    Args:
        text: The input text string.
        
    Returns:
        The text with PII entities replaced by [LABEL] placeholders.
    """
    if not text:
        return ""
        
    doc = nlp(text)
    return _mask_entities(doc)

def mask_pii_batch(texts: list[str]) -> list[str]:
    """
    Masks PII in a list of texts using spaCy's batch processing (nlp.pipe).
    Entities masked: PERSON, ORG, GPE.
    
    Args:
        texts: A list of input text strings.

    Returns:
        A list of texts with PII entities replaced by [LABEL] placeholders.
    """
    if not texts:
        return []
    
    return [_mask_entities(doc) for doc in nlp.pipe(texts)]

def _mask_entities(doc) -> str:
    """
    Internal helper to mask entities in a spaCy Doc object.
    """
    text = doc.text
    replacements = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            replacements.append((ent.start_char, ent.end_char, f"[{ent.label_}]"))
            
    # Sort replacements by start_char in reverse order to apply without offset issues
    replacements.sort(key=lambda x: x[0], reverse=True)
    
    result = text
    for start, end, label in replacements:
        result = result[:start] + label + result[end:]
        
    return result

