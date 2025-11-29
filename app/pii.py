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
    
    # We iterate and replace characters with '*' or replace the whole entity with [LABEL]
    # Replacing with [LABEL] is cleaner for RAG context.
    
    # We need to be careful with replacing in-place if we change string length.
    # Easier to rebuild string or use replacements.
    
    # Let's use a list of replacements and apply them.
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

