from sentence_transformers import SentenceTransformer
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the model instance
_model = None
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# Note: CACHE_DIR is controlled by HF_HOME environment variable, not this variable directly.
# Ensure HF_HOME is set before running the application if you want a custom cache location.

def get_embedding_model():
    """
    Lazy loads the SentenceTransformer model.
    Uses the global _model instance to avoid reloading on every request.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        # Load model. This will download to default cache (or HF_HOME) if not present.
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embeds a list of texts using the local MPNet model.
    Returns a list of vectors (lists of floats).
    
    Args:
        texts: List of strings to embed.
        
    Returns:
        List of embeddings, where each embedding is a list of floats.
    """
    if not texts:
        return []
        
    model = get_embedding_model()
    # normalize_embeddings=True is good for cosine similarity
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


