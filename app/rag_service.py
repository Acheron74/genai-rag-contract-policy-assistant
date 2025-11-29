import os
import chromadb
from typing import List, Dict, Tuple
import logging
# Import transformers only if needed to avoid overhead if just importing the class
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from app.embeddings import embed_texts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
COLLECTION_NAME = "compliance_contract_docs"
# Distance threshold for relevance. Lower is better for L2.
# 1.0 corresponds to cosine similarity of 0.5.
DISTANCE_THRESHOLD = 1.0 

class RAGService:
    """
    Service for Retrieval-Augmented Generation (RAG).
    Handles:
    1. Connecting to ChromaDB vector store.
    2. Loading the local LLM (Phi-3-mini).
    3. Retrieving relevant documents based on query embeddings.
    4. Generating answers using the LLM and retrieved context.
    """
    def __init__(self):
        logger.info(f"Connecting to ChromaDB at {VECTOR_STORE_DIR}")
        self.client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        
        # Load LLM
        # Using Phi-3-mini-4k-instruct
        self.model_id = "microsoft/Phi-3-mini-4k-instruct"
        self.pipe = None
        self._load_llm()

    def _load_llm(self):
        """
        Loads the Phi-3-mini model and tokenizer from local cache or Hugging Face.
        Configures device (GPU/CPU) automatically.
        """
        logger.info(f"Loading LLM: {self.model_id}")
        try:
            # Check for GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map=device, 
                torch_dtype="auto", 
                trust_remote_code=True
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            logger.info("LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self.pipe = None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieves the top_k most relevant document chunks for a given query.
        """
        query_embedding = embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_docs = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                retrieved_docs.append({
                    "text": doc,
                    "metadata": metadata,
                    "score": distance 
                })
        return retrieved_docs

    def answer_question(self, question: str) -> Dict:
        """
        Generates an answer to a compliance question using RAG.
        1. Retrieves relevant chunks.
        2. Filters chunks by similarity threshold.
        3. Constructs a prompt with context.
        4. Generates answer via LLM.
        """
        docs = self.retrieve(question)
        
        # Filter by threshold
        relevant_docs = [d for d in docs if d['score'] < DISTANCE_THRESHOLD]
        
        if not relevant_docs:
            return {
                "answer": "No relevant info found.",
                "citations": [],
                "similarity_scores": []
            }
            
        context = "\n\n".join([f"Source: {d['metadata']['source']}\nContent: {d['text']}" for d in relevant_docs])
        
        prompt = f"""<|system|>
You are a helpful compliance assistant. Answer the question based ONLY on the provided context. 
If the answer is not in the context, say "No relevant info found."
Include citations to the source documents.
<|end|>
<|user|>
Context:
{context}

Question: {question}
<|end|>
<|assistant|>"""

        if self.pipe:
            output = self.pipe(prompt, max_new_tokens=900, do_sample=False)
            generated_text = output[0]['generated_text']
            # Extract assistant response
            answer = generated_text.split("<|assistant|>")[-1].strip()
        else:
            answer = "LLM not loaded. Cannot generate answer."

        citations = list(set([d['metadata']['source'] for d in relevant_docs]))
        scores = [d['score'] for d in relevant_docs]

        return {
            "answer": answer,
            "citations": citations,
            "similarity_scores": scores
        }

