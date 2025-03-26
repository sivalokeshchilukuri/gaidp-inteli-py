import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class PDFAgent:
    def __init__(self, collection_name: str = "pdf_documents", model_name: str = "microsoft/phi-2"):
        """
        Initialize the PDF Agent with Chroma DB and Phi model
        
        Args:
            collection_name: Name of the Chroma collection
            model_name: Name of the Phi model to use for embeddings
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path="chroma_db")
        
        # Create or get collection
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def extract_text_from_pdf(self, file_path: str) -> List[str]:
        """
        Extract text from PDF and split into chunks
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of text chunks
        """
        reader = PdfReader(file_path)
        text_chunks = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Simple chunking - you may want more sophisticated chunking
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                text_chunks.extend(chunks)
                
        return text_chunks
    
    def store_document(self, file_path: str, metadata: Optional[Dict] = None) -> None:
        """
        Process and store a PDF document in Chroma DB
        
        Args:
            file_path: Path to the PDF file
            metadata: Optional metadata to associate with the document
        """
        if not metadata:
            metadata = {}
            
        # Extract text from PDF
        text_chunks = self.extract_text_from_pdf(file_path)
        
        # Generate embeddings and store in Chroma
        ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(text_chunks))]
        metadatas = [{**metadata, "source": file_path} for _ in range(len(text_chunks))]
        
        self.collection.add(
            documents=text_chunks,
            metadatas=metadatas,
            ids=ids
        )
    
    def query_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Query the stored documents
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks with scores
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return [
            {
                "document": doc,
                "metadata": meta,
                "score": score
            }
            for doc, meta, score in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    
    def get_context_for_question(self, question: str, n_chunks: int = 3) -> str:
        """
        Get relevant context for a question from stored documents
        
        Args:
            question: The question to answer
            n_chunks: Number of relevant chunks to retrieve
            
        Returns:
            Context string with relevant information
        """
        results = self.query_documents(question, n_results=n_chunks)
        context = "\n\n".join([r["document"] for r in results])
        return context

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = PDFAgent()
    
    # Store a PDF document
    agent.store_document("FR_Y-14Q20240331_i.pdf", {"title": "Sample Document"})
    
    # Query the documents
    results = agent.query_documents("What is the main topic of the document?")

    for result in results:
        print(f"Score: {result['score']:.4f}")
        print(f"Content: {result['document'][:200]}...")
        print("---")
    
    # Get context for a question
    context = agent.get_context_for_question("Explain the key concepts")
    print("\nContext for question:")
    print(context[:500] + "...")