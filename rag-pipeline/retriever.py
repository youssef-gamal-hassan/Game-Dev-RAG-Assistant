import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class RetrieverHandler:
    """
    Handles loading and querying FAISS indexes using Google Generative AI embeddings.
    Supports per-heading FAISS index loading for efficient filtering.
    """

    def __init__(self, base_faiss_dir: str):
        """
        :param base_faiss_dir: Directory containing subfolders for each heading's FAISS index.
        """
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=self.api_key
        )
        self.base_faiss_dir = base_faiss_dir
        self.vectorstore = None
        self.retriever = None

    def _safe_heading_name(self, heading: str) -> str:
        """
        Sanitize heading for use as folder name.
        """
        return heading.replace("/", "_").replace("\\", "_").replace(" ", "_")

    def load_heading_index(self, heading: str):
        """
        Load the FAISS index for a specific heading.
        """
        safe_heading = self._safe_heading_name(heading)
        heading_dir = os.path.join(self.base_faiss_dir, safe_heading)

        if not os.path.exists(heading_dir):
            raise FileNotFoundError(f"No FAISS index found for heading: {heading}")

        self.vectorstore = FAISS.load_local(
            heading_dir,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()

    def query(self, query_text: str, k: int = 4):
        """
        Retrieve top-k relevant documents from the currently loaded heading index.
        """
        if not self.retriever:
            raise RuntimeError("No heading index loaded. Call load_heading_index() first.")
        return self.retriever.get_relevant_documents(query_text)[:k]

    def query_with_scores(self, query_text: str, k: int = 4):
        """
        Retrieve top-k relevant documents with similarity scores from the currently loaded heading index.
        """
        if not self.vectorstore:
            raise RuntimeError("No heading index loaded. Call load_heading_index() first.")
        return self.vectorstore.similarity_search_with_score(query_text, k=k)
