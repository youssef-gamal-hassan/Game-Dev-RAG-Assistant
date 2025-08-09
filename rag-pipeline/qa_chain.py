import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import RetrieverHandler


class QAChain:
    def __init__(self, faiss_base_dir: str, headings_path: str, score_threshold: float = 0.4):
        """
        :param faiss_base_dir: Directory containing all per-heading FAISS indexes
        :param headings_path: Path to .txt file containing all section headings
        :param score_threshold: Minimum similarity score to keep a document
        """
        load_dotenv()
        self.__api_key = os.getenv("GOOGLE_API_KEY")
        self.score_threshold = score_threshold
        self.faiss_base_dir = faiss_base_dir

        # Load headings from file
        with open(headings_path, "r", encoding="utf-8") as f:
            self.headings = [h.strip() for h in f if h.strip()]

        # Model for heading selection + QA
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=self.__api_key
        )

        # Retriever now supports per-heading FAISS loading
        self.retriever = RetrieverHandler(self.faiss_base_dir)

    def select_heading(self, query: str) -> str:
        """
        Use the LLM to decide the best heading for the query.
        """
        headings_str = "\n".join(self.headings)
        prompt = f"""
            You are assisting in a RAG pipeline.  
            The document is split into sections with the following headings:

            {headings_str}

            Given the user query:
            '{query}'

            Choose the single most relevant heading from the list above.
            Respond with ONLY the heading text, nothing else.
        """
        heading = self.llm.invoke(prompt).content.strip()
        return heading

    def answer_query(self, query: str):
        """
        Full pipeline:
        1. LLM picks heading
        2. Load FAISS index for that heading
        3. Retrieve only from that heading
        4. LLM answers with context + sources
        """
        chosen_heading = self.select_heading(query)
        print(f"Chosen heading: {chosen_heading}")

        # Load the FAISS index for this heading
        try:
            self.retriever.load_heading_index(chosen_heading)
        except FileNotFoundError:
            return {
                "heading": chosen_heading,
                "answer": "No FAISS index found for this heading.",
                "sources": []
            }

        # Retrieve documents with scores
        docs_with_scores = self.retriever.query_with_scores(query_text=query, k=8)

        # Filter by score threshold
        docs_with_scores = [
            (doc, score) for doc, score in docs_with_scores
            if score >= self.score_threshold
        ]

        if not docs_with_scores:
            return {
                "heading": chosen_heading,
                "answer": "No relevant context found above score threshold.",
                "sources": []
            }

        # Prepare context for LLM
        context = "\n\n".join(doc.page_content for doc, _ in docs_with_scores)
        sources = [doc.page_content[:200] for doc, _ in docs_with_scores]

        # Ask model
        prompt = f"""
            Answer the following question based on the provided context and provide steps in which the user can implement any solutions found.  
            If the answer is not in the context, say so.
            
            Question:
            {query}
            
            Context:
            {context}
            
            Answer:
        """
        answer = self.llm.invoke(prompt).content.strip()

        return {
            "heading": chosen_heading,
            "answer": answer,
            "sources": sources
        }


    
