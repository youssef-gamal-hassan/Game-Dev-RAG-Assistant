import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import tiktoken
import json

class DocumentHandler:
    """
    Handles the ingestion, cleaning, splitting, and storage of PDF documents
    for use in Retrieval-Augmented Generation (RAG) pipelines or other
    natural language processing workflows.
    """
    def __init__(self, pdf_path: str):
        """
        Initialize a DocumentHandler for processing a specific PDF file.
        :param pdf_path: Absolute file path to the PDF document that will be cleaned, split, and prepared for use.
        :type pdf_path: str
        """
        self.pdf_path = pdf_path

    def clean_text(self, skip_pages:int=7):
        """
        Extract and clean text from a PDF file while preserving structure for heading detection.
        :param skip_pages: Number of initial pages to skip (e.g., cover, table of contents).
        :type skip_pages: int
        :return: Cleaned text from the PDF, with headings normalized and unnecessary characters removed.
        :rtype: str
        """
        cleaned_pages = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i < skip_pages:
                    continue

                text = page.extract_text() or ""

                # footers
                text = re.sub(r"©\s*2023\s*Unity\s*Technologies.*?unity\s*\.com", "", text, flags=re.IGNORECASE)
                text = re.sub(r"Unity .*?Guide", "", text, flags=re.IGNORECASE)
                text = re.sub(r"unity\s*\.com", "", text, flags=re.IGNORECASE)

                text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # page numbers
                text = re.sub(r"[•\u2022\u25CF]", "", text)  # bullets

                def _normalize_headings(m):
                    heading = re.sub(r"\s+", "", m.group())  # remove spaces between letters
                    return heading + "\n\n"

                text = re.sub(r"(?:[A-Z]\s+){2,}[A-Z](?=\s|$)", _normalize_headings, text)

                text = re.sub(r"\n{3,}", "\n\n", text)

                lines = []
                for line in text.splitlines():
                    line = re.sub(r" {2,}", " ", line.strip())
                    lines.append(line)

                page_text = "\n".join(lines).strip()
                cleaned_pages.append(page_text)

        cleaned_text = "\n\n".join([p for p in cleaned_pages if p])
        with open("D:/Projects/game-dev-rag/processed/cleaned_unity_guide.txt", "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        return cleaned_text


    @staticmethod
    def _count_tokens(text, model="cl100k_base"):
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text))


    def load_and_split(self, text_path:str="", chunk_size:int=500, chunk_overlap:int=100, token_limit:int=512):
        """
            Split cleaned text into chunks while tracking section headings.

            :param text_path: Path to cleaned text file.
            :type text_path: str
            :param chunk_size: Maximum number of characters per chunk.
            :type chunk_size: int
            :param chunk_overlap: Number of overlapping characters between chunks.
            :type chunk_overlap: int
            :param token_limit: Maximum number of tokens per section before splitting further.
            :type token_limit: int
            :return: A list of dictionaries, each containing 'content' and 'heading' keys.
            :rtype: list[dict]
        """

        if not os.path.exists(text_path):
            with open("D:/Projects/game-dev-rag/processed/cleaned_unity_guide.txt", "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = open(text_path, "r", encoding="utf-8").read()

        def _is_heading_line(s: str) -> bool:
            s = s.strip()
            if not s:
                return False
            if not re.fullmatch(r"[A-Z0-9&\-\:\(\)\s]{3,}", s):
                return False
            if s != s.upper():
                return False
            if len(re.sub(r'[^A-Z0-9]', '', s)) < 3:
                return False
            return True

        lines = text.splitlines()
        sections = []
        current_heading = None
        buffer = []

        for line in lines:
            if _is_heading_line(line):
                if buffer:
                    sections.append({
                        "heading": current_heading,
                        "text": "\n".join(buffer).strip()
                    })
                    buffer = []
                current_heading = line.strip()
            else:
                buffer.append(line)

        if buffer:
            sections.append({
                "heading": current_heading,
                "text": "\n".join(buffer).strip()
            })

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        final_chunks = []
        for sec in sections:
            content = sec["text"]
            if not content:
                continue

            if self._count_tokens(content) > token_limit:
                docs = splitter.create_documents([content])
                for d in docs:
                    final_chunks.append({"content": d.page_content, "heading": sec["heading"]})
            else:
                final_chunks.append({"content": content, "heading": sec["heading"]})

        with open("D:/Projects/game-dev-rag/processed/unity_chunks.json", "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=2)

        return final_chunks
