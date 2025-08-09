import json
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import time
import random


def load_chunks(json_path:str):
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

def create_langchain_doc(chunks:dict):
    doc = [
        Document(
            page_content = chunk["content"],
            metadata = {"heading" : chunk["heading"]}
        )
        for chunk in chunks
    ]
    return doc

def create_embeddings(docs: list[Document], output_path="D:/Projects/game-dev-rag/processed/embeddings.json", batch_size=5):
    """
    Create embeddings for documents with retry/backoff on rate limit errors.

    :param docs: List of LangChain Document objects
    :param output_path: Path to save embeddings JSON
    :param batch_size: Number of docs to embed per request
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    embeddings_data = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]

        for attempt in range(5):  # retry up to 5 times
            try:
                batch_embeddings = embedding_model.embed_documents(
                    texts=[doc.page_content for doc in batch],
                    task_type="RETRIEVAL_DOCUMENT",
                    titles=[doc.metadata.get("heading", "") for doc in batch],
                    batch_size=batch_size
                )

                for doc, emb in zip(batch, batch_embeddings):
                    embeddings_data.append({
                        "content": doc.page_content,
                        "heading": doc.metadata.get("heading", ""),
                        "embedding": emb
                    })

                break  # success → break retry loop

            except Exception as e:
                if "429" in str(e):
                    wait_time = (2 ** attempt) + random.random()
                    print(f"429 rate limit hit. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    raise

        print(f"Processed {i + len(batch)}/{len(docs)} docs")

    # Save all embeddings to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

    return f"Embeddings saved to {output_path}"


def store_in_faiss(embeddings_path: str, faiss_dir: str):
    """
    Load precomputed embeddings from JSON and store them in a FAISS index.

    :param embeddings_path: Path to embeddings.json
    :param faiss_dir: Directory path to save the FAISS index
    """
    with open(embeddings_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text_embedding_pairs = [
        (item["content"], item["embedding"])
        for item in data
    ]
    metadatas = [{"heading": item["heading"]} for item in data]

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embedding_model,
        metadatas=metadatas
    )

    vectorstore.save_local(faiss_dir)
    print(f"FAISS index saved to {faiss_dir}")

def inspect_faiss(faiss_dir: str, embedding_model):
    """
    Load a FAISS index and inspect stored documents.

    :param faiss_dir: Directory path to the FAISS index
    :param embedding_model: The embedding model used when creating the index
    """
    vectorstore = FAISS.load_local(
        faiss_dir,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # Count entries
    print(f" Index contains {len(vectorstore.index_to_docstore_id)} documents.\n")

    docs = vectorstore.docstore._dict
    for i, (doc_id, doc) in enumerate(docs.items()):
        print(f"--- Document {i+1} ---")
        print(f"ID: {doc_id}")
        print(f"Heading: {doc.metadata.get('heading')}")
        print(f"Content preview: {doc.page_content[:150]}...\n")
        if i >= 2:
            break
