import json
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from config import Config


class DocumentManager:
    """Handles document loading and processing for search"""

    @classmethod
    def load_documents(cls) -> List[Document]:
        """Load and process JSON metadata files"""
        documents = []
        for json_path in Config.OUTPUT_DIR.glob("*.json"):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Combine key fields along with metadata to preserve context.
                combined_text = (
                    f"Summary: {data.get('summary', '')}\n"
                    f"Keywords: {data.get('keywords', '')}\n"
                    f"TimeStamp: {data.get('timestamp_prompt', '')}\n"
                )
                documents.append(Document(
                    page_content=combined_text,
                    metadata={
                        "video_id": data.get("video_id", ""),
                        "filename": data.get("filename", "")
                    }
                ))
        print("Loaded documents into FAISS:", documents)
        return documents

    @classmethod
    def create_vector_store(cls) -> FAISS:
        """Create FAISS vector store from documents using semantic splitting"""
        documents = cls.load_documents()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

        faiss_db = FAISS.from_documents(split_docs, embeddings)
        faiss_db.save_local("faiss_index_folder")
        return faiss_db

    @classmethod
    def load_vector_store(cls) -> FAISS:
        """Load FAISS vector store from local files"""
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        faiss_db = FAISS.load_local(
            "faiss_index_folder", embeddings, allow_dangerous_deserialization=True)
        return faiss_db
