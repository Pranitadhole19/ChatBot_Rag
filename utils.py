from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer

class Encoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L12-v2", cache_folder="models", device="cpu"):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs={"device": device},
        )

class FaissDb:
    def __init__(self, docs, embedding_function):
        self.db = FAISS.from_documents(
            docs, embedding_function, distance_strategy=DistanceStrategy.COSINE
        )

    def similarity_search(self, question: str, k: int = 3):
        return self.db.similarity_search(question, k=k)

def load_and_split_pdfs(file_paths: list, chunk_size: int = 512):
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2"),
        chunk_size=chunk_size,
        chunk_overlap=64,
        strip_whitespace=True,
    )
    return text_splitter.split_documents(pages)
