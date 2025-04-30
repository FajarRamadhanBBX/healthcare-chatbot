import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from preprocessing import filter_documents

## Model embedding yang digunakan
embedding = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

## membanca semua file pdf pada folder "data"
loader = DirectoryLoader(
    path="data/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
)

# Load documents from the directory
documents = loader.load()

# Pre processing documents
documents = filter_documents(documents)

for doc in documents:
    print(doc.page_content)
    print("\n")

# mengatur splitter yang digunakan dan parameternya
text_splitter = SemanticChunker(
    embedding,
    min_chunk_size = 100,
    breakpoint_threshold_type="gradient")

# Melakukan split pada dokumen yang sudah di load
texts = text_splitter.split_documents(documents)

url = "http://localhost:6333"

# Inisialisasi Qdrant untuk menyimpan data
qdrant = Qdrant.from_documents(
    texts,
    embedding,
    url=url,
    prefer_grpc = False,
    collection_name="healthcare",
)