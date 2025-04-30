from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from reranking import rerank_documents

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333/dashboard"

client = QdrantClient(url=url, prefer_grpc=False)

db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="healthcare",
)

query = "what causes of glossophobia?"
results = db.similarity_search(query, k=5)

for i, result in enumerate(results, start=1):
    print("=" * 60)
    print(f"📄 Hasil {i}")
    print("-" * 60)
    print("Konten:")
    print(result.page_content)
    print("\nMetadata:")
    print(result.metadata)
    print("=" * 60 + "\n")

reranking = rerank_documents(query=query, documents=results)

for i, result in enumerate(reranking, start=1):
    print("=" * 60)
    print(f"📄 Hasil {i}")
    print("-" * 60)
    print("Konten:")
    print(result.page_content)
    print("\nMetadata:")
    print(result.metadata)
    print("=" * 60 + "\n")