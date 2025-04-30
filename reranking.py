from langchain.schema import Document
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query, documents, llm=reranker) -> list[Document]:
    """
    Rerank documents based on their relevance to the query using LLM.
    """
    pairs = []
    for doc in documents:
        pairs.append((query, doc.page_content))
        
    score = reranker.predict(pairs)
    score_with_document = zip(score, documents)
    scored_documents = sorted(score_with_document, key=lambda x: x[0], reverse=True)
    scored_documents = [doc for _, doc in scored_documents]
    best_documents = [scored_documents[0]]
    return best_documents