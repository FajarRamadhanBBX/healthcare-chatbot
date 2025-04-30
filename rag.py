from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant.vectorstores import Qdrant
from qdrant_client import QdrantClient
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from reranking import rerank_documents
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
local_llm = "static/model/meditron-7b.Q4_K_M.gguf"

# bisa diubah
config = {
    'max_new_tokens': 1024,
    'context_length': 2048,
    'repetition_penalty': 1.1,
    'temperature': 0.1,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model=local_llm,
    model_type="llama",
    **config
)

prompt_template = """You are an experienced psychologist specializing in helping people overcome glossophobia (the fear of public speaking). 
Please answer the following question in a warm, casual, and encouraging tone, using everyday language. Avoid technical or overly academic terms.
Make sure to base your answer **only** on the information provided below. If the context does not contain enough information, simply say "The information provided is not enough to answer this question directly."

Question: {question}
Context: {context}
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url,
    prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="healthcare")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":5})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    retrieved_docs = retriever.get_relevant_documents(query)
    best_doc = rerank_documents(query=query, documents=retrieved_docs)
    best_doc = best_doc[0]
    context = best_doc.page_content

    print("\nContext:\n", context)
    final_prompt = prompt.format(question=query, context=context)
    response = llm(final_prompt)
    print("\nResponse:\n", response)
    response = jsonable_encoder(response)

    return response