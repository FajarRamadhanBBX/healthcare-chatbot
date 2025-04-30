# Wicara Chatbot v1
![WICARA cover](https://github.com/user-attachments/assets/61b74f44-5a25-41cf-bb6b-6a735311fe7c)

## Overview

**Wicara** is an application designed to help individuals overcome **glossophobia**—the fear of public speaking—by providing both theoretical lessons and practical training evaluated by AI.

To enhance user experience, this project includes the development of a **chatbot** that can be integrated into the application. The chatbot allows users to directly ask questions related to glossophobia, aiming to improve user interaction and deliver fast, accurate answers.

---

## 🚀 Technologies Used

- **LangChain** – Framework for building LLM-powered applications  
- **Qdrant** – Vector database for efficient semantic search  
- **PubMedBERT** – Embedding model trained on biomedical texts  
- **ms-marco-MiniLM-L-6-v2** – Lightweight cross-encoder for re-ranking  
- **Meditron** – Large Language Model (LLM) specialized in medical domain  
- **FastAPI** – High-performance web framework for API backend  
- **Bootstrap** – Frontend framework for responsive UI

---

## 🔁 Workflow

1. Data preprocessing
2. Semantic chunking of medical text
3. Embedding generation using PubMedBERT
4. Vector storage and retrieval with Qdrant
5. Re-ranking using ms-marco-MiniLM-L-6-v2
6. Response generation using Meditron LLM

---

## ✅ Key Features

- Multi-agent architecture  
- data preprocessing 
- Chunking with semantic search  
- Cross-encoder based re-ranking for better answer relevance

---

## 📉 Current Limitations

- The RAG (Retrieval-Augmented Generation) implementation is still in its early stages. Current results return raw chunks without meaningful transformation into natural answers. This is a critical area for improvement.

---

## 🔧 Future Improvements

- Expanding the medical dataset for broader coverage  
- Experimenting with more robust and capable LLMs for better answer generation
