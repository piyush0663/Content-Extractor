# Web Content Q-A Tool

A scalable Retrieval-Augmented Generation (RAG) pipeline for automated web content extraction and question answering.  
Uses **Playwright** for dynamic web scraping, **LangChain** for chunking & retrieval, and **Streamlit** for the UI. The Streamlit app entrypoint is `app1.py`.

---

<img width="1900" height="776" alt="Screenshot 2025-11-20 181609" src="https://github.com/user-attachments/assets/d652cb63-c5b3-4e8d-bf51-f04928d343c7" />

## Features
- Dynamic web scraping with Playwright (handles JS-heavy pages)
- Text chunking, embedding, and retrieval via LangChain
- Configurable vectorstore (FAISS / Chroma / etc.)
- Streamlit UI for interactive question answering
- Easy local and Docker-based deployment

---

## Repository Layout (suggested)
