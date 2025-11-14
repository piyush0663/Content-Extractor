# Web Content Q-A Tool

A scalable Retrieval-Augmented Generation (RAG) pipeline for automated web content extraction and question answering.  
Uses **Playwright** for dynamic web scraping, **LangChain** for chunking & retrieval, and **Streamlit** for the UI. The Streamlit app entrypoint is `app1.py`.

---

## Features
- Dynamic web scraping with Playwright (handles JS-heavy pages)
- Text chunking, embedding, and retrieval via LangChain
- Configurable vectorstore (FAISS / Chroma / etc.)
- Streamlit UI for interactive question answering
- Easy local and Docker-based deployment

---

## Repository Layout (suggested)
