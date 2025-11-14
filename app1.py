# streamlit_app.py
"""
Streamlit UI for URL-scoped Q&A (Playwright + Hybrid Retrieval + Grok)
Key changes:
 - Question box disabled until you ingest URLs
 - Suppress noisy Playwright stderr output
 - Show errors from Groq / scraping instead of failing silently
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Hide TF logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import re
import io
import sys
import json
import math
import contextlib
import warnings
warnings.filterwarnings("ignore")  
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("playwright").setLevel(logging.ERROR)

from typing import List, Tuple, Dict, Any

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from whoosh import index as whoosh_index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser
from whoosh.analysis import StemmingAnalyzer

# Playwright (optional)
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------- CONFIG ----------
INDEX_PATH = "faiss.index"
META_PATH = "meta.json"
WHOOSH_DIR = "whoosh_index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HYBRID_TOP_K = 8
HYBRID_ALPHA = 0.6
CHUNK_SUMMARY_MAXTOK = 240
FINAL_SUMMARY_MAXTOK = 600

# ---------- helpers: scraping (quiet playwright) ----------
def scrape_with_playwright_quiet(url: str, timeout: int = 15) -> str:
    """
    Run Playwright while suppressing noisy stderr to keep Streamlit logs clean.
    """
    # capture stderr for Playwright internals
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr_buf):
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page(timeout=timeout * 1000)
                page.goto(url, wait_until="networkidle")
                try:
                    article = page.locator("article")
                    if article.count() > 0:
                        text = "\n".join(article.all_text_contents())
                    else:
                        text = page.locator("body").inner_text()
                except Exception:
                    body_html = page.content()
                    soup = BeautifulSoup(body_html, "html.parser")
                    for t in soup(["script", "style", "nav", "header", "footer", "aside"]):
                        t.decompose()
                    text = soup.get_text(separator="\n")
                browser.close()
    except Exception as e:
        # if Playwright truly fails, return an informative error
        raise RuntimeError(f"Playwright scraping failed: {e}\nPlaywright stderr (truncated): {stderr_buf.getvalue()[:2000]}")
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def scrape_requests(url: str, timeout: int = 10) -> str:
    try:
        headers = {"User-Agent": "MySummaryBot/1.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        main = soup.find("article") or soup.find(attrs={"role": "main"})
        if main:
            content = main.get_text(separator="\n")
        else:
            divs = soup.find_all("div")
            if divs:
                content = max((d.get_text(separator="\n") for d in divs), key=len)
            else:
                content = soup.get_text(separator="\n")
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        filtered = [ln for ln in lines if len(ln) > 40 and not re.search(r'\b(edit|navigation|references|languages|category|cookie)\b', ln, re.I)]
        return "\n".join(filtered) if filtered else content.strip()
    except Exception as e:
        return f"ERROR: {e}"

def scrape_url(url: str) -> str:
    if PLAYWRIGHT_AVAILABLE:
        try:
            return scrape_with_playwright_quiet(url)
        except Exception as e:
            # fallback + show a warning
            st.warning(f"Playwright scrape failed for {url}: {str(e)[:300]} — falling back to requests.")
            return scrape_requests(url)
    else:
        return scrape_requests(url)

# ---------- chunking/embeddings/faiss ----------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text: return []
    chunks=[]
    start=0; L=len(text)
    while start < L:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap if (end-overlap) > start else end
    return chunks

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True); norms[norms==0]=1.0
    emb = emb / norms
    return emb.astype("float32")

def build_faiss_index_from_embeddings(embs: np.ndarray) -> faiss.IndexFlatIP:
    d = embs.shape[1]; idx = faiss.IndexFlatIP(d); idx.add(embs); return idx

def save_index_and_meta(index: faiss.IndexFlatIP, meta: List[Dict]):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def load_index_and_meta():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None
    idx = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta

# ---------- Whoosh lexical ----------
def build_whoosh_index(meta: List[Dict], whoosh_dir: str = WHOOSH_DIR):
    if not os.path.exists(whoosh_dir): os.makedirs(whoosh_dir, exist_ok=True)
    schema = Schema(chunk_id=ID(stored=True, unique=True), source=ID(stored=True), content=TEXT(stored=True, analyzer=StemmingAnalyzer()))
    if whoosh_index.exists_in(whoosh_dir):
        for fname in os.listdir(whoosh_dir):
            try: os.remove(os.path.join(whoosh_dir, fname))
            except: pass
    ix = whoosh_index.create_in(whoosh_dir, schema)
    writer = ix.writer()
    for m in meta:
        writer.add_document(chunk_id=m["id"], source=m["source"], content=m["text"][:10000])
    writer.commit()

def whoosh_search(query: str, whoosh_dir: str = WHOOSH_DIR, top_n: int = 50):
    if not whoosh_index.exists_in(whoosh_dir):
        return []
    ix = whoosh_index.open_dir(whoosh_dir)
    parser = MultifieldParser(["content"], schema=ix.schema)
    q = parser.parse(query)
    out=[]
    with ix.searcher() as s:
        hits = s.search(q, limit=top_n)
        for h in hits:
            out.append((h["chunk_id"], float(h.score)))
    return out

# ---------- semantic + hybrid ----------
def semantic_search_faiss(query: str, model, faiss_index, meta: List[Dict], top_n: int = 50):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_emb = q_emb.astype("float32")
    D,I = faiss_index.search(q_emb, top_n)
    results=[]
    for idx, score in zip(I[0], D[0]):
        if 0 <= idx < len(meta):
            results.append((meta[idx]["id"], float(score)))
    return results

def normalize_scores(scores: List[Tuple[Any, float]]):
    if not scores: return {}
    vals = np.array([s for _,s in scores], dtype=float)
    min_v, max_v = float(vals.min()), float(vals.max())
    out={}
    if math.isclose(max_v, min_v):
        for k,_ in scores: out[k]=1.0
        return out
    for k,v in scores:
        out[k] = float((v - min_v) / (max_v - min_v))
    return out

def hybrid_rank(query: str, model, faiss_index, meta, whoosh_dir=WHOOSH_DIR, top_n_sem=50, top_n_lex=50, final_k:int=HYBRID_TOP_K, alpha:float=HYBRID_ALPHA):
    sem = semantic_search_faiss(query, model, faiss_index, meta, top_n_sem)
    lex = whoosh_search(query, whoosh_dir, top_n_lex)
    sem_norm = normalize_scores(sem); lex_norm = normalize_scores(lex)
    candidate_ids = set(list(sem_norm.keys()) + list(lex_norm.keys()))
    combined=[]
    for cid in candidate_ids:
        s = sem_norm.get(cid,0.0); l = lex_norm.get(cid,0.0)
        combined.append((cid, alpha*s + (1-alpha)*l))
    combined.sort(key=lambda x: x[1], reverse=True)
    top = combined[:final_k]
    id_to_meta = {m["id"]: m for m in meta}
    results=[]
    for cid, sc in top:
        if cid in id_to_meta:
            results.append((id_to_meta[cid], sc))
    return results

# ---------- Groq (with error capture) ----------
def call_grok(prompt: str, max_tokens: int = 512) -> str:
    try:
        from groq import Groq
    except Exception as e:
        raise RuntimeError("groq SDK not installed. Install with: python -m pip install groq") from e
    try:
        client = Groq(api_key=GROQ_API_KEY)
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        chat = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model=model_name, max_tokens=max_tokens)
        try:
            return chat.choices[0].message.content
        except Exception:
            if hasattr(chat, "text"): return chat.text
            if isinstance(chat, dict) and "text" in chat: return chat["text"]
            return str(chat)
    except Exception as e:
        # surface a helpful error message
        raise RuntimeError(f"Groq call failed: {e}")

# ---------- ingestion (fresh, overwrite) ----------
def ingest_urls_fresh(urls: List[str], model: SentenceTransformer):
    all_chunks=[]; meta=[]
    for url in urls:
        st.info(f"Scraping {url} ...")
        text = scrape_url(url)
        if text.startswith("ERROR:"):
            st.error(f"Failed to scrape {url}: {text}")
            continue
        chunks = chunk_text(text)
        for i,ch in enumerate(chunks):
            if len(ch.strip()) < 120: continue
            cid = f"{url}__chunk_{i}"
            all_chunks.append(ch); meta.append({"id":cid,"source":url,"text":ch})
    if not all_chunks:
        st.error("No content ingested from provided URLs.")
        return None, None
    st.info(f"Embedding {len(all_chunks)} chunks ...")
    embs = embed_texts(model, all_chunks)
    st.info("Building FAISS index ...")
    idx = build_faiss_index_from_embeddings(embs)
    save_index_and_meta(idx, meta)
    st.info("Building Whoosh index ...")
    build_whoosh_index(meta)
    st.success("Ingest complete. Index built for provided URL(s).")
    return idx, meta

def strip_chunk_suffix(s: str) -> str:
    return s.split("__chunk_")[0] if "__chunk_" in s else s

# ---------- Streamlit UI ----------
st.set_page_config(page_title="URL-scoped Q&A", layout="wide")
st.title("URL-scoped Q&A (Playwright + Hybrid Retrieval + Grok)")

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("**Enter one or more URLs (comma-separated).** The app will build a fresh index from these URLs and answer questions using only their content.")
    urls_input = st.text_area("URLs", value="", height=90, placeholder="https://example.com/article1, https://example.com/article2")
    ingest_btn = st.button("Ingest (build index)")

with col2:
    st.markdown("**Options**")
    st.checkbox("Use Playwright (if installed)", value=PLAYWRIGHT_AVAILABLE, disabled=True)
    st.write(f"Playwright available: {PLAYWRIGHT_AVAILABLE}")
    st.write("Grok API Key set:" , bool(GROQ_API_KEY))

# session-state for model/index/meta
if "model" not in st.session_state:
    st.session_state.model = None
if "index" not in st.session_state:
    st.session_state.index = None
if "meta" not in st.session_state:
    st.session_state.meta = None
if "ingested_urls" not in st.session_state:
    st.session_state.ingested_urls = None

# load existing index automatically (optional)
if st.session_state.index is None:
    idx_disk, meta_disk = load_index_and_meta()
    if idx_disk is not None and meta_disk is not None:
        st.session_state.index = idx_disk
        st.session_state.meta = meta_disk
        if st.session_state.model is None:
            st.session_state.model = SentenceTransformer(EMBED_MODEL_NAME)
        st.success(f"Loaded existing index with {st.session_state.index.ntotal} vectors. You can overwrite by ingesting new URLs.")

# handle ingest
if ingest_btn:
    if not urls_input.strip():
        st.warning("Please enter at least one URL to ingest.")
    else:
        urls = [u.strip() for u in urls_input.split(",") if u.strip()]
        with st.spinner("Loading embedding model (may take a moment)..."):
            st.session_state.model = SentenceTransformer(EMBED_MODEL_NAME)
        with st.spinner("Ingesting URLs and building index... This can take time depending on page size and network."):
            try:
                idx, meta = ingest_urls_fresh(urls, st.session_state.model)
                if idx is not None:
                    st.session_state.index = idx
                    st.session_state.meta = meta
                    st.session_state.ingested_urls = urls
            except Exception as e:
                st.error(f"Ingest failed: {e}")

st.write("---")
st.header("Ask a question (answers use only the ingested URL(s))")

# Only show the question box if an index exists in session state
if st.session_state.index is None:
    st.info("No index available. Ingest URL(s) first to enable the question box.")
    st.stop()

# question UI (visible only after ingest / index loaded)
q = st.text_input("Your question", value="", placeholder="e.g. Provide a summary of the page")
ask_btn = st.button("Ask")

if ask_btn:
    if not q.strip():
        st.warning("Please enter a question.")
    else:
        model = st.session_state.model
        idx = st.session_state.index
        meta = st.session_state.meta

        with st.spinner("Retrieving relevant chunks..."):
            candidates = hybrid_rank(q, model, idx, meta, final_k=HYBRID_TOP_K, alpha=HYBRID_ALPHA)

        if not candidates:
            st.info("I cannot answer this based on the provided web content.")
        else:
            chunks = [c[0]["text"] for c in candidates]
            sources = []
            for c,_ in candidates:
                s = strip_chunk_suffix(c["source"])
                if s not in sources: sources.append(s)

            try:
                with st.spinner("Generating answer from Grok..."):
                    if len(chunks) == 1 and re.search(r'\bsummary|summarize|overview\b', q, re.I):
                        prompt = f'Using ONLY the CONTEXT below, provide a clear 3-6 sentence summary. If not present, reply exactly: "I cannot answer this based on the provided web content."\n\nCONTEXT:\n{chunks[0]}'
                        out = call_grok(prompt, max_tokens=FINAL_SUMMARY_MAXTOK)
                    elif len(chunks) == 1:
                        prompt = f"Using ONLY the CONTEXT below, answer the question concisely.\n\nCONTEXT:\n{chunks[0]}\n\nQUESTION:\n{q}"
                        out = call_grok(prompt, max_tokens=400)
                    else:
                        chunk_summaries = []
                        for ch in chunks:
                            p = f"Summarize the following text in 2 concise bullet points and a 1-sentence TL;DR. Use only the text provided.\n\n{ch[:4000]}"
                            chunk_summaries.append(call_grok(p, max_tokens=CHUNK_SUMMARY_MAXTOK))
                        if re.search(r'\bsummary|summarize|overview\b', q, re.I):
                            synth_prompt = f"""
You are an assistant. Using ONLY the chunk summaries below, produce:
1) A one-paragraph summary (4-7 sentences).
2) Up to 6 short bullet points of key facts.
If the information is missing, reply exactly: "I cannot answer this based on the provided web content."

Chunk summaries:
{chr(10).join(chunk_summaries)}
"""
                            out = call_grok(synth_prompt, max_tokens=FINAL_SUMMARY_MAXTOK)
                        else:
                            synth_prompt = f"""
You are an assistant. Using ONLY the chunk summaries below, answer the question exactly and concisely. If not present, reply exactly: "I cannot answer this based on the provided web content."

QUESTION:
{q}

CHUNK SUMMARIES:
{chr(10).join(chunk_summaries)}
"""
                            out = call_grok(synth_prompt, max_tokens=400)
            except Exception as e:
                st.error(f"LLM/Grok error: {e}")
                out = None

            if out:
                if "Sources:" not in out:
                    out = out.strip() + "\n\nSources:\n" + "\n".join(sources)
                st.subheader("Answer")
                st.write(out)
                with st.expander("Provenance (top chunks)"):
                    for rank, (m, score) in enumerate(candidates, start=1):
                        st.markdown(f"**Chunk {rank} — score {score:.4f}**")
                        st.write(m["text"][:1000])
                        st.write(f"Source: {strip_chunk_suffix(m['source'])}")

# end of file
