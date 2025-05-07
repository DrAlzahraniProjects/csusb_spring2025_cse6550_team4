import os
import time
import streamlit as st
import requests
import random
import pandas as pd
from dotenv import load_dotenv
import json
# ─── Scrapy core ───
import scrapy
import unicodedata
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from langchain.schema import Document
import logging
import hashlib
import argparse
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import re
from urllib.parse import urlparse
from flashrank import Ranker, RerankRequest
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import Counter

def pick_primary_url(urls: list[str]) -> str | None:
    return urls[0] if urls else None


# --- Add IP Check Functions ---
def get_user_ip() -> str:
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=None)
        return response.json().get("ip", "")
    except Exception:
        return ""

def is_US_ip(ip: str) -> bool:
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}?fields=country", timeout=None)
        return response.json().get("country", "").lower() == "united states"
    except Exception:
        return False

# --- IP Check Block ---
user_ip = get_user_ip()
if not is_US_ip(user_ip):
    st.error(f"Access to this webpage is prohibited.")
    st.stop()

INDEX_DIR = "faiss_index"

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# Instantiate FlashRank
RERANKER = Ranker(max_length=4096)

# Load Environment Variables
load_dotenv()

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='RecWell Chatbot App')
parser.add_argument('--groq_api_key', type=str, help='Groq API Key (optional override)')
try:
    args, _ = parser.parse_known_args()
except Exception as e:
    logging.error(f"Argument parsing error: {e}")
    args = type('Args', (), {'groq_api_key': None})()

# --- Define and Initialize API Key ---
api_key = os.environ.get("GROQ_API_KEY") or args.groq_api_key
if api_key:
    st.session_state["GROQ_API_KEY"] = api_key

# Initialize embeddings model globally
try:
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

except Exception as e:
    logging.error(f"Error loading embedding model: {e}")
    embedding_function = None

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="CSUSB RecWell Chatbot", page_icon="🏋️‍♂️")

# --- Header & Logo & Subheading ---
st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
col1, col2 = st.columns([2,6], gap="small")
with col1:
    st.image("logo.png", width=500)  # adjust width to taste
with col2:
    st.markdown(
        """
        <h1 style="margin-bottom:0; color:#4A90E2;">
          CSUSB RecWell Chatbot
        </h1>
        
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <h3 style='text-align: center; color: #666; margin-bottom: 1rem;'>
      Your Guide to Fitness, Fun & Adventure at&nbsp;
      <a href="https://www.csusb.edu/recreation-wellness"
         target="_blank"
         style="color:#4A90E2; text-decoration: underline;">
        CSUSB Recreation & Wellness
      </a>
    </h3>
    """,
    unsafe_allow_html=True
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Adjust page header */
    h1 {
        padding-top: 0;
        margin-top: 30px !important;
    }
    
    /* Main container padding */
    .block-container { 
        padding-top: 1rem !important; 
        padding-bottom: 0rem !important; 
    }
    
    /* Hide footer */
    footer { visibility: hidden; }
    
    /* Ensure table text is readable */
    .stTable table { 
        font-size: 14px; 
        text-align: center; 
    }
    
    /* Fix chat input position */
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 20px;
        width: 60%;
        left: 20%;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        padding: 10px;
        z-index: 1000;
    }
    
    /* Make chat container larger */
    .main .block-container {
        max-width: 1200px;
        padding-bottom: 100px;
    }
    
    /* Make chat messages display better */
    [data-testid="stChatMessageContent"] {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# === Helper Functions ===
def normalize_text(text):
    if not isinstance(text, str): text = str(text)
    return unicodedata.normalize("NFKC", text.lower().strip())

def hash_scraped_output(output):
    if not isinstance(output, str): output = json.dumps(output, sort_keys=True)
    return hashlib.sha256(normalize_text(output).encode('utf-8')).hexdigest()

def format_response_time(elapsed: float) -> str:
    if elapsed < 1:
        return f"{elapsed*1000:.0f}ms"
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    return f"{minutes}m {seconds:.1f}s"

# ─── helper functions ───
TAG_RE = re.compile(r'<[^>]+>')
WS_RE  = re.compile(r'\s+')

def clean_text(text: str) -> str:
    no_tags = TAG_RE.sub('', text)
    return WS_RE.sub(' ', no_tags).strip()

def segment_text(text: str, max_chunk_size: int = 512) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=80,
    )
    chunks = splitter.split_text(text)

    seen = set()
    unique_chunks = []
    for c in chunks:
        normalized = normalize_text(c)
        if len(normalized) > 20 and normalized not in seen:
            seen.add(normalized)
            unique_chunks.append(c)
    return unique_chunks

def rerank_results(question, documents, top_n=5):
    """Use FlashRank to rerank a small candidate set."""
    if not documents:
        return []

    # Build (id, text) pairs for FlashRank
    pairs  = [{"id": i, "text": doc.page_content}
              for i, doc in enumerate(documents)]
    ranked = RERANKER.rerank(RerankRequest(question, pairs))
    top_ids = [item["id"] for item in ranked[:top_n]]
    return [documents[i] for i in top_ids]

# ─── RecWellSpider in place of ContentSpider ───
class RecWellSpider(scrapy.Spider):
    name = "recwell"
    allowed_domains = ["csusb.edu"]
    start_urls = ["https://www.csusb.edu/recreation-wellness"]

    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1,
        "AUTOTHROTTLE_MAX_DELAY": 3,
        "LOG_LEVEL": "INFO",
    }

    def parse(self, response):
        self.logger.info(f"→ Scraping: {response.url}")
        url         = response.url
        title       = response.xpath("//title/text()").get(default="").strip()
        meta_desc   = response.css('meta[name="description"]::attr(content)').get(default="").strip()
        json_ld     = response.xpath("//script[@type='application/ld+json']/text()").getall()

        # 1) Extract and clean all body text, then chunk
        raw_nodes   = response.xpath("//body//text()[normalize-space()]").getall()
        joined      = " ".join(n.strip() for n in raw_nodes if n.strip())
        cleaned     = clean_text(joined)
        segments    = segment_text(cleaned, max_chunk_size=512)

        # 2) Collect internal csusb.edu links
        links = []
        for href in response.css("a::attr(href)").getall():
            full = response.urljoin(href)
            parsed = urlparse(full)
            if parsed.hostname and parsed.hostname.endswith("csusb.edu") \
               and parsed.path.startswith("/recreation-wellness"):
                links.append(full)

        yield {
            "url":            url,
            "title":          title,
            "meta_desc":      meta_desc,
            "json_ld":        json_ld,
            "cleaned_text":   cleaned,
            "segments":       segments,
            "internal_links": list(set(links)),
        }

        # 3) Follow internal links
        for link in set(links):
            yield scrapy.Request(link, callback=self.parse)
        
# === Update-Aware Scrapy Runner ===
def run_scrapy_if_changed():
    temp_file  = "scraped_data_temp.json"
    final_file = "scraped_data.json"

    settings = get_project_settings()
    settings.set("FEED_FORMAT", "json")
    settings.set("FEED_URI", temp_file)

    if os.path.exists(temp_file):
        os.remove(temp_file)

    try:
        process = CrawlerProcess(settings)
        process.crawl(RecWellSpider)
        process.start(stop_after_crawl=True)
    except Exception as e:
        if hasattr(st, "error"):
            st.error(f"Web scraping error: {e}")
        return False

    if not os.path.exists(temp_file):
        return False

    with open(temp_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content or content == "[]":
        os.remove(temp_file)
        return False

    new_hash = hash_scraped_output(content)
    old_hash = None
    if os.path.exists(final_file):
        with open(final_file, "r", encoding="utf-8") as f:
            old_hash = hash_scraped_output(f.read())
    if new_hash != old_hash:
        os.replace(temp_file, final_file)
        if hasattr(st, "success"):
            st.success("✅ Knowledge base updated.")
        return True
    else:
        os.remove(temp_file)
        if hasattr(st, "info"):
            st.info("🔄 Knowledge base is up-to-date.")
        return False

# === Data Loading Function ===
def load_scraped_data(file_path="scraped_data.json"):
    if not os.path.exists(file_path):
        st.warning("Knowledge file missing.")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Data not in list format")
            
        docs = []
        seen = set()

        for item in data:
            # now we look at the spider’s "segments" list instead of "text"
            for seg in item.get("segments", []):
                txt = normalize_text(seg)
 
                if not txt or len(txt) <= 20 or txt in seen:
                    continue
                meta = {
                   "url": item.get("url", ""),
                   "title": item.get("title", ""),
                   "source": "scraped"
                }
                docs.append(Document(page_content=txt, metadata=meta))
                seen.add(txt)

                
        if not docs:
            raise ValueError("No valid docs")
        return docs

    except Exception as e:
        st.error("Load error.")
        return []

# === Core Response Logic ===
SYSTEM_PROMPT = """You are the official AI assistant for CSUSB Recreation & Wellness.

Use the following context to answer the user's question clearly and professionally. Do not mention documents, sources, or context in your answer. Simply respond with accurate and helpful information as if you know it directly.

Guidelines:
- Do NOT say things like “based on the context,” “the document says,” or “Doc 1 mentions.”
- Just answer naturally, like a well-informed staff member.
- If asked about hours or schedules, list them clearly in plain sentences (e.g., “The Aquatic Center is open from...”).
- Only use contact info if it is explicitly in the context.
- If you don’t have enough info, say: “I don’t have enough information to answer that.”
- Never make up details.
- Unless the user specifically mentions “Palm Desert” or “PDC,” assume the question is about the main CSUSB campus only. Do not include information about Palm Desert unless asked.
Context:
{context}

Question:
{user_question}

Answer:
"""

def retrieve_relevant_docs(query, k=10) -> tuple[str, list[str]]:
    vs = st.session_state.get("vectorstore")
    if not vs:
        return "", []

    try:
        # first pass: grab a lot more chunks
        fetch_k = max(4*k, 100)
        candidates = vs.similarity_search(query, k=fetch_k)

        # fallback: if no hits, try a broader search 
        if not candidates:
            candidates = vs.similarity_search(query, k=200)
            if candidates:
                logging.info("Fallback broader search returned results.")

        if not candidates:
            return "", []

        # rerank and take top k
        docs = rerank_results(query, candidates, top_n=k)

        # build the context string
        context = ""
        MAX_LEN = 800
        for doc in docs:
            context += f"{doc.page_content[:MAX_LEN]}\n\n"

        # collect the URLs alongside the text
        urls = [doc.metadata.get("url", "") for doc in docs]
        return context.strip(), urls

    except Exception as e:
        return f"Error retrieving documents: {e}", []


def rewrite_query_with_groq(original_query: str, api_key: str) -> str:
    prompt = f"""Rewrite the following user question so that it matches the kind of language used on a college recreation and wellness website. Be clear and keyword-rich.

User question:
{original_query}

Rewritten query:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 100
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=10
        )
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.warning(f"Query rewrite failed: {e}")
        return original_query

def get_response(user_input):
    start = time.perf_counter()

    # ────────────────────────────────────────────────────────────────────────────
    # 0) Block any questions about other CSUSB departments
    disallowed = [
        "housing",
        "residence",
        "health center",
        "student health",
        "financial aid",
        "admissions",
        # …add more as needed…
    ]
    if any(term in user_input.lower() for term in disallowed):
        elapsed = time.perf_counter() - start
        return (
            "I’m sorry, but I can only answer questions about CSUSB Recreation & Wellness. "
            "Please ask me something from the RecWell website.",
            0.0,
            format_response_time(elapsed),
            [] 
        )



    # 1) Missing-key early exit (now with real timing)
    groq_key = st.session_state.get("GROQ_API_KEY")
    if not groq_key:
        elapsed = time.perf_counter() - start
        return (
            "API Key needed for detailed responses. Please enter your Groq API key in the sidebar.",
            0,
            format_response_time(elapsed),
            []
        )

    # 2) Gather context
    rewritten_query = rewrite_query_with_groq(user_input, groq_key)
    context, source_urls = retrieve_relevant_docs(rewritten_query, k=10)

    # If no RecWell docs were found, refuse to answer
    if isinstance(context, str) and "No relevant documents found" in context:
        elapsed = time.perf_counter() - start
        return (
            "I’m sorry, but I can only answer questions about CSUSB Recreation & Wellness. "
            "Please ask me something about the RecWell website.",
            0.0,
            format_response_time(elapsed),
            []
        )
    # ───────────────────────────────────────────────────────────────────────────

    # 3) Build headers & messages
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    msgs = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Context:\n{context}\n\nQ: {user_input}"}
    ]

    try:
        # 4) Do the API call
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama3-8b-8192",
                "messages": msgs,
                "temperature": 0.7,
                "max_tokens": 500
            },
            headers=headers,
            timeout=30
        )

        # 5) Handle non-200 errors
        if resp.status_code != 200:
            error_msg = f"API Error (Status {resp.status_code})"
            try:
                err = resp.json().get("error", {})
                error_msg += f": {err.get('message','Unknown error')}"
            except:
                error_msg += ": Could not parse error response"
            elapsed = time.perf_counter() - start
            return (
                f"Sorry, I encountered an issue: {error_msg}. Please try again later.",
                0,
                format_response_time(elapsed),
                []
            )

        # 6) Parse good response
        content = resp.json()["choices"][0]["message"]["content"]
        confidence = 0.8

        # 7) Final timing
        elapsed = time.perf_counter() - start
        return content, confidence, format_response_time(elapsed), source_urls

    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - start
        return (
            "I'm sorry, but I'm having trouble connecting to my knowledge base right now. Please try again in a moment.",
            0.3,
            format_response_time(elapsed),
            []
        )
    except requests.exceptions.RequestException:
        elapsed = time.perf_counter() - start
        return (
            "I'm having trouble processing your question. Please try again. (Error: Network issue)",
            0.3,
            format_response_time(elapsed),
            []
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return (
            f"I'm having trouble processing your question. Please try again. (Error: {str(e)[:100]})",
            0.3,
            format_response_time(elapsed),
            [] 
        )

# === Send Message Function ===
def send_message(user_input=None):
    if user_input is None:
        user_input = st.session_state.get("user_input", "").strip()
    if not user_input:
        return None, 0

    try:
        response_text, confidence, response_time_str, source_urls = get_response(user_input)

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": response_text,
            # only attach sources if we flagged it as RecWell
            "sources": source_urls 
        })

        if "user_input" in st.session_state:
            st.session_state.user_input = ""

        return response_text, confidence
    except Exception as e:
        error_message = f"I'm sorry, but something went wrong while processing your message. Please try again. (Error: {str(e)[:50]})"
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"{error_message}\n\nResponse time: <error>"
        })
        
        return error_message, 0

# === Initialization ===
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []	

# ─── Ensure knowledge base exists ───
def ensure_knowledge_base():
    json_exists  = os.path.exists("scraped_data.json")
    faiss_exists = os.path.isdir(INDEX_DIR)
    # Scrape if JSON is missing OR FAISS index folder is missing
    if not json_exists or not faiss_exists:
        run_scrapy_if_changed()

# Call it once before we load/build FAISS
ensure_knowledge_base()
emb = embedding_function

if os.path.isdir(INDEX_DIR):
    try:
        idx = FAISS.load_local(
            INDEX_DIR,
            emb,
            allow_dangerous_deserialization=True
        )
        st.session_state.vectorstore = idx
    except Exception as e:
        logging.error(f"Failed to load FAISS index, rebuilding: {e}")
        docs = load_scraped_data()
        if docs and emb:
            idx = FAISS.from_documents(documents=docs, embedding=emb)
            idx.save_local(INDEX_DIR)
            st.session_state.vectorstore = idx
            st.success("⚙️ Rebuilt FAISS index after load failure.")
        else:
            st.session_state.vectorstore = None
            st.error("❌ Could not initialize FAISS index.")
else:
    with st.spinner("📚 Loading knowledge base and creating embeddings..."):
        docs = load_scraped_data()
        if docs and emb:
            idx = FAISS.from_documents(documents=docs, embedding=emb)
            idx.save_local(INDEX_DIR)
            st.session_state.vectorstore = idx
            st.success(f"Knowledge base loaded into FAISS ({len(docs)} docs).")
        else:
            st.session_state.vectorstore = None
            if not docs:
                st.error("Failed to load any documents.")
            if not emb:
                st.error("Failed to initialize embedding model.")


# === Main Chat UI ===
chat_container = st.container()
with chat_container:
    st.subheader("💬 Chatbot")
    
    # Chat display
    for message in st.session_state.chat_history:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])
        
            # Only show the first URL, as full text
            if message["role"] == "assistant" and message.get("sources"):
                primary = pick_primary_url(message["sources"])
                if primary:
                    st.markdown(f"🔗 Source: [{primary}]({primary})")

# Chat input
user_input_val = st.chat_input("Type your message here...")
if user_input_val:
    with st.spinner("Thinking..."):
        send_message(user_input_val)
    st.rerun()
