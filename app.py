import os
import time
import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
import random
import json
import scrapy
import unicodedata
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from langchain.schema import Document
import logging
import hashlib
import argparse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Environment Variables
load_dotenv()

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='RecWell Chatbot App')
parser.add_argument('--groq_api_key', type=str, help='Groq API Key (optional override)')
args, _ = parser.parse_known_args()

# --- Define api_key Globally (AFTER parsing args) ---
api_key = os.environ.get("GROQ_API_KEY") or args.groq_api_key

# Initialize embeddings model globally
try:
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logging.info("Embedding model loaded successfully")
except Exception as e:
    logging.error(f"Error loading embedding model: {e}")
    embedding_function = None

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="CSUSB RecWell Chatbot", page_icon="🚀")

# --- Single line title with adjusted position ---
st.markdown("<div style='height: 70px;'></div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #4A90E2; margin-top: 0;'>🐺 CSUSB Team 4 Chatbot</h1>", unsafe_allow_html=True)

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

def format_response_time(start_time):
    end_time = time.time(); response_time = end_time - start_time
    if response_time < 1: return f"{response_time*1000:.0f}ms"
    if response_time < 60: return f"{response_time:.1f}s"
    else: minutes = int(response_time // 60); seconds = response_time % 60; return f"{minutes}m {seconds:.1f}s"

# === Scrapy Spider Definition ===
class ContentSpider(scrapy.Spider):
    name = "content"
    allowed_domains = ["csusb.edu"]
    start_urls = [
        "https://www.csusb.edu/recreation-wellness",
        "https://www.csusb.edu/recreation-wellness/about-us",
        "https://www.csusb.edu/recreation-wellness/memberships",
        "https://www.csusb.edu/recreation-wellness/programs",
        "https://www.csusb.edu/recreation-wellness/facilities"
    ]
    custom_settings = {'DEPTH_LIMIT': 1}
    
    def parse(self, response):
        page_title = response.css('title::text').get()
        url = response.url
        logging.info(f"Parsing: {url}")
        
        # Paragraphs
        for paragraph in response.css("p"):
            text = paragraph.get()
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip():
                yield {
                    "url": url,
                    "title": page_title,
                    "text": clean_text.strip()
                }
        
        # Headers
        for header in response.css("h1, h2, h3, h4, h5, h6"):
            text = header.get()
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip():
                yield {
                    "url": url,
                    "title": page_title,
                    "text": f"Header: {clean_text.strip()}"
                }
        
        # List Items
        for list_item in response.css("li"):
            text = list_item.get()
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip():
                yield {
                    "url": url,
                    "title": page_title,
                    "text": f"List item: {clean_text.strip()}"
                }
        logging.info(f"Finished parsing: {url}")

# === Update-Aware Scrapy Runner ===
def run_scrapy_if_changed():
    logging.info("Checking if scraping is needed...")
    try:
        is_streamlit = st._is_running
    except AttributeError:
        is_streamlit = True
    
    scraped_temp_file = "scraped_data_temp.json"
    scraped_final_file = "scraped_data.json"
    settings = get_project_settings()
    settings.set("FEED_FORMAT", "json")
    settings.set("FEED_URI", scraped_temp_file)
    settings.set("LOG_LEVEL", "WARNING")

    if os.path.exists(scraped_final_file):
        logging.info("Using existing scraped_data.json file")
        return False

    if os.path.exists(scraped_temp_file):
        try:
            os.remove(scraped_temp_file)
            logging.info(f"Removed old temp file: {scraped_temp_file}")
        except OSError as e:
            logging.warning(f"Could not remove old temp file {scraped_temp_file}: {e}")

    try:
        logging.info("Starting Scrapy process...")
        process = CrawlerProcess(settings)
        process.crawl(ContentSpider)
        process.start(stop_after_crawl=True)
        logging.info("Scrapy process finished.")
    except Exception as e:
        logging.error(f"Scrapy process error: {str(e)}", exc_info=True)
        if is_streamlit:
            st.error(f"Web scraping error: {str(e)}")
        return False

    if os.path.exists(scraped_temp_file):
        try:
            with open(scraped_temp_file, "r", encoding="utf-8") as f:
                temp_data_content = f.read()
            if not temp_data_content or not temp_data_content.strip() or temp_data_content == '[]':
                logging.warning("Scraping empty.")
                os.remove(scraped_temp_file)
                return False
            
            json.loads(temp_data_content)
            temp_hash = hash_scraped_output(temp_data_content)
            logging.info(f"Scraped hash: {temp_hash}")
            
            update_needed = False
            if os.path.exists(scraped_final_file):
                try:
                    with open(scraped_final_file, "r", encoding="utf-8") as f:
                        old_data_content = f.read()
                    old_hash = hash_scraped_output(old_data_content) if old_data_content else None
                    logging.info(f"Existing hash: {old_hash}")
                    update_needed = (temp_hash != old_hash)
                except Exception as hash_e:
                    logging.warning(f"Hashing old file failed: {hash_e}")
                    update_needed = True
            else:
                update_needed = True

            if update_needed:
                logging.info("Data changed. Updating.")
                os.replace(scraped_temp_file, scraped_final_file)
                if is_streamlit:
                    st.success("✅ Knowledge base updated.")
                return True
            else:
                logging.info("No change detected.")
                os.remove(scraped_temp_file)
                if is_streamlit:
                    st.info("🔄 Knowledge base is up-to-date.")
                return False
                
        except json.JSONDecodeError as e:
            logging.error(f"Decode error: {e}")
            if os.path.exists(scraped_temp_file):
                try:
                    os.remove(scraped_temp_file)
                    logging.info("Removed corrupted temp file")
                except OSError as re:
                    logging.warning(f"Could not remove corrupted temp file: {re}")
            return False
            
        except Exception as e:
            logging.error(f"Processing error: {e}", exc_info=True)
            if os.path.exists(scraped_temp_file):
                try:
                    os.remove(scraped_temp_file)
                    logging.info("Removed temp file after error")
                except OSError as re:
                    logging.warning(f"Could not remove temp file after error: {re}")
            return False
    else:
        logging.warning("Scraping failed: No output file.")
        return False

# === Data Loading Function ===
def load_scraped_data(file_path="scraped_data.json"):
    logging.info(f"Loading: {file_path}")
    if not os.path.exists(file_path):
        logging.warning(f"Not found: {file_path}")
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
            if not isinstance(item, dict) or "text" not in item:
                continue
            txt = normalize_text(item.get("text", ""))
            if txt and len(txt) > 20 and txt not in seen:
                meta = {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "source": "scraped"
                }
                docs.append(Document(page_content=txt, metadata=meta))
                seen.add(txt)
                
        if not docs:
            raise ValueError("No valid docs")
        logging.info(f"Loaded {len(docs)} docs.")
        return docs
    except Exception as e:
        logging.error(f"Load error {file_path}: {e}", exc_info=True)
        st.error("Load error.")
        return []

# === Core Response Logic ===
SYSTEM_PROMPT = """You are Beta, an assistant for the CSUSB RecWell Center. Answer questions about the CSUSB Recreation and Wellness Center based on the context provided. Keep responses concise and focus on recreation and wellness topics. If you're unsure about something, politely explain that you don't have that information and suggest the user contact the RecWell Center directly."""

def retrieve_relevant_docs(query, k=5):
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None:
        return "Vectorstore not available."
        
    try:
        start_time = time.time()
        docs = st.session_state.vectorstore.similarity_search(query, k=k)
        logging.info(f"Document retrieval took: {time.time() - start_time:.2f}s")
        
        if not docs:
            return "No relevant documents found."
            
        context = ""
        MAX_LEN = 800
        for i, doc in enumerate(docs, 1):
            src = f"(Source: {doc.metadata.get('url', '?')})" if doc.metadata.get("url") else ""
            context += f"Doc {i}: {doc.page_content[:MAX_LEN]} {src}\n\n"
            
        return context.strip()
    except Exception as e:
        logging.error(f"Document retrieval error: {e}", exc_info=True)
        return f"Error retrieving documents: {str(e)}"

def get_response(user_input):
    start_time = time.time()
    context = retrieve_relevant_docs(user_input)
    
    groq_key = st.session_state.get("GROQ_API_KEY")
    if not groq_key:
        return "API Key needed for detailed responses. Please enter your Groq API key in the sidebar.", 0, format_response_time(start_time)
    
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQ: {user_input}"}
    ]
    
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={"model": "llama3-8b-8192", "messages": msgs, "temperature": 0.7, "max_tokens": 500},
            headers=headers,
            timeout=30
        )
        
        if resp.status_code != 200:
            error_msg = f"API Error (Status {resp.status_code})"
            try:
                error_data = resp.json()
                if 'error' in error_data:
                    error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
            except:
                error_msg += ": Could not parse error response"
                
            logging.error(error_msg)
            return f"Sorry, I encountered an issue: {error_msg}. Please try again later.", 0, format_response_time(start_time)
            
        api_resp = resp.json()
        content = api_resp["choices"][0]["message"]["content"]
        logging.info("LLM Response received")
        
        return content, 0.8, format_response_time(start_time)
        
    except requests.exceptions.Timeout:
        logging.error("GROQ API request timed out")
        return "I'm sorry, but I'm having trouble connecting to my knowledge base right now. Please try again in a moment.", 0.3, format_response_time(start_time)
        
    except requests.exceptions.RequestException as e:
        logging.error(f"GROQ API request error: {str(e)}", exc_info=True)
        return f"I'm having trouble processing your question. Please try again. (Error: Network issue)", 0.3, format_response_time(start_time)
        
    except Exception as e:
        logging.error(f"LLM Error: {str(e)}", exc_info=True)
        return f"I'm having trouble processing your question. Please try again. (Error: {str(e)[:100]})", 0.3, format_response_time(start_time)

# === Send Message Function ===
def send_message(user_input=None):
    if user_input is None:
        user_input = st.session_state.get("user_input", "").strip()
    if not user_input:
        return None, 0
        
    try:
        response_text, confidence, response_time_str = get_response(user_input)
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"{response_text}\n\nResponse time: {response_time_str}"
        })
        
        if "user_input" in st.session_state:
            st.session_state.user_input = ""
            
        return response_text, confidence
    except Exception as e:
        logging.error(f"Error in send_message: {str(e)}", exc_info=True)
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

# === Conditional Scraper Run ===
@st.cache_data(ttl=86400)
def check_and_run_scraper():
    logging.info("Scraper check...")
    data_file = "scraped_data.json"
    needs_check = True
    if os.path.exists(data_file):
        try:
            if (time.time() - os.path.getmtime(data_file)) < 86400:
                needs_check = False
                logging.info("Data is recent.")
        except Exception as e:
            logging.warning(f"File modification time error: {e}")
            
    if needs_check:
        logging.info("Running scraper...")
        with st.spinner("🔄 Checking updates..."):
            try:
                data_updated = run_scrapy_if_changed()
                return data_updated
            except Exception as e:
                logging.error(f"Scrape run error: {e}", exc_info=True)
                st.error(f"Update check error: {e}")
                return False
    return False

data_updated = check_and_run_scraper()

# === Vectorstore Initialization with FAISS ===
if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None or data_updated:
    logging.info(f"Initializing FAISS vector store. Updated: {data_updated}")
    with st.spinner("📚 Loading knowledge base and creating embeddings..."):
        try:
            docs = load_scraped_data()
            if docs and embedding_function:
                st.session_state.vectorstore = FAISS.from_documents(
                    documents=docs,
                    embedding=embedding_function
                )
                st.success(f"Knowledge base loaded into FAISS ({len(docs)} docs).")
                logging.info("FAISS vector store created successfully.")
            else:
                st.session_state.vectorstore = None
                if not docs:
                    st.error("Failed to load any documents.")
                if not embedding_function:
                    st.error("Failed to initialize embedding model.")
                logging.error("Vector store initialization failed: Missing docs or embedding model.")
        except Exception as e:
            st.error(f"Critical error initializing FAISS vectorstore: {e}")
            logging.critical(f"FAISS vector store initialization failed: {e}", exc_info=True)
            st.session_state.vectorstore = None

# === Sidebar Content ===
with st.sidebar:
    st.header("Configuration")
    
    # API Key configuration
    if not st.session_state.get("GROQ_API_KEY"):
        st.warning("Groq API Key required.")
        provided_key = st.text_input(
            "🔑 Enter Groq API Key:",
            type="password",
            key="api_key_input_sidebar",
            help="Get free key from console.groq.com"
        )
        if provided_key:
            st.session_state.GROQ_API_KEY = provided_key
            st.success("API Key entered.")
            time.sleep(1)
            st.rerun()
    
    # Clear Chat History button
    if st.button("Clear Chat History", key="clear_chat_sidebar"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")
        time.sleep(1)
        st.rerun()

# === Main Chat UI ===
chat_container = st.container()
with chat_container:
    st.subheader("💬 Chatbot")
    
    # Chat display
    for message in st.session_state.chat_history:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            try:
                content_parts = message["content"].split("\n\nResponse time:")
                st.write(content_parts[0])
                if len(content_parts) > 1:
                    st.caption(f"⏱️ Response time: {content_parts[1].strip()}")
            except Exception as display_e:
                logging.error(f"Display error: {display_e}")
                st.write("Error displaying message.")

# Chat input
user_input_val = st.chat_input("Type your message here...")
if user_input_val:
    with st.spinner("Thinking..."):
        send_message(user_input_val)
    st.rerun()
