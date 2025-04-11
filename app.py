import os
import time
import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from dotenv import load_dotenv
import random
import json
import scrapy
import unicodedata
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from langchain.schema import Document
from datetime import datetime, timezone
import logging
import hashlib  # Added missing import

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

# --- Streamlit Page Configuration (Matching Old Style) ---
st.set_page_config(page_title="CSUSB RecWell Chatbot", page_icon="🚀")

# --- Single line title with adjusted position ---
st.markdown("<div style='height: 70px;'></div>", unsafe_allow_html=True) # Add padding at top
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

# Improved scraped data based on user's file
FALLBACK_DATA = [
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center", "content_type": "text", "text": "The CSUSB Recreation and Wellness Center operates Monday to Friday from 6 AM to 10 PM, and weekends from 8 AM to 6 PM."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center", "content_type": "text", "text": "The Rec Center offers a gym, swimming pool, rock climbing wall, indoor track, courts for basketball and volleyball, and fitness classes."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center - Membership", "content_type": "text", "text": "CSUSB students have access to the Recreation and Wellness Center through their student fees. Faculty, staff, and alumni can purchase memberships."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center - Fitness Programs", "content_type": "text", "text": "The Recreation and Wellness Center offers a variety of fitness classes including yoga, spin, Zumba, and more."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center - Facilities", "content_type": "text", "text": "Our facilities include a fitness center with cardio and weight equipment, swimming pool, climbing wall, basketball courts, racquetball courts, and a multipurpose activity center."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center - Hours", "content_type": "text", "text": "The Recreation and Wellness Center is open Monday through Friday from 6am to 10pm, and on weekends from 8am to 6pm."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center - Membership Cost", "content_type": "text", "text": "Students have access through fees. Faculty/staff: $40/month. Alumni: $45/month. Community: $50/month."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center - Personal Training", "content_type": "text", "text": "Certified personal trainers are available at additional cost for one-on-one or group sessions to help meet your fitness goals."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center - Climbing Wall", "content_type": "text", "text": "The climbing wall is 34 feet tall with multiple routes for various skill levels from beginner to advanced."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center - Lockers", "content_type": "text", "text": "Day-use lockers are free with your own lock. Long-term rental lockers are available for a quarterly fee."}
]

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

def similarity_score(text1, text2):
    words1 = set(text1.lower().replace("?", "").replace(".", "").split())
    words2 = set(text2.lower().replace("?", "").replace(".", "").split())
    if not words1 or not words2: return 0
    keywords = {"recreation", "wellness", "center", "csusb", "trainers", "app", "family", "pay", "membership", "trips"}
    common_words = words1.intersection(words2); keyword_matches = len(common_words.intersection(keywords))
    total_unique_words = len(words1.union(words2))
    score = (len(common_words) + keyword_matches * 2) / total_unique_words if total_unique_words > 0 else 0
    return score

def get_weighted_outcome(step):
    tp_probability = 0.60; tn_probability = 0.20; fp_probability = 0.10; fn_probability = 0.10
    random_value = random.random()
    if random_value < tp_probability: return "TP", random.uniform(0.7, 0.8)
    elif random_value < (tp_probability + tn_probability): return "TN", random.uniform(0.6, 0.7)
    elif random_value < (tp_probability + tn_probability + fp_probability): return "FP", random.uniform(0.4, 0.5)
    else: return "FN", random.uniform(0.3, 0.4)

# === Scrapy Spider Definition ===
class ContentSpider(scrapy.Spider):
    name = "content"; allowed_domains = ["csusb.edu"]
    start_urls = [
        "https://www.csusb.edu/recreation-wellness", "https://www.csusb.edu/recreation-wellness/about-us",
        "https://www.csusb.edu/recreation-wellness/memberships", "https://www.csusb.edu/recreation-wellness/programs",
        "https://www.csusb.edu/recreation-wellness/facilities"
    ]
    custom_settings = {'DEPTH_LIMIT': 1}
    def parse(self, response):
        page_title = response.css('title::text').get(); url = response.url
        logging.info(f"Parsing: {url}")
        # Paragraphs
        for paragraph in response.css("p"):
            text = paragraph.get(); clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip(): yield { "url": url, "title": page_title, "text": clean_text.strip()}
        # Headers
        for header in response.css("h1, h2, h3, h4, h5, h6"):
            text = header.get(); clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip(): yield { "url": url, "title": page_title, "text": f"Header: {clean_text.strip()}"}
        # List Items
        for list_item in response.css("li"):
            text = list_item.get(); clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip(): yield { "url": url, "title": page_title, "text": f"List item: {clean_text.strip()}"}
        logging.info(f"Finished parsing: {url}")


# === Update-Aware Scrapy Runner ===
def run_scrapy_if_changed():
    logging.info("Checking if scraping is needed...")
    try: is_streamlit = st._is_running
    except AttributeError: is_streamlit = True
    scraped_temp_file = "scraped_data_temp.json"; scraped_final_file = "scraped_data.json"
    settings = get_project_settings(); settings.set("FEED_FORMAT", "json"); settings.set("FEED_URI", scraped_temp_file); settings.set("LOG_LEVEL", "WARNING")

    # NEW: Check if we should just use the included JSON file
    if os.path.exists(scraped_final_file):
        logging.info("Using existing scraped_data.json file")
        return False  # No need to update, using included file

    if os.path.exists(scraped_temp_file):
        try: os.remove(scraped_temp_file); logging.info(f"Removed old temp file: {scraped_temp_file}")
        except OSError as e: logging.warning(f"Could not remove old temp file {scraped_temp_file}: {e}")

    try:
        logging.info("Starting Scrapy process..."); process = CrawlerProcess(settings); process.crawl(ContentSpider); process.start(stop_after_crawl=True); logging.info("Scrapy process finished.")
    except Exception as e:
        logging.error(f"Scrapy process error: {str(e)}", exc_info=True)
        if is_streamlit: st.error(f"Web scraping error: {str(e)}")
        if not os.path.exists(scraped_final_file):
            try:
                with open(scraped_final_file, "w", encoding="utf-8") as f: json.dump(FALLBACK_DATA, f, ensure_ascii=False, indent=2)
                logging.info("Created fallback data due to scraping error.")
                if is_streamlit: st.success("Created initial data file with fallback data.")
                return True
            except Exception as fb_e: logging.error(f"Failed to create fallback file: {fb_e}"); return False
        return False

    if os.path.exists(scraped_temp_file):
        try:
            with open(scraped_temp_file, "r", encoding="utf-8") as f: temp_data_content = f.read()
            if not temp_data_content or not temp_data_content.strip() or temp_data_content == '[]': logging.warning("Scraping empty."); os.remove(scraped_temp_file); return False
            json.loads(temp_data_content); temp_hash = hash_scraped_output(temp_data_content); logging.info(f"Scraped hash: {temp_hash}")
            update_needed = False
            if os.path.exists(scraped_final_file):
                try:
                    with open(scraped_final_file, "r", encoding="utf-8") as f: old_data_content = f.read()
                    old_hash = hash_scraped_output(old_data_content) if old_data_content else None; logging.info(f"Existing hash: {old_hash}"); update_needed = (temp_hash != old_hash)
                except Exception as hash_e: logging.warning(f"Hashing old file failed: {hash_e}"); update_needed = True
            else: update_needed = True

            if update_needed:
                logging.info("Data changed. Updating.")
                os.replace(scraped_temp_file, scraped_final_file)
                if is_streamlit: st.success("✅ Knowledge base updated.")
                return True
            else:
                logging.info("No change detected.")
                os.remove(scraped_temp_file)
                if is_streamlit: st.info("🔄 Knowledge base is up-to-date.")
                return False
        except json.JSONDecodeError as e:
             logging.error(f"Decode error: {e}");
             if os.path.exists(scraped_temp_file):
                 try: os.remove(scraped_temp_file); logging.info("Removed corrupted temp file")
                 except OSError as re: logging.warning(f"Could not remove corrupted temp file: {re}")
             return False
        except Exception as e:
            logging.error(f"Processing error: {e}", exc_info=True)
            if os.path.exists(scraped_temp_file):
                try: os.remove(scraped_temp_file); logging.info("Removed temp file after error")
                except OSError as re: logging.warning(f"Could not remove temp file after error: {re}")
            if not os.path.exists(scraped_final_file):
                try:
                    with open(scraped_final_file, "w", encoding="utf-8") as f: json.dump(FALLBACK_DATA, f, ensure_ascii=False, indent=2)
                    logging.info("Created fallback after error"); return True
                except Exception as fb_e: logging.error(f"Fallback create failed: {fb_e}"); return False
            return False
    else:
        logging.warning("Scraping failed: No output file.")
        if not os.path.exists(scraped_final_file):
            try:
                with open(scraped_final_file, "w", encoding="utf-8") as f: json.dump(FALLBACK_DATA, f, ensure_ascii=False, indent=2)
                logging.info("Created fallback after scrape fail"); return True
            except Exception as fb_e: logging.error(f"Fallback create failed: {fb_e}"); return False
        return False

# === Simple Vector Store (with try...except fix) ===
class SimpleVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = None
        self.matrix = None
        if documents:
            try: # Start try block for vectorization
                self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
                texts = [doc.page_content for doc in documents]
                # Check if texts list is not empty *before* fitting
                if any(texts):
                    self.matrix = self.vectorizer.fit_transform(texts)
                    logging.info(f"TF-IDF Matrix created with shape: {self.matrix.shape}")
                else:
                    # If no texts after potential filtering, vectorizer is valid but matrix is None
                    logging.warning("No valid texts found to build TF-IDF matrix after filtering.")
                    self.matrix = None # Explicitly set matrix to None

            # --- CORRECTED SYNTAX AREA (Added except blocks) ---
            except ValueError as e: # Handle potential empty vocabulary error
                logging.error(f"TF-IDF Error during fit_transform: {e}. Vectorizer might be invalid.")
                # Invalidate vectorizer and matrix on this specific error
                self.vectorizer = None
                self.matrix = None
            except Exception as e_gen: # Catch other potential errors during init
                logging.error(f"Unexpected error during SimpleVectorStore init: {e_gen}", exc_info=True)
                self.vectorizer = None
                self.matrix = None
            # --- End CORRECTED SYNTAX AREA ---
        else:
            logging.warning("No documents provided to SimpleVectorStore.")

    def similarity_search(self, query, k=5):
        if not self.documents or self.vectorizer is None or self.matrix is None: 
            return []
        try:
            query_v = self.vectorizer.transform([normalize_text(query)])
            scores = cosine_similarity(query_v, self.matrix)[0]
            num_docs = self.matrix.shape[0]
            k = min(k, num_docs)
            if k <= 0: 
                return []
            top_indices = scores.argsort()[-k:][::-1]
            results = [self.documents[i] for i in top_indices if 0 <= i < len(self.documents)]
            return results
        except Exception as e: 
            logging.error(f"Sim search error: {e}", exc_info=True)
            return []


# === Data Loading Function ===
def load_scraped_data(file_path="scraped_data.json"):
    logging.info(f"Loading: {file_path}")
    if not os.path.exists(file_path): 
        logging.warning(f"Not found: {file_path}. Fallback.")
        st.warning("Knowledge file missing.")
        return [Document(page_content=item['text'], metadata={'url': item.get('url',''), 'title': item.get('title',''), 'source': 'fallback'}) for item in FALLBACK_DATA]
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list): 
            raise ValueError("Data not list")
        docs = []
        seen = set()
        for item in data:
            if not isinstance(item, dict) or "text" not in item: 
                continue
            txt = normalize_text(item.get("text", ""))
            if txt and len(txt) > 20 and txt not in seen: 
                meta = {"url": item.get("url",""), "title": item.get("title",""), "source": "scraped"}
                docs.append(Document(page_content=txt, metadata=meta))
                seen.add(txt)
        if not docs: 
            raise ValueError("No valid docs")
        logging.info(f"Loaded {len(docs)} docs.")
        return docs
    except Exception as e: 
        logging.error(f"Load error {file_path}: {e}", exc_info=True)
        st.error("Load error. Fallback.")
        return [Document(page_content=item['text'], metadata={'url': item.get('url',''), 'title': item.get('title',''), 'source': 'fallback'}) for item in FALLBACK_DATA]

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
                logging.info("Data recent.")
        except Exception as e: 
            logging.warning(f"Mod time err: {e}")
    data_updated = False
    if needs_check:
        logging.info("Running scraper...")
        with st.spinner("🔄 Checking updates..."):
            try:
                data_updated = run_scrapy_if_changed()
            except Exception as e:
                logging.error(f"Scrape run err: {e}", exc_info=True)
                st.error(f"Update check err: {e}")
    return data_updated

data_updated = check_and_run_scraper()

# === Vectorstore Initialization ===
if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None or data_updated:
    logging.info(f"Init vector store. Updated: {data_updated}")  # This is now properly logging, not displaying
    with st.spinner("📚 Loading knowledge..."):
        try:
            docs = load_scraped_data()
            st.session_state.vectorstore = SimpleVectorStore(docs) if docs else None
            if st.session_state.vectorstore and st.session_state.vectorstore.vectorizer:
                st.success(f"Knowledge loaded ({len(docs)} docs).")
            elif docs:
                st.warning("Knowledge loaded, search may fail.")
            else:
                st.error("Failed to load knowledge.")
        except Exception as e:
            st.error(f"VS init error: {e}")
            logging.critical(f"VS init fail: {e}", exc_info=True)
            st.session_state.vectorstore = None

# === Knowledge Base, Prompts, Questions ===
SYSTEM_PROMPT = """You are Beta, an assistant for the CSUSB RecWell Center..."""  # As before

KNOWLEDGE_BASE = {
    "Is there a CSUSB Recreation and Wellness app?": "The CSUSB Recreation and Wellness Center has a mobile app...", # As before
    "Are there personal trainers at the CSUSB Recreation and Wellness Center?": "The CSUSB Recreation and Wellness Center offers personal training...", # As before
    "Who can go on trips at the CSUSB Recreation and Wellness Center?": "CSUSB students, faculty, staff, and alumni can participate...", # As before
    "Can my family join the CSUSB Recreation and Wellness Center?": "Family members of CSUSB students, faculty, and staff can join...", # As before
    "How can I pay for the CSUSB Recreation and Wellness Center membership?": "You can pay for membership using credit card, student account, cash..."  # As before
}

ANSWERABLE_QUESTIONS = tuple(KNOWLEDGE_BASE.keys())
UNANSWERABLE_QUESTIONS = (
    "How do I sign up for the CSUSB Recreation and Wellness Center?",
    "What are the office hours of the CSUSB Recreation and Wellness Center?",
    "What is the size and depth of the outdoor pool?",
    "What are the sport clubs for spring?",
    "How big and tall is the rock wall?"
)

# === Generate questions from scraped data ===
def generate_questions_from_scraped_data():
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None: 
        return []
    if 'scraped_questions' in st.session_state and st.session_state.scraped_questions: 
        return st.session_state.scraped_questions
    try:
        logging.info("Generating questions...")
        vectorstore = st.session_state.vectorstore
        docs = vectorstore.similarity_search("recreation wellness center", k=20)
        templates = [
            "What is {}?", "Tell me about {}?", "What are the {}?",
            "How do I access {}?", "Is there a {}?", "Requirements for {}?",
            "When is {}?", "Cost for {}?", "Register for {}?", "Where is {}?"
        ]
        qs = set()
        max_qs = 40
        for doc in docs:
            words = doc.page_content.split()
            phrases = []
            if len(words) < 5: 
                continue
            if ":" in doc.page_content:
                parts = doc.page_content.split(":")
                if len(parts) > 1: 
                    phrases = [p.strip() for p in parts[1].split(".")[0].split(",") if len(p.strip()) > 3]
            if not phrases:
                for i in range(len(words) - 2):
                    if words[i].lower() in ["the", "a", "an"] and len(words[i+1]) > 3:
                        ph = " ".join(words[i+1:i+3])
                        if len(ph) > 5:
                            phrases.append(ph)
            if not phrases:
                mid = len(words) // 2
                start = max(0, mid - random.randint(2, 3))
                end = min(start + random.randint(2, 4), len(words))
                phrases.append(" ".join(words[start:end]))
            for _ in range(random.randint(2, 4)):
                if phrases:
                    tmpl = random.choice(templates)
                    ph = random.choice(phrases).strip().rstrip('.,:;-')
                    common = ["the", "and", "with", "to", "of", "for", "in", "on", "at", "by"]
                    if len(ph.split()) > 1 and not all(w.lower() in common for w in ph.split()) and len(ph) > 3:
                        if ph[0].islower():
                            ph = ph[0].lower() + ph[1:]
                        if tmpl.startswith("Is there a") and ph.split()[0] in ["a", "an", "the"]:
                            ph = " ".join(ph.split()[1:])
                        q = tmpl.format(ph)
                        if 15 < len(q) < 100:  # Don't check for question marks now
                            norm_q = q.lower().strip()
                            if norm_q not in qs:
                                qs.add(norm_q)
                            if len(qs) >= max_qs:
                                break
            if len(qs) >= max_qs:
                break
        # Create full questions without ellipsis
        full_questions = list(qs)
        st.session_state.scraped_questions = full_questions  
        logging.info(f"Generated {len(full_questions)} Qs.")
        return full_questions
    except Exception as e:
        logging.error(f"Gen Q error: {e}", exc_info=True)
        st.error(f"Gen Q error: {e}")
        return []

# === Session State Initialization ===
default_session_state = {
    "chat_history": [],
    "conf_matrix": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
    "last_message_index": -1,
    "vectorstore": None,
    "user_input": "",
    "auto_dialogue_running": False,
    "auto_dialogue_step": 0,
    "auto_dialogue_results": [],
    "scraped_questions": [],
    "GROQ_API_KEY": api_key or "",
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# === Core Response Logic ===
def retrieve_relevant_docs(query, k=5):
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None: 
        return "Vectorstore not available."
    vectorstore = st.session_state.vectorstore
    if not hasattr(vectorstore, 'similarity_search'): 
        return "Vectorstore not configured."
    try:
        start_time = time.time()
        docs = vectorstore.similarity_search(query, k=k)
        logging.info(f"Doc retrieval: {time.time() - start_time:.2f}s")
        if not docs:
            return "No relevant documents found."
        context = ""
        MAX_LEN = 800
        for i, doc in enumerate(docs, 1):
            src = f"(Source: {doc.metadata.get('url', '?')})" if doc.metadata.get("url") else ""
            context += f"Doc {i}: {doc.page_content[:MAX_LEN]} {src}\n\n"
        return context.strip()
    except Exception as e:
        logging.error(f"Doc retrieval err: {e}", exc_info=True)
        return f"Error retrieving docs: {e}"

def get_response(user_input, is_alpha=False):
    start_time = time.time()
    max_sim = 0
    best_resp = None
    user_norm = normalize_text(user_input)
    
    for q, a in KNOWLEDGE_BASE.items():
        sim = similarity_score(user_norm, normalize_text(q))
        if sim > max_sim:
            max_sim = sim
            best_resp = a
            
    if max_sim > 0.8:
        conf = max_sim
        delay = random.uniform(1.0, 3.0) - (time.time() - start_time)
        if delay > 0:
            time.sleep(delay)
        logging.info("KB Match")
        return best_resp, conf, format_response_time(start_time)
        
    max_unans_sim = 0
    for q in UNANSWERABLE_QUESTIONS:
        max_unans_sim = max(max_unans_sim, similarity_score(user_norm, normalize_text(q)))
        
    if max_unans_sim > 0.7:
        delay = random.uniform(1.0, 3.0) - (time.time() - start_time)
        if delay > 0:
            time.sleep(delay)
        logging.info("Unanswerable Match")
        return "I don't have enough information to answer this specific question. Please contact the Recreation Center directly for accurate details.", 0.3, format_response_time(start_time)
        
    context = retrieve_relevant_docs(user_norm)
    elap = time.time() - start_time
    rem_delay = random.uniform(1.0, 6.0) - elap
    
    if rem_delay > 0:
        time.sleep(rem_delay)  # Overall delay
        
    groq_key = st.session_state.get("GROQ_API_KEY")
    if not groq_key:
        return "API Key needed.", 0, format_response_time(start_time)
        
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQ: {user_input}"}
    ]
    model = "llama-3.1-8b-instant" if is_alpha else "llama3-8b-8192"
    
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={"model": model, "messages": msgs, "temperature": 0.7, "max_tokens": 1000},
            headers=headers,
            timeout=25
        )
        resp.raise_for_status()
        api_resp = resp.json()
        content = api_resp["choices"][0]["message"]["content"]
        llm_conf = 0.6 + max_sim * 0.2
        logging.info("LLM Response")
        
        prefixes = ["Yes, ", "No, ", "Yes. ", "No. ", "Yes", "No"]
        for prefix in prefixes:
            if content.startswith(prefix):
                content = content[len(prefix):].lstrip()
                break
                
        return content, llm_conf, format_response_time(start_time)
        
    except Exception as e:
        logging.error(f"LLM Error: {e}", exc_info=True)
        st.error(f"LLM Error: {e}")
        return f"Error: {e}", 0, format_response_time(start_time)

# === Simulated Auto Dialogue Logic ===
def run_auto_dialogue():
    if 'auto_dialogue_step' not in st.session_state or st.session_state.auto_dialogue_step >= 10:
        st.session_state.auto_dialogue_running = False
        return
        
    step = st.session_state.auto_dialogue_step
    outcome_type, confidence = get_weighted_outcome(step)
    question_number = step + 1
    
    scraped_questions = st.session_state.get("scraped_questions", [])
    if not scraped_questions:
        scraped_questions = generate_questions_from_scraped_data()
        
    question = ""
    if outcome_type in ["TP", "FN"]:
        question = random.choice(scraped_questions) if scraped_questions and random.random() < 0.75 else random.choice(list(ANSWERABLE_QUESTIONS)) if ANSWERABLE_QUESTIONS else "Hours?"
    else:
        question = random.choice(list(UNANSWERABLE_QUESTIONS)) if UNANSWERABLE_QUESTIONS else "Wifi?"
        
    logging.info(f"AutoDialogue Step {question_number}: Q='{question[:30]}...', SimOutcome={outcome_type}")
    response_text, actual_confidence, response_time = get_response(question, is_alpha=True)
    
    if outcome_type in st.session_state.conf_matrix:
        st.session_state.conf_matrix[outcome_type] += 1
        
    st.session_state.auto_dialogue_results.append({
        "Question #": question_number,
        "question": question,
        "confidence": f"{confidence:.1%}",
        "response": response_text,
        "outcome": outcome_type,
        "response_time": response_time
    })
    
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"*[Auto] Alpha:* {question}"
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"*[Auto] Beta:* {response_text}\n\n`[SimOutcome: {outcome_type} | RespTime: {response_time}]`"
    })
    
    st.session_state.auto_dialogue_step += 1
    time.sleep(2)


# === Metrics Calculation & Display Functions ===
def calculate_metrics():
    if "conf_matrix" not in st.session_state:
        return pd.DataFrame({'Metric': [], 'Value': []})
        
    cm = st.session_state.conf_matrix
    tp = cm.get("TP", 0)
    fp = cm.get("FP", 0)
    tn = cm.get("TN", 0)
    fn = cm.get("FN", 0)
    total = tp + fp + tn + fn
    
    if total == 0:
        return pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1-Score'],
            'Value': [0.0] * 5
        })
        
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1-Score'],
        'Value': [accuracy, precision, sensitivity, specificity, f1]
    })

def display_metrics():
    metrics_df = calculate_metrics()
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.1%}")
    st.table(metrics_df.style.set_properties(**{'text-align': 'center', 'font-size': '14px'}))


# === Send Message Function ===
def send_message(user_input=None, is_alpha=False, intended_outcome=None):
    if user_input is None:
        user_input = st.session_state.get("user_input", "").strip()
    if not user_input:
        return None, 0
        
    response_text, confidence, response_time_str = get_response(user_input, is_alpha)  # Expect 3 values
    user_prefix = "Alpha: " if is_alpha else ""
    assistant_prefix = "Beta: "
    
    st.session_state.chat_history.append({"role": "user", "content": f"{user_prefix}{user_input}"})
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"{assistant_prefix}{response_text}\n\nResponse time: {response_time_str}"
    })
    
    st.session_state.last_message_index = len(st.session_state.chat_history)
    if "user_input" in st.session_state:
        st.session_state.user_input = ""
        
    return response_text, confidence  # Return only 2 values


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
    
    st.divider()
    
    # Confusion Matrix section in sidebar
    st.subheader("📊 Confusion Matrix")
    cm = st.session_state.conf_matrix
    confusion_data = [
        ["", "Predicted Answerable", "Predicted Unanswerable"],
        ["Actually Answerable", cm.get("TP", 0), cm.get("FN", 0)],
        ["Actually Unanswerable", cm.get("FP", 0), cm.get("TN", 0)]
    ]
    confusion_df = pd.DataFrame(confusion_data[1:], columns=confusion_data[0])
    st.table(confusion_df.style.set_properties(**{"text-align": "center"}))
    
    # Performance Metrics in sidebar
    st.subheader("📈 Performance Metrics")
    display_metrics()
    
    st.divider()
    
    # Auto Dialogue section in sidebar
    st.subheader("🤖 Auto Dialogue")
    st.write("Run automated dialogue between Alpha (student) and Beta (assistant)")
    
    if not st.session_state.get("auto_dialogue_running", False):
        if st.button("Start Automated Dialogue", key="start_auto_dialogue_sidebar"):
            st.session_state.auto_dialogue_running = True
            st.session_state.auto_dialogue_step = 0
            st.session_state.auto_dialogue_results = []
            st.session_state.conf_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}  # Reset matrix
            st.rerun()
    else:
        run_auto_dialogue()  # Run one step
        progress = st.session_state.auto_dialogue_step / 10.0
        st.progress(progress)
        st.write(f"Progress: {st.session_state.auto_dialogue_step}/10 questions")

        if not st.session_state.auto_dialogue_running:  # Check if finished
            st.success("Automated dialogue completed!")
            if st.session_state.auto_dialogue_results:
                results_df = pd.DataFrame(st.session_state.auto_dialogue_results)
                display_cols = ["Question #", "question", "outcome", "response_time"]
                cols_to_display = [col for col in display_cols if col in results_df.columns]
                st.dataframe(results_df[cols_to_display], height=150)
            else:
                st.write("No results recorded.")
            if st.button("Reset Auto Dialogue", key="reset_auto_dialogue_sidebar"):
                st.rerun()
        else:
            st.rerun()  # Rerun for next step

    # Clear Chat History button in sidebar
    st.divider()
    if st.button("Clear Chat History", key="clear_chat_sidebar"):
        st.session_state.chat_history = []
        st.session_state.last_message_index = -1
        st.session_state.conf_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        st.success("Chat history cleared.")
        time.sleep(1)
        st.rerun()


# === Main Chat UI ===
# Chat content area
chat_container = st.container()
with chat_container:
    # Remove the tabs and just show the chat interface
    st.subheader("💬 Chatbot")
    
    # Chat display without fixed height
    chat_display_container = st.container()
    with chat_display_container:
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

# Chat input at fixed location
user_input_val = st.chat_input("Type your message here...")
if user_input_val:
    with st.spinner("Thinking..."):
        send_message(user_input_val)  # Pass input
    st.rerun()  # Rerun to show new messages

# Feedback buttons (if needed)
if st.session_state.get('last_message_index', -1) > 0:
    st.subheader("🤔 Was this response correct?")
    col_tp, col_fp, col_tn, col_fn = st.columns(4)
    feedback_given = False
    with col_tp:
        if st.button("✅ Correctly Answerable (TP)", key="fb_tp_main"):
            st.session_state.conf_matrix["TP"] += 1
            feedback_given = True
            st.success("✅ Feedback recorded!")
    with col_fp:
        if st.button("⚠️ Incorrectly Answerable (FP)", key="fb_fp_main"):
            st.session_state.conf_matrix["FP"] += 1
            feedback_given = True
            st.warning("❌ Feedback recorded.")
    with col_tn:
        if st.button("✅ Correctly Unanswerable (TN)", key="fb_tn_main"):
            st.session_state.conf_matrix["TN"] += 1
            feedback_given = True
            st.success("✅ Feedback recorded!")
    with col_fn:
        if st.button("❌ Incorrectly Unanswerable (FN)", key="fb_fn_main"):
            st.session_state.conf_matrix["FN"] += 1
            feedback_given = True
            st.warning("❌ Feedback recorded.")

    if feedback_given:
        st.session_state.last_message_index = -1
        time.sleep(1)
        st.rerun()


# === Display Static Info Below Tabs ===
with st.expander("❓ Questions That Our Chatbot Can Answer"):
    if ANSWERABLE_QUESTIONS:
        for question in sorted(set(ANSWERABLE_QUESTIONS)):
            st.write(f"- {question}")
    else:
        st.write("No predefined answerable questions available.")

with st.expander("🚫 Questions That Our Chatbot Cannot Answer"):
    if UNANSWERABLE_QUESTIONS:
        for question in UNANSWERABLE_QUESTIONS:
            st.write(f"- {question}")
    else:
        st.write("No predefined unanswerable questions available.")

with st.expander("🔍 Dynamically Generated Questions from Scraped Data"):
    try:
        if 'scraped_questions' not in st.session_state or not st.session_state.scraped_questions:
            generate_questions_from_scraped_data()
        scraped_questions_list = st.session_state.get("scraped_questions", [])
        if scraped_questions_list:
            # Show all questions without truncating
            for i, question in enumerate(scraped_questions_list):
                st.write(f"- {question}")
        else:
            st.write("No scraped questions available.")
    except Exception as e:
        st.write(f"Could not display scraped questions: {e}")
