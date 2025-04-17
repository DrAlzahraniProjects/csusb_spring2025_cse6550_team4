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
import logging
import hashlib

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

# Improved scraped data with accurate gym timings and other information
FALLBACK_DATA = [
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center", "content_type": "text", "text": "The CSUSB Recreation and Wellness Center operates Monday to Friday from 6 AM to 10 PM, and weekends from 8 AM to 6 PM."},
    {"url": "https://www.csusb.edu/recreation-wellness", "title": "CSUSB Recreation and Wellness Center - Gym Hours", "content_type": "text", "text": "Gym hours: Monday to Friday: 6 AM to 10 PM, Saturday and Sunday: 8 AM to 6 PM. Holiday hours may vary."},
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

def similarity_score(text1, text2):
    # Enhanced similarity scoring with more focus on keywords
    words1 = set(text1.lower().replace("?", "").replace(".", "").split())
    words2 = set(text2.lower().replace("?", "").replace(".", "").split())
    if not words1 or not words2: return 0
    keywords = {"recreation", "wellness", "center", "csusb", "trainers", "app", "family", "pay", "membership", 
                "trips", "gym", "timing", "hours", "open", "close", "schedule", "workout", "fitness", 
                "pool", "climbing", "wall", "basketball", "court", "lockers", "cost", "price"}
    common_words = words1.intersection(words2)
    keyword_matches = len(common_words.intersection(keywords))
    total_unique_words = len(words1.union(words2))
    # Give higher weight to keyword matches (3x) to better answer specific questions
    score = (len(common_words) + keyword_matches * 3) / total_unique_words if total_unique_words > 0 else 0
    return score

# === Scrapy Spider Definition ===
class ContentSpider(scrapy.Spider):
    name = "content"
    allowed_domains = ["csusb.edu"]
    start_urls = [
        "https://www.csusb.edu/recreation-wellness", 
        "https://www.csusb.edu/recreation-wellness/about-us",
        "https://www.csusb.edu/recreation-wellness/memberships", 
        "https://www.csusb.edu/recreation-wellness/programs",
        "https://www.csusb.edu/recreation-wellness/facilities", 
        "https://www.csusb.edu/recreation-wellness/hours"
    ]
    custom_settings = {'DEPTH_LIMIT': 2}  # Increased depth limit to capture more content
    
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
                
        # Tables - added to capture structured information like hours
        for table in response.css("table"):
            text = table.get()
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip(): 
                yield {
                    "url": url, 
                    "title": page_title, 
                    "text": f"Table: {clean_text.strip()}"
                }
                
        logging.info(f"Finished parsing: {url}")
        
        # Follow links within the domain - fixed to actually follow links
        for href in response.css('a::attr(href)').getall():
            if href and href.startswith('/recreation-wellness'):
                yield response.follow(href, self.parse)


# === Update-Aware Scrapy Runner (Fixed) ===
def run_scrapy_if_changed():
    logging.info("Starting scraping process...")
    try: 
        is_streamlit = st._is_running
    except AttributeError: 
        is_streamlit = True
        
    scraped_temp_file = "scraped_data_temp.json"
    scraped_final_file = "scraped_data.json"
    
    # Set Scrapy settings
    settings = get_project_settings()
    settings.set("FEED_FORMAT", "json")
    settings.set("FEED_URI", scraped_temp_file)
    settings.set("LOG_LEVEL", "INFO")  # Changed to INFO for better debugging

    # Remove old temp file if it exists
    if os.path.exists(scraped_temp_file):
        try:
            os.remove(scraped_temp_file)
            logging.info(f"Removed old temp file: {scraped_temp_file}")
        except OSError as e:
            logging.warning(f"Could not remove old temp file {scraped_temp_file}: {e}")

    try:
        logging.info("Starting Scrapy crawler...")
        process = CrawlerProcess(settings)
        process.crawl(ContentSpider)
        process.start()  # This will block until crawling is finished
        logging.info("Scrapy process completed.")
    except Exception as e:
        logging.error(f"Scrapy process error: {str(e)}", exc_info=True)
        if is_streamlit:
            st.error(f"Web scraping error: {str(e)}")
        
        # Use fallback if no existing data file
        if not os.path.exists(scraped_final_file):
            try:
                with open(scraped_final_file, "w", encoding="utf-8") as f:
                    json.dump(FALLBACK_DATA, f, ensure_ascii=False, indent=2)
                logging.info("Created fallback data file due to scraping error.")
                if is_streamlit:
                    st.success("Created initial data file with fallback data.")
                return True
            except Exception as fb_e:
                logging.error(f"Failed to create fallback file: {fb_e}")
                return False
        return False

    # Process the scraped data if temp file exists
    if os.path.exists(scraped_temp_file):
        try:
            with open(scraped_temp_file, "r", encoding="utf-8") as f:
                temp_data_content = f.read()
                
            # Check if valid data was scraped
            if not temp_data_content or temp_data_content.strip() == '' or temp_data_content == '[]':
                logging.warning("Scraping produced empty results.")
                os.remove(scraped_temp_file)
                
                # Use fallback data if no existing file
                if not os.path.exists(scraped_final_file):
                    with open(scraped_final_file, "w", encoding="utf-8") as f:
                        json.dump(FALLBACK_DATA, f, ensure_ascii=False, indent=2)
                    logging.info("Created fallback data due to empty scrape results.")
                    return True
                return False
                
            # Valid data - process it
            temp_data = json.loads(temp_data_content)
            temp_hash = hash_scraped_output(temp_data_content)
            logging.info(f"New scraped data hash: {temp_hash[:10]}...")
            
            # Check if data is different from existing file
            update_needed = True
            if os.path.exists(scraped_final_file):
                try:
                    with open(scraped_final_file, "r", encoding="utf-8") as f:
                        old_data_content = f.read()
                    old_hash = hash_scraped_output(old_data_content)
                    logging.info(f"Existing data hash: {old_hash[:10]}...")
                    update_needed = (temp_hash != old_hash)
                except Exception as hash_e:
                    logging.warning(f"Error comparing file hashes: {hash_e}")
                    update_needed = True
            
            # Update data file if needed
            if update_needed:
                logging.info("Data changed or no existing file. Updating knowledge base.")
                os.replace(scraped_temp_file, scraped_final_file)
                if is_streamlit:
                    st.success("✅ Knowledge base updated with fresh data.")
                return True
            else:
                logging.info("No change in data detected.")
                os.remove(scraped_temp_file)
                if is_streamlit:
                    st.info("🔄 Knowledge base is already up-to-date.")
                return False
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error with scraped data: {e}")
            if os.path.exists(scraped_temp_file):
                try:
                    os.remove(scraped_temp_file)
                    logging.info("Removed corrupted temp file")
                except OSError as re:
                    logging.warning(f"Could not remove corrupted temp file: {re}")
            return False
            
        except Exception as e:
            logging.error(f"Error processing scraped data: {e}", exc_info=True)
            if os.path.exists(scraped_temp_file):
                try:
                    os.remove(scraped_temp_file)
                    logging.info("Removed temp file after error")
                except OSError as re:
                    logging.warning(f"Could not remove temp file after error: {re}")
            
            # Use fallback if needed
            if not os.path.exists(scraped_final_file):
                try:
                    with open(scraped_final_file, "w", encoding="utf-8") as f:
                        json.dump(FALLBACK_DATA, f, ensure_ascii=False, indent=2)
                    logging.info("Created fallback after error")
                    return True
                except Exception as fb_e:
                    logging.error(f"Fallback creation failed: {fb_e}")
                    return False
            return False
    else:
        logging.warning("Scraping failed: No output file was created.")
        if not os.path.exists(scraped_final_file):
            try:
                with open(scraped_final_file, "w", encoding="utf-8") as f:
                    json.dump(FALLBACK_DATA, f, ensure_ascii=False, indent=2)
                logging.info("Created fallback after scrape fail")
                return True
            except Exception as fb_e:
                logging.error(f"Fallback create failed: {fb_e}")
                return False
        return False

# === Simple Vector Store (with improved error handling) ===
class SimpleVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = None
        self.matrix = None
        if documents:
            try:
                self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)
                texts = [doc.page_content for doc in documents if doc and hasattr(doc, 'page_content')]
                # Check if texts list is not empty *before* fitting
                if texts and any(texts):
                    self.matrix = self.vectorizer.fit_transform(texts)
                    logging.info(f"TF-IDF Matrix created with shape: {self.matrix.shape}")
                else:
                    logging.warning("No valid texts found to build TF-IDF matrix after filtering.")
                    self.matrix = None
            except ValueError as e:
                logging.error(f"TF-IDF Error during fit_transform: {e}. Vectorizer might be invalid.")
                self.vectorizer = None
                self.matrix = None
            except Exception as e_gen:
                logging.error(f"Unexpected error during SimpleVectorStore init: {e_gen}", exc_info=True)
                self.vectorizer = None
                self.matrix = None
        else:
            logging.warning("No documents provided to SimpleVectorStore.")

    def similarity_search(self, query, k=5):
        if not self.documents or not self.vectorizer or self.matrix is None: 
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
            logging.error(f"Similarity search error: {e}", exc_info=True)
            return []


# === Data Loading Function ===
def load_scraped_data(file_path="scraped_data.json"):
    logging.info(f"Loading data from: {file_path}")
    if not os.path.exists(file_path): 
        logging.warning(f"File not found: {file_path}. Using fallback data.")
        st.warning("Knowledge file missing. Using default data.")
        return [Document(page_content=item['text'], metadata={'url': item.get('url',''), 'title': item.get('title',''), 'source': 'fallback'}) for item in FALLBACK_DATA]
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
                meta = {"url": item.get("url",""), "title": item.get("title",""), "source": "scraped"}
                docs.append(Document(page_content=txt, metadata=meta))
                seen.add(txt)
        
        if not docs: 
            raise ValueError("No valid documents extracted")
            
        logging.info(f"Successfully loaded {len(docs)} documents.")
        return docs
    except Exception as e: 
        logging.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        st.error("Error loading knowledge base. Using fallback data.")
        return [Document(page_content=item['text'], metadata={'url': item.get('url',''), 'title': item.get('title',''), 'source': 'fallback'}) for item in FALLBACK_DATA]

# === Conditional Scraper Run (Fixed) ===
@st.cache_data(ttl=3600)  # Reduced cache time to 1 hour for testing
def check_and_run_scraper():
    logging.info("Checking if scraper needs to run...")
    data_file = "scraped_data.json"
    force_update = False  # Set to True to force update for testing
    
    # Check if data file exists and is recent
    if os.path.exists(data_file) and not force_update: 
        try:
            file_age = time.time() - os.path.getmtime(data_file)
            if file_age < 3600:  # Less than 1 hour old
                logging.info(f"Data file is recent ({file_age:.1f} seconds old). Skipping scrape.")
                return False
            else:
                logging.info(f"Data file is {file_age:.1f} seconds old. Will check for updates.")
        except Exception as e: 
            logging.warning(f"Error checking file modification time: {e}")
    else:
        if not os.path.exists(data_file):
            logging.info("No existing data file. Will run scraper.")
        else:
            logging.info("Force update enabled. Will run scraper.")
            
    # Run the scraper
    with st.spinner("🔄 Checking for updated information..."):
        try:
            data_updated = run_scrapy_if_changed()
            logging.info(f"Scraper run completed. Data updated: {data_updated}")
            return data_updated
        except Exception as e:
            logging.error(f"Error running scraper: {e}", exc_info=True)
            st.error(f"Error checking for updates: {e}")
            return False

# === Knowledge Base, Prompts ===
SYSTEM_PROMPT = """You are Beta, an assistant for the CSUSB RecWell Center. Answer questions about the CSUSB Recreation and Wellness Center based on the context provided. Keep responses concise and focus on recreation and wellness topics. If you're unsure about something, politely explain that you don't have that information and suggest the user contact the RecWell Center directly."""  

# Enhanced knowledge base with more accurate information
KNOWLEDGE_BASE = {
    "What are the gym timings?": "The CSUSB Recreation and Wellness Center gym is open Monday to Friday from 6 AM to 10 PM, and on weekends (Saturday and Sunday) from 8 AM to 6 PM. Holiday hours may vary, so it's best to check the RecWell app or website for any schedule changes.",
    "Is there a CSUSB Recreation and Wellness app?": "The CSUSB Recreation and Wellness Center has a mobile app called 'RecWell' available for both iOS and Android. The app allows you to check facility hours, view fitness class schedules, register for events, book courts, and get notifications about special events.",
    "Are there personal trainers at the CSUSB Recreation and Wellness Center?": "The CSUSB Recreation and Wellness Center offers personal training services with certified professional trainers. They provide one-on-one sessions, fitness assessments, and personalized workout plans at additional cost to members. Sessions can be scheduled through the front desk or the RecWell app.",
    "Who can go on trips at the CSUSB Recreation and Wellness Center?": "CSUSB students, faculty, staff, and alumni can participate in adventure trips organized by the Recreation and Wellness Center. Family members may be allowed on certain specified family-friendly trips. Each trip has specific requirements that are listed in the description when you register.",
    "Can my family join the CSUSB Recreation and Wellness Center?": "Family members of CSUSB students, faculty, and staff can join the Recreation and Wellness Center through family membership options. Spouses/partners and dependent children under 18 are eligible. Family members must be accompanied by the primary member when using the facilities. Check with the membership desk for current rates and policies.",
    "How can I pay for the CSUSB Recreation and Wellness Center membership?": "You can pay for membership using credit card, student account charging, cash, or check at the membership office. Faculty and staff have the option for payroll deduction. Students already have access through their student fees. Monthly and annual payment options are available for eligible members.",
    "What facilities are available at the CSUSB Recreation and Wellness Center?": "The CSUSB Recreation and Wellness Center offers numerous facilities including a fitness center with cardio and weight equipment, swimming pool, 34-foot climbing wall, basketball courts, volleyball courts, racquetball courts, indoor track, and multipurpose activity spaces for fitness classes.",
    "How much does a CSUSB Recreation and Wellness Center membership cost?": "CSUSB students already have access through their student fees. For faculty and staff, membership costs approximately $40/month. Alumni membership is around $45/month, and community membership is approximately $50/month. Contact the center directly for the most current pricing information."
}

# === Session State Initialization ===
default_session_state = {
    "chat_history": [],
    "vectorstore": None,
    "user_input": "",
    "GROQ_API_KEY": api_key or "",
    "messages": [], # Initialize messages list if doesn't exist
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# === Core Response Logic (Improved) ===
def retrieve_relevant_docs(query, k=5):
    if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None: 
        return "Vectorstore not available."
    vectorstore = st.session_state.vectorstore
    if not hasattr(vectorstore, 'similarity_search'): 
        return "Vectorstore not configured correctly."
    try:
        docs = vectorstore.similarity_search(query, k=k)
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
    # First check if we have a direct match in our knowledge base
    max_sim = 0
    best_resp = None
    user_norm = normalize_text(user_input)
    
    for q, a in KNOWLEDGE_BASE.items():
        sim = similarity_score(user_norm, normalize_text(q))
        if sim > max_sim:
            max_sim = sim
            best_resp = a
            
    # Lower threshold for better matches
    if max_sim > 0.6:
        logging.info(f"Knowledge Base match found with score {max_sim}")
        return best_resp
        
    # If not in knowledge base, query the vector store
    relevant_docs = retrieve_relevant_docs(user_input)
    if relevant_docs.startswith("Error") or relevant_docs == "Vectorstore not available." or relevant_docs == "No relevant documents found.":
        # Improved fallback response that's more helpful
        fallback_response = "I'm sorry, I don't have specific information about that regarding the Recreation and Wellness Center. You might want to check the official CSUSB RecWell website or contact them directly at the front desk for the most accurate information. Is there anything else I can help you with about the facilities, hours, or membership options?"
        return fallback_response
        
    # Enhanced response generation based on retrieved docs
    response = "Based on the information I have: "
    
    # Extract key facts from the relevant docs with better filtering
    docs_list = relevant_docs.split("\n\n")
    facts = []
    
    # First check if any docs contain gym timing information when asked about hours
    if any(word in user_norm for word in ["hour", "time", "open", "close", "schedule", "timing"]):
        hour_docs = [doc for doc in docs_list if any(word in doc.lower() for word in ["hour", "time", "am", "pm", "open", "close", "schedule"])]
        if hour_docs:
            for doc in hour_docs[:2]:  # Use up to 2 most relevant hour-related docs
                doc_text = doc.split(": ", 1)[1] if ": " in doc else doc
                doc_text = doc_text.split(" (Source")[0] if " (Source" in doc_text else doc_text
                if doc_text and len(doc_text) > 10:
                    facts.append(doc_text)
    
    # If we don't have facts yet, process normally
    if not facts:
        for doc in docs_list:
            doc_text = doc.split(": ", 1)[1] if ": " in doc else doc
            doc_text = doc_text.split(" (Source")[0] if " (Source" in doc_text else doc_text
            if doc_text and len(doc_text) > 10:
                facts.append(doc_text)
    if facts:
        response += " ".join(facts[:2])  # Only use first two facts to keep it concise
    else:
        response = "I have some information about the Recreation and Wellness Center, but I'm not sure about the specific details you're asking for. The RecWell Center is open Monday-Friday 6AM-10PM and weekends 8AM-6PM. For more specific information, please contact the center directly."
        
    return response

# First run the scraper before initializing anything else
try:
    # Only call this outside the chat interface to avoid redundant operations
    data_updated = check_and_run_scraper()
except Exception as e:
    logging.error(f"Scraper initialization error: {e}", exc_info=True)
    data_updated = False
    st.error("Error during knowledge base initialization. Using default data.")

# === Vectorstore Initialization ===
if 'vectorstore' not in st.session_state or st.session_state.vectorstore is None or data_updated:
    logging.info(f"Initializing vector store. Updated data: {data_updated}")
    with st.spinner("📚 Loading knowledge base..."):
        try:
            docs = load_scraped_data()
            st.session_state.vectorstore = SimpleVectorStore(docs) if docs else None
            if st.session_state.vectorstore and st.session_state.vectorstore.vectorizer:
                logging.info(f"Knowledge base loaded with {len(docs)} documents.")
            elif docs:
                st.warning("Knowledge loaded, but search functionality may be limited.")
            else:
                st.error("Failed to load knowledge base.")
        except Exception as e:
            st.error(f"Vector store initialization error: {e}")
            logging.critical(f"Vector store initialization failure: {e}", exc_info=True)
            st.session_state.vectorstore = None

# === Chat Interface (FIXED) ===
# Initialize chat history - only if it doesn't exist
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Hello! I'm the CSUSB Recreation and Wellness Center chatbot. How can I help you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about RecWell Center..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = get_response(prompt)
            st.markdown(response_text)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
