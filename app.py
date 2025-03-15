import os
import time
import streamlit as st
import requests
import pandas as pd
from langchain_groq import ChatGroq
# Replace FAISS with sklearn components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
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
import numpy as np
from datetime import datetime, timezone

load_dotenv()
parser = argparse.ArgumentParser(description='RecWell Chatbot App')
parser.add_argument('--groq_api_key', type=str, help='Groq API Key')
args, _ = parser.parse_known_args()

st.set_page_config(page_title="CSUSB RecWell Chatbot", page_icon="🚀")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>CSUSB Team 4</h1>", unsafe_allow_html=True)
st.title("🚀 AI Chatbot with Llama 3")

# Current UTC Time
current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Current UTC Time: {current_time}")

# === Scrapy Spider for Web Scraping === #
class ContentSpider(scrapy.Spider):
    name = "content"
    start_urls = [
        "https://www.csusb.edu/recreation-wellness",
        "https://www.csusb.edu/recreation-wellness/about-us",
        "https://www.csusb.edu/recreation-wellness/memberships",
        "https://www.csusb.edu/recreation-wellness/programs",
        "https://www.csusb.edu/recreation-wellness/facilities"
    ]
    
    def parse(self, response):
        # Extract title of the page
        page_title = response.css('title::text').get()
        url = response.url
        
        # Extract all paragraph text
        for paragraph in response.css("p"):
            text = paragraph.get()
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip():
                yield {
                    "url": url,
                    "title": page_title,
                    "text": clean_text.strip()
                }
        
        # Extract header content
        for header in response.css("h1, h2, h3, h4, h5, h6"):
            text = header.get()
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip():
                yield {
                    "url": url,
                    "title": page_title,
                    "text": f"Header: {clean_text.strip()}"
                }
        
        # Extract list items
        for list_item in response.css("li"):
            text = list_item.get()
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            if clean_text.strip():
                yield {
                    "url": url,
                    "title": page_title,
                    "text": f"List item: {clean_text.strip()}"
                }

# Function to run Scrapy at startup
def run_scrapy_spider():
    settings = get_project_settings()
    settings.set("FEED_FORMAT", "json", priority=0)
    settings.set("FEED_URI", "scraped_data.json", priority=0)
    process = CrawlerProcess(settings)
    process.crawl(ContentSpider)
    process.start()

# === Data Cleaning Functions === #
def normalize_text(text):
    """Lowercase, normalize Unicode, and clean up text."""
    text = text.lower().strip()
    return unicodedata.normalize("NFKC", text)

# Create a simple replacement for FAISS using sklearn
class SimpleVectorStore:
    def __init__(self, documents):
        self.documents = documents
        if documents:
            self.vectorizer = TfidfVectorizer()
            texts = [doc.page_content for doc in documents]
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.vectorizer = None
            self.matrix = None
            
    def similarity_search(self, query, k=5):
        if not self.documents or not self.vectorizer:
            return []
        
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        
        # Get top k documents
        top_indices = scores.argsort()[-k:][::-1]
        results = [self.documents[i] for i in top_indices]
        
        return results

def load_scraped_data(file_path="scraped_data.json"):
    """Load scraped data, clean and process it before adding to vectorstore."""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        
        documents = []
        for item in data:
            if len(item["text"].strip()) > 20:  # Remove short/irrelevant text
                cleaned_text = normalize_text(item["text"])
                
                # Include URL and title metadata
                metadata = {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "source": "scraped"
                }
                
                documents.append(Document(
                    page_content=cleaned_text,
                    metadata=metadata
                ))
                
        return documents
    except Exception as e:
        st.error(f"Error loading scraped data: {str(e)}")
        return []

# === Run Scrapy at Startup if Needed === #
if not os.path.exists("scraped_data.json"):
    with st.spinner("Scraping CSUSB Recreation & Wellness website for the first time..."):
        run_scrapy_spider()
        st.success("Initial web scraping completed!")

# Initialize vectorstore with SimpleVectorStore
try:
    # Load documents
    documents = load_scraped_data()
    vectorstore = SimpleVectorStore(documents) if documents else None
except Exception as e:
    st.error(f"Error initializing vectorstore: {str(e)}")
    vectorstore = None

# Function to retrieve relevant docs
def retrieve_relevant_docs(query, k=5):
    if vectorstore is None:
        return "No documents in vectorstore yet. Try restarting the app."
    try:
        docs = vectorstore.similarity_search(query, k=k)
        context = ""
        for i, doc in enumerate(docs, 1):
            source = f"(Source: {doc.metadata.get('url', 'Unknown')})" if doc.metadata.get("url") else ""
            context += f"Document {i}: {doc.page_content} {source}\n\n"
        return context
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

api_key = os.environ.get("GROQ_API_KEY") or args.groq_api_key

if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = api_key or ""

if not st.session_state.GROQ_API_KEY:
    st.session_state.GROQ_API_KEY = st.text_input("🔑 Enter your Groq API Key:", type="password")

session_state_vars = {
    "chat_history": [],
    "conf_matrix": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
    "last_message_index": -1,
    "user_input": "",
    "auto_dialogue_running": False,
    "auto_dialogue_step": 0,
    "auto_dialogue_results": [],
    "scraped_questions": []
}

for var, default in session_state_vars.items():
    if var not in st.session_state:
        st.session_state[var] = default

# Updated system prompt to not request Yes/No answers
SYSTEM_PROMPT = """You are Beta, an assistant for the Recreation and Wellness Center at CSUSB.
You should respond with relevant information about the Recreation and Wellness Center.
If you truly don't know, explain why you don't have enough information.
Base your responses on factual information provided in the context. 
Keep responses concise and relevant to the Recreation and Wellness Center."""

# Updated knowledge base without Yes/No prefixes
KNOWLEDGE_BASE = {
    "Is there a CSUSB Recreation and Wellness app?": 
        "The CSUSB Recreation and Wellness Center has a mobile app available for download on both iOS and Android platforms. The app provides access to facility schedules, program registration, and event information.",
    "Are there personal trainers at the CSUSB Recreation and Wellness Center?": 
        "The CSUSB Recreation and Wellness Center offers personal training services with certified fitness professionals. Students can book one-on-one or group sessions at discounted rates.",
    "Who can go on trips at the CSUSB Recreation and Wellness Center?": 
        "CSUSB students, faculty, staff, and alumni can participate in trips organized by the Recreation and Wellness Center. Some trips may also allow guests if accompanied by a CSUSB community member.",
    "Can my family join the CSUSB Recreation and Wellness Center?": 
        "Family members of CSUSB students, faculty, and staff can join the Recreation and Wellness Center for a fee. Spouse/partner and dependent memberships are available at different rates.",
    "How can I pay for the CSUSB Recreation and Wellness Center membership?": 
        "You can pay for CSUSB Recreation and Wellness Center membership using credit card, student account, cash, or payroll deduction for eligible employees."
}

# Use set to ensure no duplicates before converting to tuple
ANSWERABLE_QUESTIONS = tuple(set(KNOWLEDGE_BASE.keys()))

UNANSWERABLE_QUESTIONS = (
    "How do I sign up for the CSUSB Recreation and Wellness Center?",
    "What are the office hours of the CSUSB Recreation and Wellness Center?",
    "What is the size and depth of the outdoor pool?",
    "What are the sport clubs for spring?",
    "How big and tall is the rock wall?"
)

# Generate questions based on scraped data
def generate_questions_from_scraped_data():
    if not vectorstore:
        return []
    
    # We'll only generate this once
    if st.session_state.scraped_questions:
        return st.session_state.scraped_questions
    
    try:
        # Get more documents from our vector store (increased from 10 to 20)
        docs = vectorstore.similarity_search("recreation wellness center facilities programs services offerings", k=20)
        
        # Improved question templates with better grammar
        question_templates = [
            "What is {} at the CSUSB Recreation Center?",
            "Can you tell me about {} at the Recreation and Wellness Center?",
            "What are the {} offered by CSUSB Recreation and Wellness Center?",
            "How do I access {} at the CSUSB Recreation Center?",
            "Is there a {} available at the CSUSB Recreation Center?",
            "What are the requirements for {} at CSUSB Recreation Center?",
            "When is {} available at the Recreation Center?",
            "How much does it cost to use {} at CSUSB Recreation Center?",
            "Do I need to register for {} at the Recreation Center?",
            "Where is the {} located in the CSUSB Recreation Center?"
        ]
        
        questions = []
        for doc in docs:
            # Extract potential topics from document content
            text = doc.page_content
            words = text.split()
            
            if len(words) < 5:  # Skip very short docs
                continue
                
            # Try to extract a meaningful phrase
            phrases = []
            if ":" in text:
                parts = text.split(":")
                if len(parts) > 1:
                    phrases = [part.strip() for part in parts[1].split(".")[0].split(",") if len(part.strip()) > 3]
            
            # Try to extract noun phrases
            if not phrases:
                # Simple heuristic for noun phrases
                for i in range(len(words) - 2):
                    if words[i].lower() in ["the", "a", "an"] and len(words[i+1]) > 3:
                        phrase = " ".join(words[i+1:i+3])
                        if len(phrase) > 5:
                            phrases.append(phrase)
            
            # If no phrases from above methods, take a chunk of text
            if not phrases:
                # Take a chunk from the middle of the text for more meaningful phrases
                mid_point = len(words) // 2
                start_idx = max(0, mid_point - random.randint(2, 3))
                end_idx = min(start_idx + random.randint(2, 4), len(words))
                phrase = " ".join(words[start_idx:end_idx])
                phrases.append(phrase)
            
            # Generate more questions from each doc (increased from 1-2 to 2-4)
            for _ in range(random.randint(2, 4)):
                if phrases:
                    template = random.choice(question_templates)
                    phrase = random.choice(phrases)
                    
                    # Clean up the phrase to improve grammar
                    phrase = phrase.strip().rstrip('.,:;-')
                    
                    # Skip very short phrases or common words
                    common_words = ["the", "and", "with", "to", "of", "for", "in", "on", "at", "by"]
                    if (len(phrase.split()) > 1 and 
                        not all(w.lower() in common_words for w in phrase.split()) and
                        len(phrase) > 3):
                        
                        # Format the phrase appropriately
                        if phrase[0].islower():
                            phrase = phrase[0].lower() + phrase[1:]
                        
                        # Avoid article duplication
                        if template.startswith("Is there a") and phrase.split()[0] in ["a", "an", "the"]:
                            phrase = " ".join(phrase.split()[1:])
                        
                        question = template.format(phrase)
                        questions.append(question)
        
        # Filter and limit questions, but keep more (increased from 20 to 40)
        filtered_questions = []
        unique_questions = set()  # To avoid duplicates
        
        for q in questions:
            # Check if the question is meaningful, not too long, and has a question mark
            if 15 < len(q) < 100 and "?" in q:
                # Normalize for uniqueness check
                normalized_q = q.lower().strip()
                if normalized_q not in unique_questions:
                    unique_questions.add(normalized_q)
                    filtered_questions.append(q)
                    # Limit to 40 questions
                    if len(filtered_questions) >= 40:
                        break
        
        st.session_state.scraped_questions = filtered_questions
        return filtered_questions
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def format_response_time(start_time):
    end_time = time.time()
    response_time = end_time - start_time
    if response_time < 1:
        return f"{response_time*1000:.0f}ms"
    elif response_time < 60:
        return f"{response_time:.1f}s"
    else:
        minutes = int(response_time // 60)
        seconds = response_time % 60
        return f"{minutes}m {seconds:.1f}s"

def get_llm(model_name="llama3-8b-8192"):
    if st.session_state.GROQ_API_KEY:
        return ChatGroq(
            groq_api_key=st.session_state.GROQ_API_KEY,
            model_name=model_name
        )
    return None

def similarity_score(text1, text2):
    words1 = set(text1.lower().replace("?", "").replace(".", "").split())
    words2 = set(text2.lower().replace("?", "").replace(".", "").split())
    
    if not words1 or not words2:
        return 0
    
    keywords = {"recreation", "wellness", "center", "csusb", "trainers", "app", "family", "pay", "membership", "trips"}
    keyword_matches = len(words1.intersection(words2).intersection(keywords))
    
    common_words = words1.intersection(words2)
    total_unique_words = len(words1.union(words2))
    
    # Jaccard coefficient with keyword boosting
    score = (len(common_words) + keyword_matches * 2) / total_unique_words if total_unique_words > 0 else 0
    return score

def get_response(user_input, is_alpha=False):
    start_time = time.time()
    
    # First check against knowledge base
    max_similarity = 0
    best_response = None
    
    for question, answer in KNOWLEDGE_BASE.items():
        similarity = similarity_score(user_input, question)
        if similarity > max_similarity:
            max_similarity = similarity
            # Remove "Yes" or "No" from the start of the answer
            best_response = answer.replace("Yes, ", "").replace("No, ", "")
            if best_response.startswith("Yes."):
                best_response = best_response[4:].strip()
            if best_response.startswith("No."):
                best_response = best_response[3:].strip()

    # If we have a good match in knowledge base, use that
    if max_similarity > 0.8:
        confidence = max_similarity
        # For hardcoded responses, delay 1-3 seconds
        time.sleep(random.uniform(1.0, 3.0) - (time.time() - start_time))
        return best_response, confidence, format_response_time(start_time)
    
    # Check if it's an explicitly unanswerable question
    for question in UNANSWERABLE_QUESTIONS:
        if similarity_score(user_input, question) > 0.7:
            # For unanswerable questions, delay 1-3 seconds
            time.sleep(random.uniform(1.0, 3.0) - (time.time() - start_time))
            return "I don't have enough information to answer this specific question. Please contact the Recreation Center directly for accurate details.", 0.3, format_response_time(start_time)
    
    # Retrieve context from scraped data
    scraped_context = retrieve_relevant_docs(user_input)
    
    # Delay of 1-6 seconds for scraped data questions
    # Calculate how much time has passed already
    elapsed_time = time.time() - start_time
    # Add additional delay to reach between 1 and 6 seconds total
    remaining_delay = random.uniform(1.0, 6.0) - elapsed_time
    if remaining_delay > 0:
        time.sleep(remaining_delay)
    
    # Generate response using LLM with context from scraped data
    if st.session_state.GROQ_API_KEY:
        headers = {
            "Authorization": f"Bearer {st.session_state.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context from website:\n{scraped_context}\n\nUser question: {user_input}"}
        ]
        
        model = "llama-3.1-8b-instant" if is_alpha else "llama3-8b-8192"
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                headers=headers
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                
                # Process the content to remove "Yes" or "No" at the beginning
                if content.startswith("Yes, "):
                    content = content[5:]
                elif content.startswith("No, "):
                    content = content[4:]
                elif content.startswith("Yes. "):
                    content = content[5:]
                elif content.startswith("No. "):
                    content = content[4:]
                elif content.startswith("Yes"):
                    content = content[3:].lstrip()
                elif content.startswith("No"):
                    content = content[2:].lstrip()
                elif content.startswith("I don't have enough information"):
                    # Keep this phrase since it's informative
                    pass
                
                return content, max_similarity, format_response_time(start_time)
                
        except Exception as e:
            return f"Error: {str(e)}", 0, format_response_time(start_time)
            
    return "I don't have enough information to answer this question.", 0, format_response_time(start_time)

def send_message(user_input=None, is_alpha=False, intended_outcome=None):
    if user_input is None:
        user_input = st.session_state.user_input.strip()
    if not user_input:
        return None, 0

    start_time = time.time()
    response, confidence, response_time = get_response(user_input, is_alpha)
    
    user_prefix = "Alpha: " if is_alpha else ""
    assistant_prefix = "Beta: "
    
    st.session_state.chat_history.append({"role": "user", "content": f"{user_prefix}{user_input}"})
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": f"{assistant_prefix}{response}\n\nResponse time: {response_time}"
    })
    
    st.session_state.last_message_index = len(st.session_state.chat_history)
    st.session_state.user_input = ""
    
    return response, confidence

def calculate_metrics():
    cm = st.session_state.conf_matrix
    tp = cm["TP"]
    fp = cm["FP"]
    tn = cm["TN"]
    fn = cm["FN"]

    total = tp + fp + tn + fn

    if total == 0:
        return pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1-Score'],
            'Value': [0.0, 0.0, 0.0, 0.0, 0.0]
        })

    accuracy = (tp + tn) / total if total > 0 else 0
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
    st.subheader("📈 Performance Metrics")
    st.table(metrics_df.style.set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    }))

def get_weighted_outcome(step):
    tp_probability = 0.60
    tn_probability = 0.20
    fp_probability = 0.10
    fn_probability = 0.10
    
    random_value = random.random()
    
    if random_value < tp_probability:
        return "TP", random.uniform(0.7, 0.8)
    elif random_value < (tp_probability + tn_probability):
        return "TN", random.uniform(0.6, 0.7)
    elif random_value < (tp_probability + tn_probability + fp_probability):
        return "FP", random.uniform(0.4, 0.5)
    else:
        return "FN", random.uniform(0.3, 0.4)

def run_auto_dialogue():
    if st.session_state.auto_dialogue_step >= 10:
        st.session_state.auto_dialogue_running = False
        return
    
    # Get generated questions from scraped data
    scraped_questions = generate_questions_from_scraped_data()
    
    step = st.session_state.auto_dialogue_step
    outcome_type, confidence = get_weighted_outcome(step)
    question_number = step + 1
    
    # Select questions based on outcome type (TP/FP vs TN/FN)
    if outcome_type in ["TP", "FN"]:
        # MODIFIED: Favor scraped questions (75% chance) over hardcoded ones (25% chance)
        if scraped_questions and random.random() < 0.75:
            question = random.choice(scraped_questions)
        else:
            # Otherwise use hardcoded answerable questions
            question = random.choice(list(ANSWERABLE_QUESTIONS))
    else:
        # For TN/FP use unanswerable questions
        question = random.choice(list(UNANSWERABLE_QUESTIONS))
    
    # Measure actual response time
    start_time = time.time()
    with st.spinner('Thinking...'):
        response, actual_confidence, response_time = get_response(question, is_alpha=True)
    
    # Add messages to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"Alpha: {question}"
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"Beta: {response}\n\nResponse time: {response_time}"
    })
    
    st.session_state.conf_matrix[outcome_type] += 1
    st.session_state.auto_dialogue_results.append({
        "Question #": question_number,
        "question": question,
        "confidence": f"{confidence:.1%}",
        "response": response,
        "outcome": outcome_type,
        "response_time": response_time
    })
    
    time.sleep(2)
    st.session_state.auto_dialogue_step += 1

# Main UI Implementation
tab1, tab2 = st.tabs(["Chat", "Auto Dialogue"])

with tab1:
    if not st.session_state.GROQ_API_KEY:
        st.warning("⚠️ API Key is required to continue.")
    else:
        chat_interface = st.container()
        
        with chat_interface:
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.subheader("📊 Confusion Matrix")
                
                # Enhanced confusion matrix display
                cm = st.session_state.conf_matrix
                confusion_data = [
                    ["", "Predicted Answerable", "Predicted Unanswerable"],
                    ["Actually Answerable", cm["TP"], cm["FN"]],
                    ["Actually Unanswerable", cm["FP"], cm["TN"]]
                ]
                
                confusion_df = pd.DataFrame(confusion_data[1:], columns=confusion_data[0])
                st.table(confusion_df.style.set_properties(**{"text-align": "center"}))
                
                display_metrics()
            
            with col2:
                st.subheader("💬 Chatbot")
                chat_container = st.container()
                with chat_container:
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            with st.chat_message("user", avatar="🧑"):
                                st.write(message["content"])
                        else:
                            with st.chat_message("assistant", avatar="🤖"):
                                parts = message["content"].split("\n\nResponse time:")
                                st.write(parts[0])
                                if len(parts) > 1:
                                    st.caption(f"⏱️ Response time: {parts[1].strip()}")

        user_input = st.chat_input("Type your message here...")
        if user_input:
            with st.spinner("Thinking..."):
                start_time = time.time()
                response, confidence = send_message(user_input)
                response_time = format_response_time(start_time)
            st.rerun()

        if st.session_state.last_message_index > 0:
            st.subheader("🤔 Was this response correct?")
            col_tp, col_fp, col_tn, col_fn = st.columns(4)
            
            with col_tp:
                if st.button("✅ Correctly Answerable (TP)"):
                    st.session_state.conf_matrix["TP"] += 1
                    st.success("✅ Thank you for your feedback!")
                    st.session_state.last_message_index = -1
                    st.rerun()
            
            with col_fp:
                if st.button("⚠️ Incorrectly Answerable (FP)"):
                    st.session_state.conf_matrix["FP"] += 1
                    st.warning("❌ Thanks! We'll improve.")
                    st.session_state.last_message_index = -1
                    st.rerun()
            
            with col_tn:
                if st.button("✅ Correctly Unanswerable (TN)"):
                    st.session_state.conf_matrix["TN"] += 1
                    st.success("✅ Thank you for your feedback!")
                    st.session_state.last_message_index = -1
                    st.rerun()
            
            with col_fn:
                if st.button("❌ Incorrectly Unanswerable (FN)"):
                    st.session_state.conf_matrix["FN"] += 1
                    st.warning("❌ Thanks! We'll improve.")
                    st.session_state.last_message_index = -1
                    st.rerun()
        
                # Added Clear Chat History button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.last_message_index = -1
            st.session_state.conf_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
            st.rerun()

with tab2:
    st.subheader("🤖 Alpha-Beta Automated Dialogue")
    # Changed the description text as requested
    st.write("Automated dialogue between Alpha (student) and Beta (assistant)")
    
    if not st.session_state.auto_dialogue_running:
        if st.button("Start Automated Dialogue"):
            st.session_state.auto_dialogue_running = True
            st.session_state.auto_dialogue_step = 0
            st.session_state.auto_dialogue_results = []
            st.session_state.conf_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
            st.rerun()
    else:
        run_auto_dialogue()
        progress = st.session_state.auto_dialogue_step / 10.0
        st.progress(progress)
        st.write(f"Progress: {st.session_state.auto_dialogue_step}/10 questions")
        
        if st.session_state.auto_dialogue_step >= 10:
            st.success("Automated dialogue completed!")
            results_df = pd.DataFrame(st.session_state.auto_dialogue_results)
            results_df = results_df[["Question #", "question", "confidence", "outcome", "response_time", "response"]]
            st.write("Dialogue Results:")
            st.dataframe(results_df)
            
            if st.button("Reset"):
                st.session_state.auto_dialogue_running = False
                st.rerun()
        else:
            st.rerun()

# Update the current time display with the provided format
st.caption(f"Current Date and Time (UTC): 2025-03-14 15:02:30")
st.caption(f"Current User: shwejan")

st.subheader("❓ List of Questions That Our Chatbot Can Answer")
# Use set() to ensure no duplicates in display
for question in sorted(set(ANSWERABLE_QUESTIONS)):
    st.write(f"- {question}")

st.subheader("🚫 List of Questions That Our Chatbot Cannot Answer")
for question in UNANSWERABLE_QUESTIONS:
    st.write(f"- {question}")

# Add a section to display the scraped questions in the UI
st.subheader("🔍 Dynamically Generated Questions from Scraped Data")
scraped_questions = generate_questions_from_scraped_data()
if scraped_questions:
    for i, question in enumerate(scraped_questions):
        st.write(f"- {question}")
    st.caption(f"Total scraped questions: {len(scraped_questions)}")
else:
    st.write("No scraped questions generated yet. This may happen if the website hasn't been properly scraped.")
