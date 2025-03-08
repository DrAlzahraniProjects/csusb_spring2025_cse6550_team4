import os
import json
import scrapy
import streamlit as st
from bs4 import BeautifulSoup
import unicodedata
from dotenv import load_dotenv
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("🚨 GROQ_API_KEY is missing!")
    st.stop()

# Initialize Llama 3 model
llm = ChatGroq(model_name="llama3-8b-8192", api_key=api_key)

# Initialize embeddings model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Scrapy Spider for Web Scraping === #
class ContentSpider(scrapy.Spider):
    name = "content"
    start_urls = ["https://www.csusb.edu/recreation-wellness"]  # Replace with your target website

    def parse(self, response):
        for paragraph in response.css("p"):
            text = paragraph.get()
            clean_text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
            yield {"text": clean_text.strip()}

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

def load_scraped_data(file_path="scraped_data.json"):
    """Load scraped data, clean and process it before adding to FAISS."""
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r") as file:
        data = json.load(file)

    cleaned_texts = [
        normalize_text(item["text"])
        for item in data if len(item["text"].strip()) > 20  # Remove short/irrelevant text
    ]

    unique_texts = list(set(cleaned_texts))  # Remove duplicates
    return [Document(page_content=text) for text in unique_texts]

# === Run Scrapy at Startup if Needed === #
if not os.path.exists("scraped_data.json"):
    run_scrapy_spider()

# Create FAISS index from cleaned scraped data
documents = load_scraped_data()
vectorstore = FAISS.from_documents(documents, embedding_function) if documents else None

# Function to retrieve relevant docs from FAISS
def retrieve_relevant_docs(query):
    if vectorstore is None:
        return "No documents in FAISS yet. Try restarting the app."
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

# === Streamlit Chatbot UI === #
st.title("RAG Chatbot with Web Scraping")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Ask me something..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    retrieved_docs = retrieve_relevant_docs(user_input)
    prompt = f"Context:\n{retrieved_docs}\n\nUser: {user_input}"
    bot_response = llm.invoke(prompt).content.strip()

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # Save chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": bot_response})






 
   
 
         
     
            
           
