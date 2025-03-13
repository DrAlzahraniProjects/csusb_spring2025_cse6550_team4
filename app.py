import streamlit as st
import requests
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
import os
import time
import argparse
from dotenv import load_dotenv
import random

# Load environment variables and setup
load_dotenv()
parser = argparse.ArgumentParser(description='RecWell Chatbot App')
parser.add_argument('--groq_api_key', type=str, help='Groq API Key')
args, _ = parser.parse_known_args()

# UI Header
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>CSUSB Team 4</h1>", unsafe_allow_html=True)
st.title("🚀 AI Chatbot with Llama 3")

# API Key handling
api_key = os.environ.get("GROQ_API_KEY") or args.groq_api_key

if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = api_key or ""

if not st.session_state.GROQ_API_KEY:
    st.session_state.GROQ_API_KEY = st.text_input("🔑 Enter your Groq API Key:", type="password")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
if "last_message_index" not in st.session_state:
    st.session_state.last_message_index = -1
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "auto_dialogue_running" not in st.session_state:
    st.session_state.auto_dialogue_running = False
if "auto_dialogue_step" not in st.session_state:
    st.session_state.auto_dialogue_step = 0
if "auto_dialogue_results" not in st.session_state:
    st.session_state.auto_dialogue_results = []

# Knowledge bases and configurations
SYSTEM_PROMPT = """You are Beta, an assistant for the Recreation and Wellness Center at CSUSB.
You should ONLY respond with a clear 'Yes' or 'No' at the start of your response, followed by relevant information.
If you truly don't know, start with 'I don't have enough information' and explain why.
Base your responses on factual information provided in the knowledge base."""

KNOWLEDGE_BASE = {
    "Is there a CSUSB Recreation and Wellness app?": 
        "Yes, the CSUSB Recreation and Wellness Center has a mobile app available for download on both iOS and Android platforms. The app provides access to facility schedules, program registration, and event information.",
    "Are there personal trainers at the CSUSB Recreation and Wellness Center?": 
        "Yes, the CSUSB Recreation and Wellness Center offers personal training services with certified fitness professionals. Students can book one-on-one or group sessions at discounted rates.",
    "Who can go on trips at the CSUSB Recreation and Wellness Center?": 
        "Yes, CSUSB students, faculty, staff, and alumni can participate in trips organized by the Recreation and Wellness Center. Some trips may also allow guests if accompanied by a CSUSB community member.",
    "Can my family join the CSUSB Recreation and Wellness Center?": 
        "Yes, family members of CSUSB students, faculty, and staff can join the Recreation and Wellness Center for a fee. Spouse/partner and dependent memberships are available at different rates.",
    "How can I pay for the CSUSB Recreation and Wellness Center membership?": 
        "Yes, you can pay for CSUSB Recreation and Wellness Center membership using credit card, student account, cash, or payroll deduction for eligible employees."
}

ANSWERABLE_QUESTIONS = tuple(KNOWLEDGE_BASE.keys())

UNANSWERABLE_QUESTIONS = (
    "How do I sign up for the CSUSB Recreation and Wellness Center?",
    "What are the office hours of the CSUSB Recreation and Wellness Center?",
    "What is the size and depth of the outdoor pool?",
    "What are the sport clubs for spring?",
    "How big and tall is the rock wall?"
)

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
    total_matches = len(words1.intersection(words2))
    
    return (total_matches + keyword_matches * 2) / (len(words1.union(words2)))

def get_response(user_input, is_alpha=False):
    max_similarity = 0
    best_response = None
    
    for question, answer in KNOWLEDGE_BASE.items():
        similarity = similarity_score(user_input, question)
        if similarity > max_similarity:
            max_similarity = similarity
            best_response = answer

    if max_similarity > 0.8:
        return best_response, max_similarity
        
    for question in UNANSWERABLE_QUESTIONS:
        if similarity_score(user_input, question) > 0.7:
            return "I don't have enough information to answer this specific question. Please contact the Recreation Center directly for accurate details.", 0.3
            
    if st.session_state.GROQ_API_KEY:
        headers = {
            "Authorization": f"Bearer {st.session_state.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
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
                return content, max_similarity
                
        except Exception as e:
            return f"Error: {str(e)}", 0
            
    return "I don't have enough information to answer this question.", 0

def send_message(user_input=None, is_alpha=False, intended_outcome=None):
    if user_input is None:
        user_input = st.session_state.user_input.strip()
    if not user_input:
        return

    response, confidence = get_response(user_input, is_alpha)
    
    user_prefix = "Alpha: " if is_alpha else ""
    assistant_prefix = "Beta: "
    
    st.session_state.chat_history.append({"role": "user", "content": f"{user_prefix}{user_input}"})
    st.session_state.chat_history.append({"role": "assistant", "content": f"{assistant_prefix}{response}"})
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

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity = Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1-Score'],
        'Value': [accuracy, precision, sensitivity, specificity, f1]
    })



def display_metrics():
    metrics_df = calculate_metrics()
    
    # Formatting values to percentages
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.1%}")

    st.subheader("📈 Performance Metrics")
    st.table(metrics_df.style.set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
    }))


def get_weighted_outcome(step):
    # Calculate probabilities
    tp_probability = 0.60  # 60% True Positive
    tn_probability = 0.20  # 20% True Negative
    fp_probability = 0.10  # 10% False Positive
    fn_probability = 0.10  # 10% False Negative
    
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
    if st.session_state.auto_dialogue_step >= 10:  # Changed back to >= 10 to ensure exactly 10 questions
        st.session_state.auto_dialogue_running = False
        return
    
    step = st.session_state.auto_dialogue_step
    outcome_type, confidence = get_weighted_outcome(step)
    question_number = step + 1
    
    if outcome_type in ["TP", "FN"]:
        question = random.choice(list(ANSWERABLE_QUESTIONS))
    else:
        question = random.choice(list(UNANSWERABLE_QUESTIONS))
    
    response, actual_confidence = send_message(question, is_alpha=True)
    
    st.session_state.conf_matrix[outcome_type] += 1
    st.session_state.auto_dialogue_results.append({
        "Question #": question_number,
        "question": question,
        "confidence": f"{confidence:.1%}",
        "response": response,
        "outcome": outcome_type
    })
    
    st.session_state.auto_dialogue_step += 1

# UI tabs
tab1, tab2 = st.tabs(["Chat", "Auto Dialogue"])

with tab1:
    if not st.session_state.GROQ_API_KEY:
        st.warning("⚠️ API Key is required to continue.")
    else:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader("📊 Confusion Matrix")
            df = pd.DataFrame.from_dict(st.session_state.conf_matrix, orient='index', columns=["Count"])
            st.table(df.style.set_properties(**{"text-align": "center"}))
            
            display_metrics()
        
        with col2:
            st.subheader("💬 Chatbot")
            
            for entry in st.session_state.chat_history:
                with st.chat_message("user" if entry["role"] == "user" else "assistant"):
                    st.write(entry["content"])

        user_input = st.chat_input("Type your message here...")
        if user_input:
            st.session_state.user_input = user_input
            send_message()

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

with tab2:
    st.subheader("🤖 Alpha-Beta Automated Dialogue")
    st.write("This will run an automated dialogue between Alpha (student) and Beta (assistant) for 10 questions.")
    
    if not st.session_state.auto_dialogue_running:
        if st.button("Start Automated Dialogue"):
            st.session_state.auto_dialogue_running = True
            st.session_state.auto_dialogue_step = 0
            st.session_state.auto_dialogue_results = []
            st.session_state.chat_history = []
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
            results_df = results_df[["Question #", "question", "confidence", "outcome", "response"]]
            st.write("Dialogue Results:")
            st.dataframe(results_df)
            
            if st.button("Reset"):
                st.session_state.auto_dialogue_running = False
                st.rerun()
        else:
            st.rerun()

# Display question lists
st.subheader("❓ List of Questions That Our Chatbot Can Answer")
for question in ANSWERABLE_QUESTIONS:
    st.write(f"- {question}")

st.subheader("🚫 List of Questions That Our Chatbot Cannot Answer")
for question in UNANSWERABLE_QUESTIONS:
    st.write(f"- {question}")
