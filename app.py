import streamlit as st
import requests
import pandas as pd

# Streamlit UI Header
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>CSUSB Team 4</h1>", unsafe_allow_html=True)
st.title("🚀 AI Chatbot with Llama 3")

# Store API Key in Session State
if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = ""

st.session_state.GROQ_API_KEY = st.text_input("🔑 Enter your Groq API Key:", type="password", value=st.session_state.GROQ_API_KEY)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
if "last_message_index" not in st.session_state:
    st.session_state.last_message_index = -1
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def send_message():
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return  # Prevent empty messages
    
    if not st.session_state.GROQ_API_KEY:
        st.warning("⚠️ API Key is required to send messages.")
        return
    
    headers = {"Authorization": f"Bearer {st.session_state.GROQ_API_KEY}", "Content-Type": "application/json"}
    messages = st.session_state.chat_history + [{"role": "user", "content": user_input}]
    
    data = {"model": "llama3-8b-8192", "messages": messages, "max_tokens": 1000}
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    
    chatbot_response = "No response received."
    if response.status_code == 200:
        response_data = response.json()
        choices = response_data.get("choices", [])
        if choices and "message" in choices[0]:
            chatbot_response = choices[0]["message"].get("content", "No response received.")
    else:
        chatbot_response = f"⚠️ Error {response.status_code}: Unable to get a response. Please check your API key and try again."
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": chatbot_response})
    st.session_state.last_message_index = len(st.session_state.chat_history)
    st.session_state.user_input = ""
    st.rerun()

if not st.session_state.GROQ_API_KEY:
    st.warning("⚠️ API Key is required to continue.")
else:
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("📊 Confusion Matrix")
        df = pd.DataFrame.from_dict(st.session_state.conf_matrix, orient='index', columns=["Count"])
        st.table(df.style.set_properties(**{"text-align": "center"}))
    
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

# Compute Evaluation Metrics
st.subheader("📈 Performance Metrics")
TP, FP, TN, FN = st.session_state.conf_matrix.values()
accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) else 0
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": [accuracy, precision, recall, f1_score]
})
st.table(metrics_df.style.set_properties(**{"text-align": "center"}))
