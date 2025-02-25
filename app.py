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
    st.session_state.conf_matrix = {"Positive": {"Positive": 0, "Negative": 0}, "Negative": {"Positive": 0, "Negative": 0}}
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
        df = pd.DataFrame({
            "Actual \\ Predicted": ["Positive", "Negative"],
            "Positive": [st.session_state.conf_matrix["Positive"]["Positive"], st.session_state.conf_matrix["Positive"]["Negative"]],
            "Negative": [st.session_state.conf_matrix["Negative"]["Positive"], st.session_state.conf_matrix["Negative"]["Negative"]],
        }).set_index("Actual \\ Predicted")
        st.table(df.style.set_properties(**{"text-align": "center"}))
    
    with col2:
        st.subheader("💬 Chatbot")
        
        for entry in st.session_state.chat_history:
            with st.chat_message("user" if entry["role"] == "user" else "assistant"):
                st.write(entry["content"])

    # Fix input box in the center with space from bottom, and include send button inside the box
    st.markdown("""
        <style>
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 20px;  /* Adjust the distance from the bottom */
            left: 50%;
            transform: translateX(-50%);  /* Center the input horizontally */
            width: 80%;  /* Adjust the width of the input box */
            background: white;
            padding: 10px 0;
            z-index: 999;
            display: flex;
            align-items: center;
        }
        div[data-testid="stChatInput"] input {
            width: 100%;  /* Ensure the input field takes the available space */
            padding: 10px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        div[data-testid="stChatInput"] button {
            margin-left: 10px;
            padding: 10px;
            background-color: #4A90E2;
            color: white;
            border-radius: 50%;
            border: none;
            cursor: pointer;
        }
        </style>
    """, unsafe_allow_html=True)
    
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.user_input = user_input
        send_message()

if st.session_state.last_message_index > 0:
    st.subheader("🤔 Was this response helpful?")
    col_yes, col_no = st.columns(2)
    
    with col_yes:
        if st.button("👍 Yes"):
            st.session_state.conf_matrix["Positive"]["Positive"] += 1
            st.success("✅ Thank you for your feedback!")
            st.session_state.last_message_index = -1
            st.rerun()
    
    with col_no:
        if st.button("👎 No"):
            st.session_state.conf_matrix["Positive"]["Negative"] += 1
            st.warning("❌ Thanks! We'll improve.")
            st.session_state.last_message_index = -1
            st.rerun()
