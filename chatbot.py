import streamlit as st
import requests
import pandas as pd

# Streamlit UI Header
st.markdown("<h1 style='text-align: center; color: blue;'>CSUSB Team 4</h1>", unsafe_allow_html=True)
st.title("Basic Chatbot with Llama 3")

# Ask user for the Groq API Key (hidden input)
GROQ_API_KEY = st.text_input("Please enter your Groq API Key:", type="password")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = {
        "Positive": {"Positive": 0, "Negative": 0},
        "Negative": {"Positive": 0, "Negative": 0}
    }

if "last_message_index" not in st.session_state:
    st.session_state.last_message_index = -1  # Default to -1 when there's no message

if "user_input" not in st.session_state:
    st.session_state.user_input = ""  # Initialize user input state

# Function to handle input submission
def send_message():
    user_input = st.session_state.user_input.strip()
    
    if not user_input:
        return  # Don't send empty messages
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Prepare chat history for API request
    messages = st.session_state.chat_history + [{"role": "user", "content": user_input}]
    
    # Request data format based on Groq's API docs
    data = {
        "model": "llama3-8b-8192",  # Ensure correct model is used
        "messages": messages,
        "max_tokens": 1000,
    }
    
    # Send the POST request to the Groq API
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    
    chatbot_response = "No response received."
    if response.status_code == 200:
        response_data = response.json()
        choices = response_data.get("choices", [])
        if choices and "message" in choices[0]:
            chatbot_response = choices[0]["message"].get("content", "No response received.")
    else:
        chatbot_response = f"Error: {response.status_code} - {response.text}"
    
    # Append user and bot messages to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": chatbot_response})
    
    # Store the latest message index for feedback tracking
    st.session_state.last_message_index = len(st.session_state.chat_history)

    # Clear user input
    st.session_state.user_input = ""  
    st.rerun()

# Check if API Key is provided
if not GROQ_API_KEY:
    st.warning("API Key is required to continue.")
else:
    # Create a two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        # Display the current confusion matrix
        data = {
            "Actual \\ Predicted": ["Positive", "Negative"],
            "Positive": [st.session_state.conf_matrix["Positive"]["Positive"], st.session_state.conf_matrix["Positive"]["Negative"]],
            "Negative": [st.session_state.conf_matrix["Negative"]["Positive"], st.session_state.conf_matrix["Negative"]["Negative"]],
        }
        df = pd.DataFrame(data).set_index("Actual \\ Predicted")
        st.table(df)
    
    with col2:
        st.subheader("Chatbot")
        
        # Display chat history (show last 5 messages for a clean UI)
        with st.container():
            for entry in st.session_state.chat_history[-5:]:  
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])
        
        # User input for chatbot conversation (on_change ensures input is cleared correctly)
        st.text_input("You:", key="user_input", on_change=send_message)

# Feedback section (only show if a chatbot response was given)
if st.session_state.last_message_index > 0:
    st.subheader("Was this response helpful?")
    
    col_yes, col_no = st.columns(2)

    with col_yes:
        if st.button("Yes"):
            st.session_state.conf_matrix["Positive"]["Positive"] += 1  # Correct positive response
            st.success("✅ Feedback: Answer was satisfactory!")
            st.session_state.last_message_index = -1  # Reset tracking
            st.rerun()

    with col_no:
        if st.button("No"):
            st.session_state.conf_matrix["Positive"]["Negative"] += 1  # Incorrect positive response
            st.warning("❌ Feedback: Answer was not satisfactory!")
            st.session_state.last_message_index = -1  # Reset tracking
            st.rerun()
