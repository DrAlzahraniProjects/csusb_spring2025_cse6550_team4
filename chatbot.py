import streamlit as st
import requests
import pandas as pd
import numpy as np

# Streamlit UI Header
st.markdown("<h1 style='text-align: center; color: blue;'>CSUSB Team 4</h1>", unsafe_allow_html=True)
st.title("Basic Chatbot with Llama 3")

# Ask user for the Groq API Key (hidden input)
GROQ_API_KEY = st.text_input("Please enter your Groq API Key:", type="password")

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Check if API Key is provided
if not GROQ_API_KEY:
    st.warning("API Key is required to continue.")
else:
    # Define the Groq API URL
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    # Create a two-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        # Generate a sample confusion matrix (for demonstration purposes)
        actual = np.array(np.random.choice(["Positive", "Negative"], 100))
        predicted = np.array(np.random.choice(["Positive", "Negative"], 100))
        
        data = {
            "Actual \\ Predicted": ["Positive", "Negative"],
            "Positive": [
                np.sum((actual == "Positive") & (predicted == "Positive")),
                np.sum((actual == "Positive") & (predicted == "Negative"))
            ],
            "Negative": [
                np.sum((actual == "Negative") & (predicted == "Positive")),
                np.sum((actual == "Negative") & (predicted == "Negative"))
            ],
        }
        df = pd.DataFrame(data).set_index("Actual \\ Predicted")
        st.table(df)
    
    with col2:
        st.subheader("Chatbot")
        # Display chat history
        for entry in st.session_state.chat_history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
        
        # User input for chatbot conversation
        user_input = st.text_input("You:", key="user_input")
        
        if st.button("Send") and user_input:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            
            # Prepare chat history for API request
            messages = st.session_state.chat_history + [{"role": "user", "content": user_input}]
            
            # Request data format based on Groq's API docs
            data = {
                "model": "llama3-8b-8192",
                "messages": messages,
                "max_tokens": 1000,
            }
            
            # Send the POST request to the Groq API
            response = requests.post(GROQ_API_URL, json=data, headers=headers)
            
            if response.status_code == 200:
                response_data = response.json()
                choices = response_data.get("choices", [])
                chatbot_response = choices[0].get("message", {}).get("content", "") if choices else "No response received."
                
                # Append user and bot messages to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": chatbot_response})
                
                # Clear input field safely
                st.session_state.pop("user_input", None)
                st.rerun()
            else:
                st.error(f"API request failed with status code {response.status_code}: {response.text}")
