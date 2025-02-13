import requests
import streamlit as st

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: blue;'>CSUSB Team 4</h1>", unsafe_allow_html=True)
st.title("Basic Chatbot with Llama 3")

# Input field for API key
api_key = st.text_input("Enter your Groq API Key:", type="password")

# User input
user_input = st.text_input("You: ")

if api_key and user_input:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Request data format based on Groq's docs
    data = {
        "model": "llama3-8b-8192",  # active model
        "messages": [{"role": "user", "content": user_input}],
        "max_tokens": 1000,
    }

    # Send the POST request to the Groq API
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        chatbot_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Adjust response box height dynamically
        response_length = len(chatbot_response.splitlines())
        height = max(600, 20 + response_length * 20)
        
        # Display chatbot response
        st.text_area("Chatbot:", value=chatbot_response, height=height, max_chars=None)
    else:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
elif not api_key:
    st.warning("Please enter your Groq API Key.")
