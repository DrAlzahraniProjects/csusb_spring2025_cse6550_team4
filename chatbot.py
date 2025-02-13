import os
import requests
import streamlit as st

# Fetch the Groq API key from the environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API Key not found. Please set it as an environment variable.")
    raise ValueError("Groq API Key not found. Please set it as an environment variable.")

# Define the Groq API URL
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Streamlit UI
st.title("Basic Chatbot with Llama 3")

# User input
user_input = st.text_input("You: ")

if user_input:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # Request data format based on Groq's docs
    data = {
        "model": "llama3-8b-8192",  # active model
        "messages": [{"role": "user", "content": user_input}],
        "max_tokens": 1000,
    }

    # Send the POST request to the Groq API
    response = requests.post(GROQ_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        # Extract and display chatbot's response
        chatbot_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Increase the height based on the length of the response, ensuring it can hold large content
        response_length = len(chatbot_response.splitlines())
        height = max(600, 20 + response_length * 20)  # dynamically adjust height based on content
        
        # Display response in a scrollable text area with a dynamic height
        st.text_area("Chatbot:", value=chatbot_response, height=height, max_chars=None)
    else:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
