import os
import requests
import streamlit as st
# Streamlit UI for asking API key
st.markdown("<h1 style='text-align: center; color: blue;'>CSUSB Team 4</h1>", unsafe_allow_html=True)
st.title("Basic Chatbot with Llama 3")
# Ask user for the Groq API Key
GROQ_API_KEY = st.text_input("Please enter your Groq API Key:")
# Check if API Key is provided
if not GROQ_API_KEY:
    st.warning("API Key is required to continue.")
else:
    # Define the Groq API URL
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    # User input for chatbot conversation
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
            # Increase the height based on the length of the response
            response_length = len(chatbot_response.splitlines())
            height = max(600, 20 + response_length * 20)  # dynamically adjust height based on content
            # Display response in a scrollable text area with a dynamic height
            st.text_area("Chatbot:", value=chatbot_response, height=height, max_chars=None)
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
