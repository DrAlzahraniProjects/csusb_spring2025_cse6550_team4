import streamlit as st
from llama_cpp import Llama

# Load Llama model
llm = Llama(model_path="models/llama-2-7b.Q4_K_M.gguf")

# Streamlit UI
st.title("Basic Chatbot with Llama 3")

# User input
user_input = st.text_input("You: ")

if user_input:
    response = llm(user_input)
    st.write(f"Chatbot: {response}")
