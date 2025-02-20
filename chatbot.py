import streamlit as st
import requests
import pandas as pd

# Streamlit UI Header
st.markdown("<h1 style='text-align: center; color: blue;'>CSUSB Team 4</h1>", unsafe_allow_html=True)
st.title("Basic Chatbot with Llama 3")

# Ask user for the Groq API Key (hidden input)
GROQ_API_KEY = st.text_input("Please enter your Groq API Key:", type="password")

# Initialize chat history and confusion matrix in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conf_matrix" not in st.session_state:
    st.session_state.conf_matrix = {
        "Positive": {"Positive": 0, "Negative": 0},
        "Negative": {"Positive": 0, "Negative": 0}
    }

# Initialize user input state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Ensure last message index is tracked for feedback
if "last_message_index" not in st.session_state:
    st.session_state.last_message_index = -1  # Default to -1 when there's no message

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
        
        # Create a fixed container for chat history
        with st.container():
            # Display chat history
            for entry in st.session_state.chat_history:
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])
        
        # User input for chatbot conversation (cleared after sending)
        user_input = st.text_input("You:", st.session_state.user_input)

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
                
                # Store the latest message index for feedback tracking
                st.session_state.last_message_index = len(st.session_state.chat_history)

                # Clear the input field by resetting the user input session state
                st.session_state.user_input = ""

                # Force a rerun to update the UI (this clears the input field)
                st.rerun()

# Feedback section (only show if a chatbot response was given)
if st.session_state.last_message_index > 0:
    st.subheader("Was this response helpful?")
    
    col_yes, col_no = st.columns(2)

    with col_yes:
        if st.button("Yes"):
            st.session_state.conf_matrix["Positive"]["Positive"] += 1  # Correct positive response
            st.success("✅ Feedback: Answer was satisfactory!")
            st.rerun()  # Rerun to refresh UI

    with col_no:
        if st.button("No"):
            st.session_state.conf_matrix["Negative"]["Negative"] += 1  # Incorrect response
            st.warning("❌ Feedback: Answer was not satisfactory!")
            st.rerun()  # Rerun to refresh UI
