"""
Streamlit Chat UI for GPU Optimize Demo.

Connects to the OpenAI-compatible serving endpoint.

Usage:
    streamlit run examples/chat_ui.py
"""

import streamlit as st
import requests
import json

# Page Config
st.set_page_config(
    page_title="LLM Chat Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GPU Optimize Demo: Chat")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", "http://localhost:8000/v1/chat/completions")
    model_name = st.text_input("Model Name", "gpu-optimized-model")
    
    st.divider()
    
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 10, 2048, 512, 10)
    
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Say something..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare Payload
        payload = {
            "model": model_name,
            "messages": st.session_state.messages,
            "stream": True, # Enable Streaming
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            with requests.post(api_url, json=payload, stream=True) as response:
                if response.status_code != 200:
                    st.error(f"Error: {response.status_code} - {response.text}")
                else:
                    # Process SSE Stream
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    delta = data["choices"][0]["delta"]
                                    if "content" in delta:
                                        content = delta["content"]
                                        full_response += content
                                        message_placeholder.markdown(full_response + "▌")
                                except json.JSONDecodeError:
                                    continue
                                    
            # Final Update
            message_placeholder.markdown(full_response)
            
            # Add to History
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Connection Error: {e}")
