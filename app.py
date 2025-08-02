import streamlit as st
from rag_utils import get_answer

# Page config
st.set_page_config(
    page_title="Cycle AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# App title and description
st.title("🤖 Cycle AI Assistant")
st.markdown("Your personal companion for menstrual health questions and guidance.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your cycle..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from RAG system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, updated_history = get_answer(prompt, st.session_state.chat_history)
                st.session_state.chat_history = updated_history
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                if "api_key" in str(e).lower():
                    error_msg = "⚠️ API key configuration issue. Please check your .env file and ensure GROQ_API_KEY is set."
                else:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"

                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with information
with st.sidebar:
    st.header("About Cycle AI")
    st.markdown("""
    This AI assistant helps you understand:
    - Menstrual cycle patterns
    - Period symptoms and management
    - When to consult healthcare providers
    - General reproductive health guidance
    
    💡 **Tip**: Ask specific questions for better answers!
    
    ⚙️ **Setup**: Only requires:
    - GROQ_API_KEY in your .env file
    
    🤖 **Powered by**: GROQ + HuggingFace (free embeddings)
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
