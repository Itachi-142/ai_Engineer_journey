import streamlit as st
import os
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from tutor_engine import explain_concept, socratic_chat

st.set_page_config(page_title="AI Learning Tutor", page_icon="🎓")
st.title("🎓 AI Learning Tutor")

mode = st.radio("Choose mode:", ["Concept Explanation", "Socratic Tutor"])

st.divider()

# --- CONCEPT EXPLANATION MODE ---
if mode == "Concept Explanation":
    st.subheader("📘 Concept Explanation")
    topic = st.text_input("Enter a concept to explain:", placeholder="e.g. gradient descent, binary search, transformers")
    
    if st.button("Explain"):
        if topic.strip():
            with st.spinner("Thinking..."):
                result = explain_concept(topic)
            st.markdown(result)
        else:
            st.warning("Please enter a concept first.")

# --- SOCRATIC TUTOR MODE ---
elif mode == "Socratic Tutor":
    st.subheader("🧠 Socratic Tutor")
    st.caption("I will never give you the answer directly. I will guide you there.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "display_history" not in st.session_state:
        st.session_state.display_history = []

    # display chat
    for msg in st.session_state.display_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question or describe a problem...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, st.session_state.chat_history = socratic_chat(
                    user_input,
                    st.session_state.chat_history
                )
            st.markdown(response)

        st.session_state.display_history.append({"role": "user", "content": user_input})
        st.session_state.display_history.append({"role": "assistant", "content": response})

    if st.button("🔄 Reset conversation"):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()