import streamlit as st
from rag_engine import RAGChatbot
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip()

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.caption("Upload a document and ask questions about it")

# Initialize chatbot in session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot(api_key=GROQ_API_KEY)
    st.session_state.messages = []
    st.session_state.doc_loaded = False

# Sidebar — document input
with st.sidebar:
    st.header("📄 Document")

    input_method = st.radio(
        "Input method:",
        ["Paste Text", "Upload File"]
    )

    if input_method == "Paste Text":
        doc_text = st.text_area(
            "Paste your document here:",
            height=300,
            placeholder="Paste any text document here..."
        )
        if st.button("Load Document", type="primary"):
            if doc_text.strip():
                with st.spinner("Processing document..."):
                    num_chunks = st.session_state.chatbot.load_document(
                        doc_text
                    )
                st.session_state.doc_loaded = True
                st.success(f"✅ Document loaded — {num_chunks} chunks created")
            else:
                st.error("Please paste some text first")

    else:  # Upload File
        uploaded_file = st.file_uploader(
            "Upload a document:",
            type=["txt", "pdf"]
        )

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

        if st.button("Load Document", type="primary"):
            if "uploaded_file" not in st.session_state:
                st.error("Please upload a file first")
            else:
                with st.spinner("Processing document..."):
                    uploaded = st.session_state.uploaded_file
                    if uploaded.type == "application/pdf":
                        import fitz
                        pdf_bytes = uploaded.read()
                        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
                        text = ""
                        for page in pdf:
                            text += page.get_text()
                    else:
                        text = uploaded.read().decode("utf-8")

                    num_chunks = st.session_state.chatbot.load_document(text)
                st.session_state.doc_loaded = True
                st.success(f"✅ Document loaded — {num_chunks} chunks created")
    st.divider()

    if st.button("🔄 Reset Chat"):
        st.session_state.chatbot.reset()
        st.session_state.messages = []
        st.session_state.doc_loaded = False
        st.rerun()

    if st.session_state.doc_loaded:
        st.success("📚 Document ready")
    else:
        st.warning("⚠️ No document loaded")

# Main chat area
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, sources = st.session_state.chatbot.chat(prompt)
        st.write(response)

        # Show sources
        if sources:
            with st.expander("📚 Sources"):
                for i, doc in enumerate(sources, 1):
                    st.caption(f"Chunk {i}: {doc.page_content[:150]}...")

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })