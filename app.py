import streamlit as st
from rag_module import extract_and_chunk_pdf, embed_chunks, query_with_groq_llama3

st.set_page_config(page_title="📄 PDF RAG App", layout="centered")
st.title("🔍 PDF Question Answering using GROQ + FAISS")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Only reprocess if the file changes
    if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        with st.spinner("📖 Reading and chunking the PDF..."):
            chunks = extract_and_chunk_pdf(uploaded_file)

        with st.spinner("🔎 Embedding and indexing..."):
            index, chunk_data = embed_chunks(chunks)

        # Store for later use
        st.session_state.index = index
        st.session_state.chunk_data = chunk_data
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.chat_history = []  # Reset history on new file

    # Chat interface
    query = st.text_input("💬 Ask a question based on the uploaded PDF:")

    if st.button("🧠 Answer"):
        if query.strip():
            with st.spinner("🤖 Querying LLaMA 3 via GROQ..."):
                answer = query_with_groq_llama3(query, st.session_state.index, st.session_state.chunk_data)

            # Save to history
            st.session_state.chat_history.append((query, answer))

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("🗂️ Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}: {q}**")
            st.markdown(f"> {a}")
