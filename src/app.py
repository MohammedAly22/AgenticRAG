import os
import tempfile

from langchain_cohere import CohereEmbeddings
from langchain_core.messages import HumanMessage

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

# local
from utils import (
    load_pdf,
    create_embeddings,
    split_documents,
    build_vector_store,
    process_query,
    process_chunks_with_rate_limit_cohere,
    add_chunks_to_vector_store_hf_embeddings,
    reset_memory,
)


# Page configuration
st.set_page_config(
    page_title="AI Index Report 2025 - RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Set custom CSS styling
with open("src/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main title
st.title("🤖 AI Index Report 2025 - RAG Agent")
st.markdown(
    "Ask questions about the AI Index Report 2025 and get intelligent responses."
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "rendered_pages" not in st.session_state:
    st.session_state.rendered_pages = None

if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = False

if "rag_agent_executer" not in st.session_state:
    st.session_state.rag_agent_executer = None

if "docs" not in st.session_state:
    st.session_state.docs = []


# Load message avatars
mohammed_avatar = "docs/Mohammed_avatar.png"
agent_avatar = "docs/agent_avatar.png"


# Sidebar for API key and settings
with st.sidebar:
    st.header("📝 1. Configuration")

    # 1. Cohere API Key
    cohere_api_key = st.text_input(
        "Enter Cohere API Key",
        type="password",
        placeholder="Enter you Cohere API key",
    )
    if cohere_api_key:
        os.environ["COHERE_API_KEY"] = cohere_api_key

    # 2. Select an embedding model
    embeddings_type = st.selectbox(
        "Select an Embedding Model",
        [
            "cohere/embed-v4.0",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        help="**Note**: The `cohere/embed-v4.0` model, when used with a `trial_key`, "
        "is limited to processing **100,000** tokens per minute. This rate limit may "
        "cause **slower** processing for large documents due to enforced waiting "
        "between batches. However, despite the slower throughput, it is much more "
        "efficient and accurate compared to `sentence-transformers/all-mpnet-base-v2`"
        "especially for high-quality semantic embeddings.",
    )

    # Show reasoning steps toggle
    st.session_state.show_reasoning = st.toggle(
        "Show Reasoning Steps",
        st.session_state.show_reasoning,
        help="Controls whether to display the agent's intermediate reasoning "
        "steps—such as decisions and tool calls. When disabled, only the final "
        "answer will be shown.",
    )

    st.subheader("📄 2. Document Upload")

    # File uploader
    uploaded_file = st.file_uploader("Upload Your PDF", type="pdf")

    if uploaded_file and (
        st.session_state.uploaded_file is None
        or uploaded_file.name != st.session_state.uploaded_file.name
    ):
        with st.spinner("Processing PDF..."):
            st.session_state.uploaded_file = uploaded_file

            # Create a temporary file to save the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            # Load the PDF
            docs = load_pdf(pdf_path)
            st.session_state.docs = docs

            if cohere_api_key:
                with st.spinner("Splitting PDF..."):
                    # Create embeddings and split the documents
                    embeddings = create_embeddings(
                        embeddings_type, cohere_api_key=cohere_api_key
                    )
                    st.success("✅ Embeddings created successfully!")
                    chunks = split_documents(docs)

                st.markdown(
                    f"📄 Found a total `{len(docs)}` pages splitted into `{len(chunks)}` chunks"
                )

                # Build vector store
                vector_store = build_vector_store(embeddings)

                # Add documents to vectorstore
                if isinstance(embeddings, CohereEmbeddings):
                    vector_store = process_chunks_with_rate_limit_cohere(
                        chunks, vector_store
                    )
                else:
                    vector_store = add_chunks_to_vector_store_hf_embeddings(
                        chunks, vector_store
                    )

                st.session_state.vector_store = vector_store

            else:
                st.error("⚠️ Please enter your Cohere API key to process the document.")

            # Clean up the temporary file
            os.unlink(pdf_path)

    # Number of rendered pages
    number_of_rendered_page = st.slider(
        "Number of PDF Rendered Pages",
        min_value=10,
        max_value=min(len(st.session_state.docs), 100),
        value=30,
        help="Limits the number of previewed pages from the uploaded "
        "PDF to improve performance, as rendering more pages takes "
        "longer. A maximum of `100` pages can be previewed.",
    )
    st.session_state.rendered_pages = number_of_rendered_page

    # Show a status indicator for document loading
    if st.session_state.vector_store is not None:
        st.sidebar.success("✅ Document loaded and indexed successfully!")
    else:
        st.sidebar.warning("⚠️ No document loaded")

    # Clear conversation button
    st.subheader("🗑️ 3. Clear Chat History (Optional)")
    if st.button("Clear Conversation"):
        if len(st.session_state.messages):
            st.session_state.messages = []
            st.success("✅ Conversation cleared successfully!")
            reset_memory()
        else:
            st.info("There are no messages to clear!")


# Disprning if necessary components are missing
if not cohere_api_key:
    st.warning("⚠️ Please enter your Cohere API key in the sidebar to continue.")

elif not st.session_state.vector_store:
    st.warning(
        "⚠️ Please upload the AI Index Report 2025 PDF in the sidebar to continue."
    )


pdf_preview, chat_area = st.columns(2)

# PDF preview container
with pdf_preview:

    if uploaded_file:
        with st.container(border=True):
            binary_data = uploaded_file.getvalue()
            pdf_viewer(
                input=binary_data,
                height=600,
                pages_to_render=[*range(st.session_state.rendered_pages)],
                resolution_boost=2,
                pages_vertical_spacing=1,
                render_text=True,
            )

# Chat container
if st.session_state.vector_store:
    with chat_area:
        chat_container = st.container(height=585, border=False)

        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user", avatar=mohammed_avatar):
                        st.markdown(message.content)
                else:
                    with st.chat_message("assistant", avatar=agent_avatar):
                        st.markdown(message.content)

        # Chat input
        query = st.chat_input("Ask a question about the AI Index Report 2025...")

        if query:
            # Display user message
            with chat_container:
                with st.chat_message("user", avatar=mohammed_avatar):
                    st.markdown(query)

                # user message to session state
                st.session_state.messages.append(HumanMessage(content=query))

                # Process the query
                process_query(query, cohere_api_key, agent_avatar)
