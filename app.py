# streamlit_app.py
import os
import math
import streamlit as st
import uuid
from dotenv import load_dotenv
from huggingface_hub import login
from core.state import RuntimeState
from configs.weaviate import *
from core.llms.weaviate.client import get_weaviate_client
from core.llms.weaviate.collections.chunk_ops import create_chunks_collection
from core.llms.weaviate.retrieve import retrieve_and_process_top_chunks
from core.llms.weaviate.ingest import ingest_text
from fast_api.api_models import LLMInput


load_dotenv()
login(os.environ["HUGGINGFACE_APIKEY"])

@st.cache_resource
def get_cached_client():
    return get_weaviate_client()

def run_app(rs: RuntimeState):
    client = get_cached_client()
    create_chunks_collection(client, collection_name)

    st.set_page_config(page_title="abysalto", layout="centered")
    st.title("AI-driven Document Insight Service")

    # ---------- File Upload ----------
    uploaded_files = st.file_uploader(
        "Upload images or PDFs",
        type=["png", "pdf"],
        accept_multiple_files=True
    )
    if uploaded_files:
        rs.uploaded_files = uploaded_files

    if st.button("Extract texts"):
        extracted_texts = rs.api.upload(uploaded_files)
        for text in extracted_texts:
            full_text = text['text']
            if not full_text:
                continue

            ingest_text(rs, client, full_text, collection_name)

    st.divider()
    st.header("Ask a question")
    query = st.text_area("Your question:", "Hello World")

    col1, col2, col3 = st.columns(3)
    with col1:
        top_chunks = st.slider("Top Chunks", 1, 10, 2,
                            help="Number of best-matching chunks to feed into the LLM")
    with col2:
        window_len_perc = st.slider("Sliding Window  Size %", 10, 100, 70,
                                    help="Length of sliding window relative to document length")
    with col3:
        temperature = st.slider("Temperature", 0.1, 1.0, 0.6,
                                help="LLM randomness: low → deterministic, high → creative")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        top_p = st.slider("Top P", 0.1, 1.0, 0.75,
                        help="Nucleus sampling probability cutoff for token selection")
    with col2:
        chunk_similarity_threshold = st.slider("Chunk similarity", 0.1, 1.0, 0.65,
                                            help="Minimum cosine similarity for a chunk to be included")
    with col3: 
        dynamic_main_alpha = st.slider("Chunk alpha", 0.1, 1.0, 0.35,
                                help="Percentage of vector/word search in chunks, e.g. 0.3 -> 30% vector search, 70% word match")
    with col4: 
        dynamic_sentence_alpha = st.slider("Sentence alpha", 0.1, 1.0, 0.35,
                                help="Percentage of vector/word search in sentences, e.g. 0.3 -> 30% vector search, 70% word match")

    submit_btn = st.button("Get Answer")

    if submit_btn:
        retrival_params = {
            "top_chunks": top_chunks,
            "window_len_perc": window_len_perc,
            "temperature": temperature,
            "top_p": top_p,
            "chunk_similarity_threshold": chunk_similarity_threshold,
            "main_alpha": dynamic_main_alpha,
            "sentence_alpha": dynamic_sentence_alpha
        }
        chunk_results = retrieve_and_process_top_chunks(rs, client, collection_name, query, params=retrival_params)


        st.subheader("Chunks")
        st.write(chunk_results)

        st.subheader("Answer")
        answer_placeholder = st.empty()
        full_answer = ""

        llm_input = LLMInput(
            context_chunks=chunk_results,
            question=query,
            temperature=temperature,
            top_p=top_p
        )
        for token in rs.api.ask(llm_input):
            full_answer += token
            answer_placeholder.markdown(full_answer + "▌", unsafe_allow_html=True)

        if full_answer== "": 
            full_answer = "I don't know"

        answer_placeholder.markdown(full_answer)