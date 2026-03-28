import torch
import torch.nn.functional as F
import spacy
import math
from core.state import RuntimeState
import math
import uuid
import streamlit as st
from configs.weaviate import cosine_threshold, max_chunk_size, overlap_percent, batch_size
from core.llms.weaviate.collections.chunk_ops import batch_insert_chunks
from core.llms.weaviate.utility import split_text_by_token_limit



def ingest_text(rs, client, full_text, collection_name, progress_bar=None):
    """split text into chunks, batch-process them, and insert into DB with progress tracking"""
    rows = split_text_into_sentences(full_text)
    n_rows = len(rows)
    n_batches = math.ceil(n_rows / batch_size)
    doc_id = str(uuid.uuid4())
    chunk_idx = 0

    # If no progress bar is passed, create one
    if progress_bar is None:
        progress_bar = st.progress(0, text="Ingesting data...")

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_rows)
        batch_rows = rows[start_idx:end_idx]

        chunks = process_batch(rs, batch_rows)
        better_chunks = []
        for chunk in chunks:
            better_chunks.append({"doc_id": doc_id, "chunk_idx": chunk_idx, "text": chunk})
            chunk_idx += 1

        batch_insert_chunks(rs, client, better_chunks, collection_name)

        st.success(f"Inserted lines {start_idx}–{end_idx}")
        progress_bar.progress((i + 1)/n_batches, text=f"Ingesting data... {int((i+1)/n_batches*100)}%")

    st.success("Ingestion complete")
    progress_bar.empty()



def process_text(rs: RuntimeState, text, token_limit_per_chunk=512):
    """
    Splits text into chunks and embeds them using remote RS API.
    """
    text_batches = split_text_by_token_limit(rs, text, custom_limit=token_limit_per_chunk)
    embeddings = rs.api.embed(text_batches)

    return chunkify(text_batches, embeddings,
                    max_chunk_size=max_chunk_size,
                    cosine_similarity_value=cosine_threshold)

def process_batch(rs: RuntimeState, data_batch, token_limit_per_chunk=512):
    """
    Splits texts into chunks and embeds them using remote RS API.
    """
    """text_batches = []
    for row in data_batch:
        text_batches.extend(
            split_text_by_token_limit(
                rs, row, custom_limit=token_limit_per_chunk))"""

    embeddings = rs.api.embed(data_batch)
    return chunkify(rs,
                    data_batch, 
                    embeddings,
                    max_chunk_size=max_chunk_size,
                    cosine_similarity_value=cosine_threshold,
                    overlap_percent=overlap_percent)



# Load spaCy English model for sentence tokenization
nlp = spacy.load("en_core_web_sm")
def split_text_into_sentences(text):
    """split a text into individual sentences using NLP"""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def chunkify(
    rs: RuntimeState,
    text_batches,
    embeddings,
    max_chunk_size=64,
    cosine_similarity_value=0.15,
    overlap_percent=0.1,
):
    tokenized_batches = rs.api.tokenize(text_batches)
    # tokenized_batches is a list of lists of token ids
    token_lengths = [len(toks) for toks in tokenized_batches]

    chunks = []
    current_chunk = []
    current_tokens = 0
    min_chunk_size = int(0.85 * max_chunk_size)

    for i, sentence in enumerate(text_batches):
        tokens = tokenized_batches[i]
        token_len = token_lengths[i]

        would_overflow = current_tokens + token_len > max_chunk_size
        should_split = False

        if would_overflow:
            if current_chunk:
                should_split = True
            else:
                for j in range(0, len(tokens), max_chunk_size):
                    sub_tokens = tokens[j:j+max_chunk_size]
                    chunks.append(rs.api.decode([sub_tokens]))
                continue
        elif i < len(text_batches) - 1:
            emb1 = torch.tensor(embeddings[i])
            emb2 = torch.tensor(embeddings[i + 1])
            sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            if sim < cosine_similarity_value and current_tokens >= min_chunk_size:
                should_split = True

        if should_split:
            chunks.append(" ".join(current_chunk))

            target_overlap_tokens = max(1, int(current_tokens * overlap_percent))

            overlap = []
            overlap_count = 0
            for j in reversed(range(len(current_chunk))):
                toks = tokenized_batches[i - len(current_chunk) + j]
                if overlap_count + len(toks) > target_overlap_tokens:
                    break
                overlap.insert(0, current_chunk[j])
                overlap_count += len(toks)

            current_chunk = overlap
            current_tokens = overlap_count

        current_chunk.append(sentence)
        current_tokens += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks