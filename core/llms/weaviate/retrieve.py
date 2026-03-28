from core.state import RuntimeState
import math
import torch
import torch.nn.functional as F
from core.llms.weaviate.ingest import split_text_into_sentences
from weaviate.classes.query import Filter
from configs.weaviate import *
from sentence_transformers import util

def retrieve_and_process_top_chunks(
        rs: RuntimeState, 
        client, 
        collection_name, 
        query, 
        params={
            "top_chunks": 5,
            "window_len_perc": 70,
            "temperature": 0.6,
            "top_p": 0.75,
            "chunk_similarity_threshold": 0.65,
            "main_alpha": 0.35,
            "sentence_alpha": 0.35,
        }):
    """
    Retrieves the most relevant text chunks from a collection for a query, 
    merges neighboring chunks if semantically similar, and returns the top scoring results.
    """
    collection = client.collections.get(collection_name)
    main_alpha = params["main_alpha"]
    sentence_alpha = params["sentence_alpha"]
    
    entities = rs.api.ner(query)
    hybrid_query = query + " " + " ".join([e["word"] for e in entities])
    query_emb = rs.api.embed(hybrid_query)

    response = collection.query.hybrid(
        query=hybrid_query,
        vector=query_emb,
        alpha=main_alpha,
        limit=params["top_chunks"],
    )
    objs = response.objects
    all_candidates = []

    for obj in objs:
        doc_text    = obj.properties["text"]
        doc_id      = obj.properties["doc_id"]
        chunk_idx   = obj.properties["chunk_idx"]

        sentences = split_text_into_sentences(doc_text)
        min_window_len = math.ceil(len(sentences) * (params["window_len_perc"] / 100))

        windows, embeddings = get_sentence_and_window_embeddings(
            rs, sentences,
            min_window_len=min_window_len,
            max_window_len=min_window_len*3.0
        )

        best_score, best_idx = get_top_window_sim(hybrid_query, query_emb, embeddings, windows, alpha=sentence_alpha)
        best_window = windows[best_idx].split()
        
        start_sentence = None
        for i, sent in enumerate(sentences):
            if best_window[0] in sent.split():
                start_sentence = i
                break

        if start_sentence is not None:
            full_text = " ".join(sentences[start_sentence:])

            chunk_start = start_sentence
            chunk_end = len(sentences) - 1
            chunk_midpoint = (chunk_start + chunk_end) / 2

            if add_neighboring_chunk and chunk_midpoint / max(len(sentences), 1) >= 0.5:
                next_idx = chunk_idx+1
                next_chunk_resp = collection.query.fetch_objects(
                    filters=(
                        Filter.by_property("doc_id").equal(doc_id) & \
                        Filter.by_property("chunk_idx").equal(next_idx)
                    ),
                    limit=1
                )
                if next_chunk_resp.objects:
                    next_text = next_chunk_resp.objects[0].properties["text"]
                    nt_words = next_text.split(" ")
                    overlap_idx = int(len(nt_words)*overlap_percent)
                    next_chunk = " ".join(nt_words[overlap_idx : max(1, int(len(nt_words)*perc_of_next_chunk))])
                    emb1 = rs.api.embed(full_text)
                    emb2 = rs.api.embed(next_chunk)
                    if util.cos_sim(emb1, emb2).item() > merge_cos_sim:
                        full_text += "\n" + next_chunk
            
            all_candidates.append((best_score, full_text))

    all_candidates.sort(key=lambda x: x[0], reverse=True)
    return [
        text for score, text in all_candidates
        if score >= params["chunk_similarity_threshold"]
    ][:params["top_chunks"]]


def get_sentence_and_window_embeddings(rs: RuntimeState, sentences, min_window_len=15, max_window_len=20):
    """
    Generates overlapping windows of sentences from text and computes 
    embeddings for each window to capture local context.
    """
    n = len(sentences)
    windows = []

    k = int(max(1, min(min_window_len, n)))         # minimum window length
    max_k = int(min(max_window_len, n))             # maximum window length
    stride = max(1, k // 2)

    for start in range(n):
        for end in range(start + k, min(start + max_k + 1, n + 1), stride):
            window_text = " ".join(sentences[start:end])
            windows.append(window_text)

    if not windows:
        return [], []

    embeddings = rs.api.embed(windows)
    return windows, embeddings


def get_top_window_sim(query, query_emb, window_vecs, window_texts, alpha=0.75):
    """
    Calculates hybrid similarity scores between a query and text windows using vector 
    similarity and word overlap, returning the highest scoring window.
    """
    all_vecs = torch.as_tensor(window_vecs)
    query_tensor = torch.as_tensor(query_emb)
    all_vecs = F.normalize(all_vecs, dim=1)
    query_tensor = F.normalize(query_tensor, dim=0)
    vec_sims = all_vecs @ query_tensor

    query_words = set(query.lower().split())
    text_sims = []
    for t in window_texts:
        t_words = set(t.lower().split())
        text_sims.append(len(query_words & t_words) / max(1, len(query_words)))
    text_sims = torch.as_tensor(text_sims)


    hybrid_scores = alpha * vec_sims + (1 - alpha) * text_sims
    best_idx = torch.argmax(hybrid_scores).item()
    best_score = hybrid_scores[best_idx].item()
    return best_score, best_idx
