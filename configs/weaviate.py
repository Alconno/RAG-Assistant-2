# ---------------------- Config ----------------------
collection_name = "abychunks"
batch_size = 1024
token_limit_per_chunk = 50  # For split_text_by_token_limit (not used atm)

# chunkify
max_chunk_size = 200        # in tokens
cosine_threshold = 0.6
overlap_percent = 0.1

# Data querying -> if 0.8 -> 0.8 vector, 0.2 text
#main_alpha = 0.3 # set dynamically now
#sentence_alpha = 0.4 # set dynamically now

# Neighboring chunk addition
add_neighboring_chunk = True
perc_of_next_chunk = 0.25
merge_cos_sim = 0.8