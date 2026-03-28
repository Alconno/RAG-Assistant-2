from core.state import RuntimeState
import hashlib

def uuid_from_string(s):
    return hashlib.md5(s.encode()).hexdigest() 

def split_text_by_token_limit(rs: RuntimeState, text, max_tokens=None, custom_limit:int = None):
    """
    Splits a long text into batches based on a token limit using a tokenizer.
    It ensures that each batch stays within the model's max token length.
    """
    if max_tokens is None:
        max_tokens = tokenizer.model_max_length if custom_limit == None else custom_limit

    tokens = rs.api.tokenize(text)
    if len(tokens) <= max_tokens:
        return [text]
    token_chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return rs.api.decode(token_chunks)
