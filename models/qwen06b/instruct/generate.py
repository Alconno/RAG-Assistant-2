from peft import PeftModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re



def generate_answer(model, tokenizer, context_chunks, question):
    """
    Generates a full answer from a language model in one pass using the provided context and question. 
    It constructs a prompt, tokenizes it, feeds it to the model, and decodes the generated tokens into text.
    """
    context = "Context:\n".join(context_chunks)+"\n" if len(context_chunks) else ""

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{context}"
        f"Question: {question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        # answer
        #<|im_end|>
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.9,
        top_p=0.95
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return output_text[len(prompt):].strip()






def stream_generate_answer(model, tokenizer, context_chunks, question, temperature=0.9, top_p=0.85):
    """
    Generates an answer token-by-token in a streaming fashion. 
    It allows partial outputs to be processed as they are generated, using 
    temperature and top-p sampling for more controllable randomness.
    """
    context = "Context:\n".join(context_chunks)+"\n" if len(context_chunks) else ""

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{context}"
        f"Question: {question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        # answer
        #<|im_end|>
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    generated_ids = input_ids.clone()
    past_key_values = None  # Cache for transformer layers to speed up generation

    for _ in range(512):
        if past_key_values is None:
            outputs = model(input_ids=input_ids, use_cache=True)
        else:
            outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)

        logits = outputs.logits  # [batch=1, seq_len=1, vocab_size]
        past_key_values = outputs.past_key_values

        next_token_logits = logits[0, -1, :]

        probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Filter tokens outside top_p cumulative probability
        filtered_indices = cumulative_probs <= top_p
        filtered_probs = sorted_probs * filtered_indices.float()

        if filtered_probs.sum() == 0:
            # no tokens, revert to unfiltered probs
            filtered_probs = sorted_probs

        filtered_probs /= filtered_probs.sum() # Norm

        # Sample next token id from filtered probability distribution
        next_token = torch.multinomial(filtered_probs, 1)
        next_token_index = next_token.item()
        next_token_id = sorted_indices[next_token_index]

        # Ensure next_token_id is a tensor with batch dimension
        if next_token_id.dim() == 0:
            next_token_id = next_token_id.unsqueeze(0)

        next_token_id = next_token_id.unsqueeze(1)

        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        input_ids = next_token_id 

        new_token_text = tokenizer.decode(next_token_id.squeeze(1).tolist(), skip_special_tokens=True)
        yield new_token_text 
 
        if next_token_id.item() == tokenizer.eos_token_id:
            break