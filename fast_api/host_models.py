import os
import logging
import traceback
from fastapi import FastAPI, UploadFile, File
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.responses import StreamingResponse
from peft import PeftModel
from typing import List, Union
import uvicorn
from PIL import Image
import io, sys
from fastapi import Body
from typing import List
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fast_api.unions import EmbedRequest
from models.OCR.OCR import OCR
from core.llms.langchain.lc_embeddings import LCEmbedder
from models.Embedder import Embedder
from fast_api.api_models import LLMInput, NERInput
from configs.ocr import *
from core.process_files import extract_texts
from models.qwen06b.instruct.generate import stream_generate_answer
from transformers import pipeline

import io, base64
from PIL import Image
import numpy as np

device = 0 if torch.cuda.is_available() else -1

# -------------------- setup --------------------
logging.basicConfig(level=logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.WARNING)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI(title="Local ML Model Host")

# ----------- Load models ONCE -----------
ocr = OCR(
    conf = conf, 
    tile_h=tile_h,
    tile_w=tile_w,
    tile_overlap=tile_overlap,
    scale = scale,
    max_workers = max_workers, 
    box_condense = box_condense
)

# Embedder
embedder = Embedder()
LCemb = LCEmbedder(embedder)
tokenizer = embedder._model.tokenizer

# LLM
base_model_path = "Qwen/Qwen3-0.6B"
lora_checkpoint = "./models/qwen06b/instruct/checkpoint-342"
llm_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
llm_base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, device_map="auto")
llm_model = PeftModel.from_pretrained(llm_base_model, lora_checkpoint) # LoRA
llm_model.eval()

# NER
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True, device=device)


print("✅ Models loaded, FastAPI ready")


# -------------------- File processing endpoints --------------------
@app.post("/upload")
async def ocr_api(files: List[UploadFile] = File(...)):
    try:
        results = extract_texts(ocr, files)
        return results
    except Exception:
        return {
            "error": "processing failed",
            "traceback": traceback.format_exc()
        }
    

# -------------------- LLM endpoints --------------------
@app.post("/ask")
def ask(input: LLMInput):
    def generate():
        try:
            for token in stream_generate_answer(
                model=llm_model,
                tokenizer=llm_tokenizer,
                context_chunks=input.context_chunks,
                question=input.question,
                temperature=input.temperature,
                top_p=input.top_p
            ):
                yield token
        except Exception as e:
            yield f"\n[ERROR]: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

# -------------------- LLM endpoints --------------------
@app.post("/ner")
def ner(nerInput: NERInput):
    try:
        result = ner_pipeline(nerInput.text)

        clean = []
        for ent in result:
            clean.append({
                "entity_group": ent["entity_group"],
                "word": ent["word"],
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "score": float(ent["score"])
            })
        return clean

    except Exception as e:
        print("NER ERROR:", e)
        print(traceback.format_exc())
        return {
            "error": "NER failed",
            "traceback": traceback.format_exc()
        }

    
# -------------------- Embedder endpoints --------------------
@app.post("/embed")
async def embed_api(req: EmbedRequest):
    try:
        is_single = isinstance(req.texts, str)
        texts = [req.texts] if is_single else req.texts
        embeddings = LCemb.embed_documents(texts)

        return (
            {"embedding": embeddings[0]}
            if is_single else
            {"embeddings": embeddings}
        )
    except Exception:
        return {
            "error": "embedding failed",
            "traceback": traceback.format_exc()
        }
    
# -------------------- tokenizer endpoints --------------------
@app.post("/tokenize")
async def tokenize_api(text: Union[str, List[str]] = Body(..., embed=True)):
    """Return token IDs for a string or a list of strings"""
    try:
        if isinstance(text, list):
            tokens_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
            return {"tokens": tokens_list}
        else:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            return {"tokens": tokens}
    except Exception:
        return {
            "error": "tokenization failed",
            "traceback": traceback.format_exc()
        }

@app.post("/decode")
async def decode_api(token_ids: Union[List[int], List[List[int]]] = Body(..., embed=True)):
    """Return string(s) from token IDs"""
    try:
        if token_ids and isinstance(token_ids[0], list):
            texts = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in token_ids
            ]
            return {"text": texts}

        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return {"text": text}

    except Exception:
        return {
            "error": "decode failed",
            "traceback": traceback.format_exc()
        }
    
@app.post("/tokenizer_info")
async def tokenizer_info_api():
    """Return tokenizer configuration info"""
    try:
        info = {
            "model_max_length": tokenizer.model_max_length,
            "pad_token_id": tokenizer.pad_token_id,
            "cls_token_id": getattr(tokenizer, "cls_token_id", None),
            "sep_token_id": getattr(tokenizer, "sep_token_id", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "unk_token_id": getattr(tokenizer, "unk_token_id", None)
        }
        return info
    except Exception:
        return {
            "error": "failed to get tokenizer info",
            "traceback": traceback.format_exc()
        }


# -------------------- run --------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5555,
        log_level="error"
    )