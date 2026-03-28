import requests
import io, os, sys
from PIL import Image
from typing import List, Union
from fast_api.unions import EmbedRequest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.process_files import process_file_types
from fast_api.api_models import LLMInput, NERInput





class AccessModels:
    def __init__(self):
        self.base = "http://127.0.0.1:5555"
        self.session = requests.Session()


    # ---------- helpers ----------
    def _post(self, endpoint, payload, timeout):
        if endpoint in ("ner", "embed", "ask", "tokenize", "decode", "tokenizer_info"):
            if not isinstance(payload, dict):
                raise TypeError(f"Payload must be dict, got {type(payload)}")

            for key, val in payload.items():
                if isinstance(val, str):
                    continue

                elif isinstance(val, list):
                    if all(isinstance(x, str) for x in val):
                        continue
                    elif all(isinstance(x, int) for x in val):
                        continue
                    elif all(isinstance(x, list) and all(isinstance(i, int) for i in x) for x in val):
                        continue

                raise TypeError(
                    f"Payload value for '{key}' must be str, list[str], list[int], or list[list[int]], got {type(val)}"
                )

        r = self.session.post(f"{self.base}/{endpoint}", json=payload, timeout=timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"POST /{endpoint} failed ({r.status_code}): {r.text}") from e

        data = r.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"] + "\n" + data.get("traceback", ""))

        return data



    # ---------------------------------------- 
    # public API 
    # ----------------------------------------
    
    # ---------- file processing ----------
    def upload(self, files: list):
        multipart_files = process_file_types(files)
        resp = self.session.post(
            f"{self.base}/upload",
            files=multipart_files,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    
    # ---------- LLM ----------
    def ask(self, input: LLMInput):
        url = f"{self.base}/ask"
        with self.session.post(
            url,
            json=input.model_dump(),
            stream=True
        ) as r:
            r.raise_for_status()

            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk.decode("utf-8")
                    
    # ---------- NER ----------
    def ner(self, text: str):
        nerInput = NERInput(text=text)
        url = f"{self.base}/ner"
        resp = self.session.post(
            url,
            json=nerInput.model_dump(),
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    
    # ---------- Embedder ----------
    def embed(self, texts: EmbedRequest):
        """Call remote /embed endpoint and return embedding(s)."""
        if isinstance(texts, str):
            payload = {"texts": texts}
        elif isinstance(texts, list) and all(isinstance(t, str) for t in texts):
            payload = {"texts": texts}
        else:
            raise TypeError(f"embed expects str or list[str], got {type(texts)}")

        resp = self._post("embed", payload, timeout=120)
        if "embedding" in resp:
            return resp["embedding"]
        return resp.get("embeddings")

    # ---------- tokenizer methods ----------
    def tokenize(self, text):
        """Return token IDs for a string"""
        payload = {"text": text}
        resp = self._post("tokenize", payload, timeout=60)
        return resp["tokens"]
    
    def decode(self, token_ids):
        """Return string from token IDs"""
        payload = {"token_ids": token_ids}
        resp = self._post("decode", payload, timeout=60)
        return resp["text"]

    def tokenizer_info(self) -> dict:
        """Return tokenizer configuration like model_max_length, special tokens, etc."""
        resp = self._post("tokenizer_info", {}, timeout=60)
        return resp