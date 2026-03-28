import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class Embedder():
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        import inspect

        stack = inspect.stack()
        # The caller is usually at stack[1]
        caller = stack[1]
        print(f"Embedder created from file: {caller.filename}, line: {caller.lineno}, in function: {caller.function}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(
            "BAAI/llm-embedder",
            device=device
        )
        
        # Fix meta tensor params if needed
        for name, param in self._model.named_parameters():
            if param.device.type == "meta":
                param.data = param.data.to(device)

        print("Model loaded on", device)

    def __call__(self, texts):
        return self._model.encode(texts, batch_size=64)