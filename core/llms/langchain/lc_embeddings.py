from langchain_core.embeddings import Embeddings

class LCEmbedder(Embeddings):
    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts):
        return self.embedder(texts).tolist()

    def embed_query(self, text):
        return self.embedder([text]).tolist()[0]
    

class RemoteLCEmbedder:
    """Wraps remote / API embed call to behave like LangChain Embeddings."""
    def __init__(self, rs_api):
        self.rs_api = rs_api

    def embed_documents(self, texts):
        return self.rs_api.embed(texts)
    
    def embed_query(self, text):
        return self.rs_api.embed([text])[0]