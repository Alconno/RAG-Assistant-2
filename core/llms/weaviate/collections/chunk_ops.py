from weaviate.classes.config import Configure
from weaviate.collections.classes.config import Property, DataType, Configure
from weaviate.client import WeaviateClient
from weaviate.classes.query import MetadataQuery
from models.Embedder import Embedder
from core.llms.weaviate.utility import uuid_from_string
from core.state import RuntimeState
import json


def create_chunks_collection(
    client: WeaviateClient, 
    collection_name: str,
    vectorizer_config=Configure.Vectorizer.none(),
):
    """
    Create a collection in Weaviate if it doesn't exist
    """
    if not client.collections.exists(collection_name):
        print("Creating collection ", collection_name)
        client.collections.create(
            name=collection_name,
            properties=[Property(name="data", data_type=DataType.TEXT)],
            vectorizer_config=vectorizer_config,
        )


def batch_insert_chunks(
    rs: RuntimeState,
    client: WeaviateClient, 
    chunks, 
    collection_name: str,
):
    """
    Batch insert multiple chunks into the Weaviate collection
    """
    collection = client.collections.get(collection_name)
    texts = [obj["text"] for obj in chunks]
    object_vectors = rs.api.embed(texts)

    with collection.batch.fixed_size(batch_size=64) as batch:
        for obj, vector in zip(chunks, object_vectors):
            batch.add_object(
                properties={
                    "text": obj["text"],
                    "doc_id": obj["doc_id"],
                    "chunk_idx": obj["chunk_idx"],
                },
                vector=vector,
                uuid=uuid_from_string(f"{obj['doc_id']}_{obj['chunk_idx']}")
            )
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")



def get_top_k_chunks(
    rs: RuntimeState,
    client, 
    collection_name: str, 
    query_text: str, 
    k: int = 2,
    similarity_threshold: float = 0.85,
): 
    """ 
    Retrieve top k most similar chunks to a query, filtered by similarity threshold 
    """
    collection = client.collections.get(collection_name)
    near_vector = rs.api.embed(query_text)

    response = collection.query.near_vector(
        near_vector=near_vector,
        limit=k,
        return_metadata=MetadataQuery(distance=True)
    )

    filtered_results = []
    for obj in response.objects:
        cosine_sim = 1 - obj.metadata.distance
        print("Cosine sim:", cosine_sim)
        if cosine_sim >= similarity_threshold:
            filtered_results.append(obj)

    return filtered_results