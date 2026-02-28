from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class VectorStoreManager:
    def __init__(self, endpoint="http://localhost:6333", index_name="documents", vector_size=3072):
        self.connection = QdrantClient(url=endpoint, timeout=30)
        self.index_name = index_name

        if not self.connection.collection_exists(self.index_name):
            self.connection.create_collection(
                collection_name=self.index_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def insert_embeddings(self, record_ids, embedding_vectors, metadata_list):
        records = [
            PointStruct(
                id=record_ids[i],
                vector=embedding_vectors[i],
                payload=metadata_list[i]
            )
            for i in range(len(record_ids))
        ]

        self.connection.upsert(
            collection_name=self.index_name,
            points=records
        )

    def query_similar(self, query_embedding, limit: int = 5):
        search_results = self.connection.search(
            collection_name=self.index_name,
            query_vector=query_embedding,
            with_payload=True,
            limit=limit
        )

        retrieved_texts = []
        source_identifiers = set()

        for result in search_results:
            metadata = getattr(result, "payload", None) or {}
            content = metadata.get("text", "")
            source_label = metadata.get("source", "")

            if content:
                retrieved_texts.append(content)
                source_identifiers.add(source_label)

        return {
            "contexts": retrieved_texts,
            "sources": list(source_identifiers)
        }
