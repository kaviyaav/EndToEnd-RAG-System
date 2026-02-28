import pydantic


class DocumentChunkBatch(pydantic.BaseModel):
    text_chunks: list[str]
    document_id: str = None


class VectorUpsertSummary(pydantic.BaseModel):
    total_inserted: int


class RetrievalResult(pydantic.BaseModel):
    retrieved_chunks: list[str]
    source_documents: list[str]


class QueryResponse(pydantic.BaseModel):
    generated_answer: str
    source_documents: list[str]
    context_count: int
