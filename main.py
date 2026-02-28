import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import datetime

from dataload import extract_and_segment_pdf, generate_embeddings
from vector_db import VectorStoreManager
from customtype import (
    QueryResponse,
    RetrievalResult,
    VectorUpsertSummary,
    DocumentChunkBatch,
)

load_dotenv()

workflow_client = inngest.Inngest(
    app_id="document_ai_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@workflow_client.create_function(
    fn_id="Document: Ingest PDF",
    trigger=inngest.TriggerEvent(event="document/ingest_pdf"),
    throttle=inngest.Throttle(
        count=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.document_id",
    ),
)
async def handle_document_ingestion(ctx: inngest.Context):
    def _extract(ctx: inngest.Context) -> DocumentChunkBatch:
        file_path = ctx.event.data["pdf_path"]
        document_id = ctx.event.data.get("document_id", file_path)

        segmented_chunks = extract_and_segment_pdf(file_path)

        return DocumentChunkBatch(
            text_chunks=segmented_chunks,
            document_id=document_id,
        )

    def _embed_and_store(batch: DocumentChunkBatch) -> VectorUpsertSummary:
        chunks = batch.text_chunks
        document_id = batch.document_id

        embeddings = generate_embeddings(chunks)

        record_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{i}"))
            for i in range(len(chunks))
        ]

        metadata_payloads = [
            {"source": document_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        VectorStoreManager().insert_embeddings(
            record_ids,
            embeddings,
            metadata_payloads,
        )

        return VectorUpsertSummary(total_inserted=len(chunks))

    chunk_batch = await ctx.step.run(
        "extract-and-segment",
        lambda: _extract(ctx),
        output_type=DocumentChunkBatch,
    )

    upsert_summary = await ctx.step.run(
        "embed-and-store",
        lambda: _embed_and_store(chunk_batch),
        output_type=VectorUpsertSummary,
    )

    return upsert_summary.model_dump()


@workflow_client.create_function(
    fn_id="Document: Query",
    trigger=inngest.TriggerEvent(event="document/query_ai"),
)
async def handle_document_query(ctx: inngest.Context):
    def _retrieve(query_text: str, limit: int = 5) -> RetrievalResult:
        query_embedding = generate_embeddings([query_text])[0]

        vector_store = VectorStoreManager()
        search_output = vector_store.query_similar(
            query_embedding,
            limit,
        )

        return RetrievalResult(
            retrieved_chunks=search_output["contexts"],
            source_documents=search_output["sources"],
        )

    query_text = ctx.event.data["question"]
    retrieval_limit = int(ctx.event.data.get("top_k", 5))

    retrieval_result = await ctx.step.run(
        "embed-and-retrieve",
        lambda: _retrieve(query_text, retrieval_limit),
        output_type=RetrievalResult,
    )

    formatted_context = "\n\n".join(
        f"- {chunk}" for chunk in retrieval_result.retrieved_chunks
    )

    user_prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{formatted_context}\n\n"
        f"Question: {query_text}\n"
        "Answer concisely using the context above."
    )

    llm_response = await ctx.step.ai.infer(
        "generate-answer",
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the provided context.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=1024,
        temperature=0.2,
    )

    generated_answer = llm_response["choices"][0]["message"]["content"].strip()

    final_response = QueryResponse(
        generated_answer=generated_answer,
        source_documents=retrieval_result.source_documents,
        context_count=len(retrieval_result.retrieved_chunks),
    )

    return final_response.model_dump()


app = FastAPI()

inngest.fast_api.serve(
    app,
    workflow_client,
    [
        handle_document_ingestion,
        handle_document_query,
    ],
)
