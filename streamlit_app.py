import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(
    page_title="Document AI – PDF Ingestion",
    page_icon="📄",
    layout="centered"
)


@st.cache_resource
def get_workflow_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="document_ai_app",
        is_production=False
    )


def persist_uploaded_file(file) -> Path:
    storage_dir = Path("uploads")
    storage_dir.mkdir(parents=True, exist_ok=True)

    saved_path = storage_dir / file.name
    saved_path.write_bytes(file.getbuffer())

    return saved_path


async def trigger_document_ingestion(file_path: Path) -> None:
    workflow_client = get_workflow_client()

    await workflow_client.send(
        inngest.Event(
            name="document/ingest_pdf",
            data={
                "pdf_path": str(file_path.resolve()),
                "document_id": file_path.name,
            },
        )
    )


st.title("Upload a PDF for Processing")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    with st.spinner("Uploading file and triggering ingestion workflow..."):
        saved_file_path = persist_uploaded_file(uploaded_file)

        asyncio.run(trigger_document_ingestion(saved_file_path))

        time.sleep(0.3)

    st.success(f"Ingestion triggered for: {saved_file_path.name}")
    st.caption("You can upload another document if needed.")

st.divider()
st.title("Ask Questions About Your Documents")


async def trigger_document_query(user_query: str, retrieval_limit: int):
    workflow_client = get_workflow_client()

    response = await workflow_client.send(
        inngest.Event(
            name="document/query_ai",
            data={
                "question": user_query,
                "top_k": retrieval_limit,
            },
        )
    )

    return response[0]


def get_workflow_api_base() -> str:
    return os.getenv(
        "INNGEST_API_BASE",
        "http://127.0.0.1:8288/v1"
    )


def fetch_workflow_runs(event_id: str) -> list[dict]:
    endpoint = f"{get_workflow_api_base()}/events/{event_id}/runs"

    http_response = requests.get(endpoint)
    http_response.raise_for_status()

    response_data = http_response.json()
    return response_data.get("data", [])


def wait_for_workflow_output(
    event_id: str,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 0.5
) -> dict:

    start_time = time.time()
    last_known_status = None

    while True:
        runs = fetch_workflow_runs(event_id)

        if runs:
            current_run = runs[0]
            run_status = current_run.get("status")
            last_known_status = run_status or last_known_status

            if run_status in ("Completed", "Succeeded", "Success", "Finished"):
                return current_run.get("output") or {}

            if run_status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Workflow execution {run_status}")

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for workflow output "
                f"(last status: {last_known_status})"
            )

        time.sleep(poll_interval_seconds)


with st.form("document_query_form"):
    user_question = st.text_input("Enter your question")
    retrieval_limit = st.number_input(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    submit_clicked = st.form_submit_button("Ask")

    if submit_clicked and user_question.strip():
        with st.spinner("Sending query and generating response..."):
            event_id = asyncio.run(
                trigger_document_query(
                    user_question.strip(),
                    int(retrieval_limit)
                )
            )

            workflow_output = wait_for_workflow_output(event_id)

            generated_answer = workflow_output.get("generated_answer", "")
            source_documents = workflow_output.get("source_documents", [])

        st.subheader("Answer")
        st.write(generated_answer or "(No answer generated)")

        if source_documents:
            st.caption("Sources")
            for source in source_documents:
                st.write(f"- {source}")
