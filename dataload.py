from llama_index.core import SimpleDirectoryReader
from openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()

EMBEDDING_MODEL_NAME = "text-embedding-3-large"
EMBEDDING_VECTOR_SIZE = 3072

text_chunker = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def extract_and_segment_pdf(file_path: str):
    documents = SimpleDirectoryReader(
        input_files=[file_path],
        filename_as_id=True
    ).load_data()

    document_texts = [
        doc.text for doc in documents
        if getattr(doc, "text", None)
    ]

    segmented_chunks = []

    for text_content in document_texts:
        segmented_chunks.extend(
            text_chunker.split_text(text_content)
        )

    return segmented_chunks


def generate_embeddings(text_inputs: list[str]) -> list[list[float]]:
    embedding_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=text_inputs,
    )

    return [
        item.embedding
        for item in embedding_response.data
    ]
