Document AI – Event-Driven RAG System

This project is a production-ready Retrieval-Augmented Generation (RAG) system designed to make PDFs “queryable” with AI. You can upload a PDF, have its content automatically extracted, chunked, and embedded, and then ask questions about it. The system retrieves the most relevant sections and generates answers using an LLM, providing clear responses along with their sources. It’s built with a focus on real-world workflows, modularity, and maintainability.

What It Does:

When you upload a PDF, the system automatically splits it into manageable chunks, generates embeddings for each chunk, and stores them in Qdrant, a vector database optimized for semantic search. When you ask a question, the system searches the stored embeddings, retrieves the most relevant content, and generates an answer that is grounded in the document. This ensures that responses are accurate, context-aware, and traceable to the original source.

Architecture:

The backend uses FastAPI as the API layer, with Inngest handling event-driven workflows for both document ingestion and query processing. OpenAI is used for embeddings and LLM-based answers, while Qdrant serves as the vector store for semantic search. On the frontend, a Streamlit app provides an intuitive interface for uploading documents and asking questions. This architecture separates concerns clearly, making the system scalable, maintainable, and easy to extend.

Key Features:

The system is designed for production use. Workflows are throttled and rate-limited to prevent overload, and vector IDs are deterministic to avoid duplicates. Semantic search uses cosine similarity for accurate retrieval, and all responses are structured using **Pydantic models**. Prompts are carefully crafted to ensure the LLM only answers using the retrieved context. Overall, the system combines clean modular design with real-world AI workflow considerations.

Project Structure:

The project is organized for clarity and maintainability. `main.py` contains the API and workflow definitions, `vector_db.py` abstracts vector database operations, `dataload.py` handles PDF extraction and embedding, `customtype.py` defines typed data models, and `streamlit_app.py` provides the user interface. Uploaded PDFs are stored in the `uploads/` directory for processing.

Running the Project Locally:

To run this project locally, start Qdrant using Docker - docker run -p 6333:6333 qdrant/qdrant
Set your OpenAI API key in a `.env` file - OPENAI_API_KEY=your_api_key_here
Start the FastAPI server - uvicorn main:app --reload
Launch the Inngest development server - npx inngest-cli dev
Finally, start the Streamlit interface - streamlit run streamlit_app.py

This setup allows you to upload PDFs, trigger ingestion, and ask questions through the UI end-to-end.

Why This Project Matters:
Unlike simple RAG demos, this project demonstrates a real-world, production-oriented AI workflow. It integrates event-driven processing, vector database management, semantic search, and LLM-based responses in a clean, modular architecture. It’s a strong example of combining backend engineering best practices with AI capabilities, ready to extend or deploy for more advanced document intelligence use cases.

Author
Kaviyaa Vasudevan – Software Developer specializing in backend systems and AI workflows.




