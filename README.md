# Context Aware Video Retrieval

This project processes video files by extracting metadata such as summaries, keywords, and timestamps using Google Gemini. It then builds a semantic search engine using FAISS, LangChain, and Sentence Transformers.

## Features

- **Video Processing:** Upload and analyze videos to generate metadata.
- **Metadata Extraction:** Extract summaries, keywords, and timestamp details.
- **Semantic Search:** Build and query a vector store for efficient video content retrieval.

## Installation

Install the required packages:

```bash
pip install -U -q google-genai langchain-community faiss-cpu
pip install sentence-transformers
pip install --upgrade tensorflow
