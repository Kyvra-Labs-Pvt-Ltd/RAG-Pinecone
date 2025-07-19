# RAG-Pinecone

A simple Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://python.langchain.com/), [Pinecone](https://www.pinecone.io/), and [Groq](https://groq.com/) for fast, scalable, and accurate question answering over your own documents.

---

## Features

- **Document Indexing:** Load and split your documents, then embed and index them in Pinecone.
- **RAG Chain:** Retrieve relevant context and generate answers using Groq's LLMs.
- **Easy to Run:** Jupyter notebook for step-by-step execution.

---

## Quickstart

### 1. Install Requirements

```bash
pip install -qU langchain langchain-pinecone langchain-groq "pinecone[grpc]" langchain-community