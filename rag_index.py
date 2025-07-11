from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load document
with open("book.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunking
# chunks = [text[i:i+500] for i in range(0, len(text), 500)]
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_text(text)

documents = [Document(page_content=chunk) for chunk in chunks]

# Set up embeddings
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #this is fast less memory less accurate
# embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") #this is more effecient but memory consuming

# # Create vector store
# vectordb = Chroma.from_documents(
#     documents=documents,
#     embedding=embedding,
#     persist_directory="./chroma_db"
# )

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory="./chroma_db_bge"
)


print("✅ Indexed", len(chunks), "chunks using langchain_chroma.")
