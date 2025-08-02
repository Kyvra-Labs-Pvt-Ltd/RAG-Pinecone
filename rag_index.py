import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

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
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create or connect to index
index_name = "cycle-rag"
dimension = 768  # BGE-base-en-v1.5 embedding dimension

# Check if index exists, create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine"
    )

index = pc.Index(index_name)

# Generate embeddings and upload to Pinecone
vectors = []
for i, chunk in enumerate(chunks):
    # Generate embedding for this chunk
    chunk_embedding = embedding.embed_query(chunk)
    
    vectors.append({
        "id": f"chunk_{i}",
        "values": chunk_embedding,
        "metadata": {"text": chunk}
    })

# Upload in batches
batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(vectors=batch)

print(f"✅ Indexed {len(chunks)} chunks to Pinecone index '{index_name}'")
