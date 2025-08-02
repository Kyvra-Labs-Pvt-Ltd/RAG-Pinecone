# 🤖 RAG-Pinecone: Cycle Health Assistant

![alt text](image.png)

A Retrieval Augmented Generation (RAG) system for women's health and menstrual cycle information, powered by Pinecone vector database and GROQ API.

## 🚀 Features

- **Vector Search**: Uses Pinecone for efficient semantic search
- **Advanced Embeddings**: BAAI/bge-base-en-v1.5 for high-quality embeddings
- **Smart Chunking**: Recursive text splitting for optimal context
- **Fast API**: RESTful API with FastAPI
- **Interactive UI**: Streamlit web interface
- **Cloud Ready**: Easy deployment to Hugging Face Spaces

## 📋 Prerequisites

- Python 3.8+
- Pinecone API key
- GROQ API key

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd RAG-Pinecone
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
   Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

## 🗄️ Database Setup

1. **Index your documents**

```bash
python rag_index.py
```

This will:

- Load and chunk your knowledge base (`book.txt`)
- Generate embeddings using BAAI/bge-base-en-v1.5
- Create a Pinecone index called "cycle-rag"
- Upload vectors to Pinecone

## 🖥️ Running the Application

### FastAPI Server

```bash
python main.py
```

Access at: http://localhost:8000

### Streamlit Interface

```bash
streamlit run app.py
```

Access at: http://localhost:8501

### Direct Query

```bash
python rag_query.py
```

## 📁 Project Structure

```
RAG-Pinecone/
├── rag_index.py      # Document indexing to Pinecone
├── rag_query.py      # Direct query interface
├── rag_utils.py      # RAG utility functions
├── main.py           # FastAPI application
├── app.py            # Streamlit interface
├── book.txt          # Knowledge base document
├── requirements.txt  # Python dependencies
└── .env             # Environment variables
```

## 🔧 API Endpoints

- `GET /` - Health check
- `POST /query` - Ask questions about menstrual health

Example:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the phases of menstrual cycle?"}'
```

## 🌐 Deployment

### Hugging Face Spaces

1. Create a new Streamlit Space
2. Upload project files
3. Add API keys to Space secrets
4. Your app will be live at: `https://huggingface.co/spaces/username/space-name`

### Local Docker

```bash
docker build -t rag-pinecone .
docker run -p 8000:8000 --env-file .env rag-pinecone
```

## 🧠 How It Works

1. **Document Processing**: Text is split into 500-character chunks with 100-character overlap
2. **Embedding Generation**: Each chunk is converted to 768-dimensional vectors using BGE embeddings
3. **Vector Storage**: Embeddings are stored in Pinecone for fast similarity search
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Response Generation**: Retrieved context is sent to GROQ's language model for answer generation

## 🎯 Use Cases

- Menstrual cycle education
- Fertility awareness
- Hormonal health questions
- Period tracking insights
- Women's health consultation

## 📊 Performance

- **Embedding Model**: BAAI/bge-base-en-v1.5 (768 dimensions)
- **Search**: Sub-second response times with Pinecone
- **Accuracy**: High-quality retrieval with semantic understanding
- **Scalability**: Cloud-native vector database
