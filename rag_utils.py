# rag_utils.py
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from openai import OpenAI

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Replace OpenAI embeddings with HuggingFace (free, no API key needed)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectordb = Chroma(persist_directory="./chroma_db_hf", embedding_function=embedding)
    print("✅ Using HuggingFace embeddings (free)")
except Exception as e:
    print(f"Warning: Could not initialize embeddings: {e}")
    print("Using simple text search fallback...")
    vectordb = None

def get_answer(query: str, chat_history: list) -> tuple[str, list]:
    # Predefined small talk responses
    casual_responses = {
        "hi": "Hi there! I'm here to help you understand your cycle. Ask me anything!",
        "hello": "Hello! Feel free to ask questions about your period or health.",
        "hey": "Hey! How can I assist you today?",
        "how are you": "I'm doing great! How can I help with your menstrual health?",
        "what can you do": "I can help you understand your cycle, answer questions about periods, symptoms, and when to see a doctor.",
        "who are you": "I'm your cycle companion—here to guide you on your menstrual health journey.",
    }

    normalized_query = query.strip().lower()
    if normalized_query in casual_responses:
        return casual_responses[normalized_query], chat_history

    # Search docs with fallback
    if vectordb:
        try:
            docs_and_scores = vectordb.similarity_search_with_score(query, k=4)
            context_chunks = [doc.page_content for doc, score in docs_and_scores if score < 0.8]
            if not context_chunks:
                context_chunks = [doc.page_content for doc, score in docs_and_scores]
            context = "\n".join(context_chunks[:3])
        except Exception as e:
            print(f"Vector search failed: {e}")
            context = "General menstrual health knowledge available."
    else:
        context = "General menstrual health knowledge available."

    chat_history.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly and knowledgeable assistant that helps users understand their menstrual cycle, symptoms, and general women’s health. "
                "Give clear, factual, and supportive answers in simple language. "
                "If you're unsure or the answer isn't available, respond gently with something like: 'I'm not sure about that, but I recommend checking with a doctor.' "
                "Avoid guessing or making up answers. Keep responses short, caring, and human — like a helpful health companion, not a robot."
            )
        }
    ] + chat_history + [
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
    ]

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages
    )

    answer = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": answer})

    # Limit memory to last 10 messages
    chat_history = chat_history[-10:]
    return answer, chat_history
