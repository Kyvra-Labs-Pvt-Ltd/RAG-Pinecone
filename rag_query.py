import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

# Load .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Set up Groq-compatible OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Load embeddings and Chroma vector store
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectordb = Chroma(persist_directory="./chroma_db_bge", embedding_function=embedding)

# Start chat memory
chat_history = []

print("💬 Ask me anything (type 'exit' to stop)\n")

while True:
    query = input("You: ")
    if query.strip().lower() == "exit":
        break
    # Predefined small talk / greeting responses
    casual_responses = {
        "hi": "Hi there! I'm here to help you understand your cycle. Ask me anything!",
        "hello": "Hello! Feel free to ask questions about your period or health.",
        "hey": "Hey! How can I assist you today?",
        "how are you": "I'm doing great! How can I help with your menstrual health?",
        "what can you do": "I can help you understand your cycle, answer questions about periods, symptoms, and when to see a doctor.",
        "who are you": "I'm your cycle companion—here to guide you on your menstrual health journey.",
    }

    # Normalize the input
    normalized_query = query.strip().lower()

    # Check for casual input
    if normalized_query in casual_responses:
        print("🤖", casual_responses[normalized_query])
        continue

    # Retrieve docs with scores
    docs_and_scores = vectordb.similarity_search_with_score(query, k=4)
    context_chunks = [doc.page_content for doc, score in docs_and_scores if score < 0.8]

    if not context_chunks:
        context_chunks = [doc.page_content for doc, score in docs_and_scores]  # fallback

    context = "\n".join(context_chunks[:3])  # limit context size
    print("📄 Retrieved context:\n", context[:400], "...\n")  # preview

    # Add user query with context into message
    chat_history.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    # Construct full message payload with memory
    # Construct full message payload with memory
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
        {
            "role": "user",
            "content": f"CONTEXT:\n{context.strip()}\n\nQUESTION:\n{query.strip()}"
        }
    ]


    # Get model response
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages
    )

    answer = response.choices[0].message.content
    print("🤖", answer, "\n")

    # Add assistant's response to chat history
    chat_history.append({"role": "assistant", "content": answer})

    # (Optional) Limit memory to last N turns
    chat_history = chat_history[-10:]
