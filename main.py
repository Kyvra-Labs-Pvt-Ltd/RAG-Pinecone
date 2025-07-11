# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_utils import get_answer

app = FastAPI()
chat_memory = []  # Global chat memory (simple version)

class Question(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Cycle AI API is running 🚀"}

@app.post("/ask")
def ask(question: Question):
    answer, updated_history = get_answer(question.query, chat_memory)
    chat_memory.clear()
    chat_memory.extend(updated_history)
    return {"answer": answer}
