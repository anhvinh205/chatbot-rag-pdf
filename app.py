import chainlit as cl
from llm import load_llm
from rag import build_vector_db, answer_question

PDF_PATH = "data/documents.pdf"

@cl.on_chat_start
async def start():
    await cl.Message(content="📄 Load PDF...").send()

    # Load LLM and Database
    llm = load_llm()
    vector_db = build_vector_db(PDF_PATH)

    # Save to session 
    cl.user_session.set("llm", llm)
    cl.user_session.set("vector_db", vector_db)

    await cl.Message(content="✅ DONE! Show me your questions?").send()

@cl.on_message
async def main(message: cl.Message):
    llm = cl.user_session.get("llm")
    vector_db = cl.user_session.get("vector_db")

    if llm and vector_db:
        res = answer_question(llm, vector_db, message.content)
        await cl.Message(content=res).send()
