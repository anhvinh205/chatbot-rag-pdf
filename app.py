import chainlit as cl
from llm import load_llm
from rag import build_vector_db, answer_question
import os

PDF_PATH = "data/documents.pdf"

_llm = None
_vector_db = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = load_llm()
    return _llm

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = build_vector_db(PDF_PATH)
    return _vector_db


@cl.on_chat_start
async def start():
    await cl.Message(
        content="📄 Loading PDF and initializing the model..."
    ).send()

    # Check if the PDF file exists
    if not os.path.exists(PDF_PATH):
        await cl.Message(
            content=(
                f"❌ PDF file not found at `{PDF_PATH}`.\n"
                "Please place the PDF file inside the `data/` folder."
            )
        ).send()
        return

    try:
        llm = get_llm()
        vector_db = get_vector_db()

        # Save to session
        cl.user_session.set("llm", llm)
        cl.user_session.set("vector_db", vector_db)

        await cl.Message(
            content="✅ System is ready! Ask any question about your document."
        ).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ Initialization error: {str(e)}\n\nPlease check the model and the PDF file."
        ).send()


@cl.on_message
async def main(message: cl.Message):
    llm = cl.user_session.get("llm")
    vector_db = cl.user_session.get("vector_db")

    if not llm or not vector_db:
        await cl.Message(
            content="⚠️ The system is not initialized. Please start a new conversation."
        ).send()
        return

    try:
        res = answer_question(llm, vector_db, message.content)
        await cl.Message(content=res).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ Error while processing the question: {str(e)}"
        ).send()