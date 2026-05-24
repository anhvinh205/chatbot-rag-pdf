from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_db(pdf_path: str):
     # Check files
    import os
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Not found file PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError(f"File PDF is empty or cannot be read: {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding
    )

    return vector_db

def answer_question(llm, vector_db, question: str):
    def _call_llm(llm_obj, prompt_text: str) -> str:
        try:
            res = llm_obj(prompt_text)
        except Exception:
            try:
                # fallback for some pipeline wrappers
                res = llm_obj.generate(prompt_text)
            except Exception:
                res = None

        if isinstance(res, str):
            return res
        if res is None:
            return ""

        # LangChain LLMResult shape
        try:
            return res.generations[0][0].text
        except Exception:
            pass

        # transformers pipeline -> list[dict]{'generated_text'}
        try:
            if isinstance(res, list) and len(res) and isinstance(res[0], dict):
                return res[0].get("generated_text") or str(res[0])
        except Exception:
            pass

        return str(res)

    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    try:
        docs = retriever.get_relevant_documents(question)
    except Exception:
        try:
            docs = retriever.get_documents(question)
        except Exception:
            docs = []

    if not docs:
        return _call_llm(llm, question).strip()

    context = "\n\n".join([d.page_content[:700] for d in docs])

    prompt = f"""
You are a helpful assistant.
Use the context below to answer the question.
If the answer is not in the context, say "I don't know".


Context:
{context}

Question:
{question}

Answer:
"""

    response = _call_llm(llm, prompt)
    return response.strip()
