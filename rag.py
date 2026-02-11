from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_db(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

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
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(question)
    if not docs:
        return llm.invoke(question)
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

    response = llm.invoke(prompt)
    return response.strip()