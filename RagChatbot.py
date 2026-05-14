from multiprocessing import context

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, chat_models, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


OPENAI_API_KEY = "sk-xxxxx"
st.header("Rag Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Choose a file and Start asking Questions", type="pdf")

#Extract contents from PDF and Chunk it

if file is not None:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text = text + page.extract_text() + "\n\n"
    #st.write(text)

    #split into chunk

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ",", "."],
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    #generating embeddings and store it in vector DB
    with st.spinner("Loading embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks,embeddings)

    #get user question
    user_question = st.text_input("Enter your question here")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":4}
    )

    #define the LLMs and prompts

    llm = ChatOllama(
        model="mistral"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant answering questions about a PDF document.\n\n"

             "Guidelines:\n"
             "1. Provide complete, well-explained answers using the context below.\n"
             "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
             "3. If the context mentions related information, include it to give fuller picture.\n"
             "4. Only use information from the provided context - do not use outside knowledge.\n"
             "5. Summarize long information, ideally in bullets where needed\n"
             "6. If the information is not in the context, say so politely.\n\n"

             "Context:\n{context}"),
            ("human", "{question}")
        ]

    )

    chain =(
        {"context": retriever | format_docs, "question" : RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()

    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)