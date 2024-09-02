from dotenv import load_dotenv
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory


def handle_user_input(user_question):
    persist_directory = "DB/chroma"

    llm = GoogleGenerativeAI(model="models/text-bison-001", temperature=0.7)

    embeddings = GooglePalmEmbeddings()

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    retriever = vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )

    response = qa({"question": user_question})
    st.write(response)


def main():
    load_dotenv()
    st.set_page_config(page_title="Kaanoon ChatBot", layout="wide", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Kaanoon ChatBot :books:")
    user_question = st.text_input("Ask Questions:")

    if user_question:
        handle_user_input(user_question)


if __name__ == "__main__":
    main()
