import streamlit as st
from streamlit_chat import message as msg
import os
from dotenv import load_dotenv

import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Sequence
from typing_extensions import Annotated, TypedDict


# Function to create the RAG chain
def create_chain(text):
    # Split text into chunks for embedding and retrieval
    text_splitter = RecursiveCharacterTextSplitter()
    text = text_splitter.split_text(text=text)

    # Initialize embeddings model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=API_KEY)

    # Create an in-memory vector store for the embeddings
    vectorstore = InMemoryVectorStore.from_texts(
        texts=text, embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    # Initialize the large language model (LLM) for chat
    llm = ChatMistralAI(model="mistral-large-latest", mistral_api_key=API_KEY)

    # Define system prompt for context-aware question reformulation
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Define system prompt for answering customer queries based on retrieved context
    system_prompt = (
        "You are a customer support assistant. "
        "Use the following pieces of context to answer the customer's question. "
        "If you don't know the answer, say politely that you don't know. "
        "Keep your answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create the retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Store the chain in session state
    st.session_state['rag_chain'] = rag_chain

    # Define state graph for handling chat history
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Memory saver to manage memory state across interactions
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# Define the state for chat history and context
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


# Function to call the model and generate a response
def call_model(state: State):
    rag_chain = st.session_state['rag_chain']
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


# Function to process the uploaded PDF and extract text
def file_processing(pdfs):
    text = ""
    for file in pdfs:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Enable chain when documents are uploaded
def enable_chain():
    st.session_state['process_chain'] = 1    


# Disable chain when no new documents are uploaded
def disable_chain():
    st.session_state['process_chain'] = 0 


if __name__ == "__main__":
    # Load environment variables (API keys)
    API_KEY = st.secrets["MISTRAL_API_KEY"]
    HF_TOKEN = st.secrets["HF_TOKEN"]
    
    os.environ["MISTRAL_API_KEY"] = API_KEY
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    # Initialize session state for chain processing
    if 'process_chain' not in st.session_state:
        st.session_state['process_chain'] = 0

    st.title("Customer Support Q&A System")

    # Tab layout for uploading PDFs and asking questions
    tab1, tab2 = st.tabs(["Upload Manual", "Ask a Question"])

    tab1.subheader("Customer Support Manual Upload")
    with tab1:
        pdfs = tab1.file_uploader("Upload customer support manual (PDF)", accept_multiple_files=True, on_change=enable_chain)
        text = ""
        if pdfs:
            text = file_processing(pdfs)
        
        # Build RAG chain if documents are uploaded and processed
        if (len(text) != 0) and (st.session_state['process_chain'] != 0):
            app = create_chain(text)
            st.session_state['chain'] = app
        else:
            if 'chain' in st.session_state:
                del st.session_state['chain']

    tab2.subheader("Chat with Customer Support")
    with tab2:
        with st.form(key='user_form', clear_on_submit=True):
            user_input = st.text_input(label="Your Question:", placeholder="Ask about your support manual", key='input')
            submit_button = st.form_submit_button(label='Send', on_click=disable_chain)

            if submit_button and user_input:
                if 'chain' in st.session_state:
                    with st.spinner('Generating response...'):
                        app = st.session_state['chain']
                        config = {"configurable": {"thread_id": "thread1"}}
                        result = app.invoke({"input": user_input}, config=config)
                        
                        # Display chat history
                        chat_history = app.get_state(config).values["chat_history"]
                        for message in chat_history:
                            if message.type == 'human':
                                msg(message.content, is_user=True)
                            else:
                                msg(message.content)
                else:
                    st.write("Please upload the customer support manual and try again.")
