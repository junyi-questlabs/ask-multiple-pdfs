import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

import os
import shutil
from tqdm import tqdm
import pickle

def get_pdf_text(pdf_docs):
    data = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for (idx, page) in enumerate(pdf_reader.pages):
            data.append({
                "meta": {
                    "file": pdf.name,
                    "page": idx + 1
                },
                "text": page.extract_text()
            })
    return data


def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=200,
        chunk_overlap=20
    )

    return_data = [d.copy() for d in data]

    for page, return_page in zip(data, return_data):
        return_page["chunks"] = text_splitter.split_text(page['text'])

    return return_data


def get_vectorstore(pgbar, text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstores = FAISS.from_texts(["empty"], embeddings)

    # Ensure the cache directory exists
    if not os.path.exists("cache"):
        os.makedirs("cache")

    for (idx, page) in enumerate(text_chunks):
        pgbar.progress(idx / len(text_chunks),
                       text=f"Processing {idx + 1} of {len(text_chunks)}")
        # Create a unique filename for the page's embeddings
        filename = f"cache/{page['meta']['file']}_{page['meta']['page']}.pkl"

        # Check if embeddings file exists for the page
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                vectorstore = pickle.load(f)
        else:
            vectorstore = FAISS.from_texts(texts=page["chunks"],
                                           embedding=embeddings, 
                                           metadatas=[page["meta"] for _ in page["chunks"]])
            
            # Save the embeddings to a file
            with open(filename, "wb") as f:
                pickle.dump(vectorstore, f)

        vectorstores.merge_from(vectorstore)

    # print(vectorstores.docstore._dict)
    pgbar.empty()

    return vectorstores

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    prompt_template = "System: You are an agent to interactive chats with documents of any size, complete with page references for fact checking.\
THINK STEP BY STEP. ALWAYS PROVIDE QUOTES AND SOURCE CITIATIONS.\n\
{context}\n\
Human: {question}\n\
Answer: "

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    document_combine_prompt = PromptTemplate(
        input_variables=["file", "page", "page_content"],
        template= """[DOCUMENT-START]
source: {file}, page: {page}
{page_content}
[DOCUMENT-END]
        """
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        memory=memory,
        max_tokens_limit=4000,
        combine_docs_chain_kwargs={
            "prompt": PROMPT,
            "document_prompt": document_combine_prompt,
        },
        verbose=True
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Private LLM: Proof of Concept",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Private LLM :robot_face: Proof of Concept")
    user_question = st.text_input("Ask a question about your private document base:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your private document base")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            pgbar = st.progress(0, text="Extracting from files...")
            with st.spinner("Please Wait"):
                # get pdf text
                data = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(data)

                # create vector store
                vectorstore = get_vectorstore(pgbar, text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        if st.button("Clear Cached Embeddings"):
            if os.path.exists("cache"):
                shutil.rmtree("cache")


if __name__ == '__main__':
    main()
