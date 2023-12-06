import langchain
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema.messages import SystemMessage

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import os
import pickle
from pathlib import Path

# the knowledge folder under thesame directory as this file
DOC_DIR = os.path.join(os.path.dirname(__file__), "knowledge")

def get_pdf_text(pdf_docs):
    data = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for (idx, page) in enumerate(pdf_reader.pages):
            data.append({
                "meta": {
                    "file": os.path.basename(pdf.name),
                    "page": idx + 1,
                    "source": "{file}, page {page}".format(file=os.path.basename(pdf.name), page=idx + 1),
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
        tmp_chunks = text_splitter.split_text(page['text'])
        new_chunks = ["[" + Path(page["meta"]["file"]).stem + "] " + chunk for chunk in tmp_chunks]
        return_page["chunks"] = new_chunks

    return return_data

# TODO PRIORITY-LOW
# TODO Batch the embeddings calls
# TODO Use a local embeddings model
def get_vectorstore(pgbar, text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstores = FAISS.from_texts(["empty"], embeddings, metadatas=[text_chunks[0]["meta"]])

    # Ensure the cache directory exists
    if not os.path.exists("cache"):
        os.makedirs("cache")

    for (idx, page) in enumerate(text_chunks):
        pgbar.progress(idx / len(text_chunks),
                       text=f"Processing page {idx + 1} of {len(text_chunks)}")
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

@st.cache_resource
def prepare_pdfs():
    pgbar = st.progress(0, "Preparing Knowledges...")
    pdfs = [f for f in os.listdir(DOC_DIR) if f.endswith(".pdf")]
    handles = [open(os.path.join(DOC_DIR, pdf), "rb") for pdf in pdfs]
    data = get_pdf_text(handles)
    # Close the files
    for pdf in handles:
        pdf.close()

    # get the text chunks
    text_chunks = get_text_chunks(data)

    # create vector store
    vectorstore = get_vectorstore(pgbar, text_chunks)

    return vectorstore


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Private Knowledge Grounding**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Related Knowledge:**")
        self.status.update(label=f"**Private Knowledge Grounding:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # print(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Knowledge Snippet {idx+1}** _{source}_")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")
        print(prompts)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


load_dotenv()
st.set_page_config(page_title="Private LLM: Proof of Concept",
                    page_icon=":robot_face:")

vs = prepare_pdfs()

st.header("Motherson LLM POC")
    

llm = ChatOpenAI(streaming=True, temperature=0, model_name="gpt-4")
# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

document_combine_prompt = PromptTemplate(
    input_variables=["source", "page_content"],
    template= """[DOCUMENT-START]
source: {source}
{page_content}
[DOCUMENT-END]
    """
)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key='chat_history', chat_memory=msgs, return_messages=True)

SYSTEM_TEMPLATE = """You are an agent to interactive chats with annual report documents, complete with source references for fact checking. THINK STEP BY STEP. ALWAYS PROVIDE QUOTES AND SOURCE CITIATIONS. Use the following related documents fetched by semantic search to answer the users question. BE AWARE OF THE TIME OF REFERENCE.
----------------
{context}"""

messages = [
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template("{question}"),
]

CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vs.as_retriever(search_kwargs={"k": 10}),
    memory=memory,
    max_tokens_limit=4000,
    combine_docs_chain_kwargs={
        "prompt": CHAT_PROMPT,
        "document_prompt": document_combine_prompt,
    }
)

if len(msgs.messages) == 0:
    msgs.add_ai_message("Ask me anything about Motherson's annual reports (2020-2023)!")
    

roles = {"human": "user", "ai": "assistant"}
avatars = {
    "human": "https://yestherapyhelps.com/images/frases-y-reflexiones/339/75-frases-y-reflexiones-de-michel-foucault-3.jpg", 
    "ai": "https://questlabs.io/favicon.ico"
}
for msg in msgs.messages:
    if isinstance(msg, SystemMessage):
        continue
    st.chat_message(roles[msg.type], avatar=avatars[msg.type]).write(msg.content)

if prompt := st.chat_input("Ask a question about Motherson's annual reports (2020-2023)"):
    st.chat_message("user", avatar=avatars["human"]).write(prompt)
    with st.chat_message("assistant", avatar=avatars["ai"]):
        retrieval_handler = PrintRetrievalHandler(st.container())
        box = st.empty()
        stream_handler = StreamHandler(box)
        response = conversation_chain.run(prompt, callbacks=[retrieval_handler, stream_handler])
