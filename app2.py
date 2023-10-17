from typing import List


import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreInfo,
)
from langchain.tools.vectorstore.tool import (
    BaseVectorStoreTool,
)
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.llms.openai import OpenAI
from langchain.pydantic_v1 import Field
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

from PyPDF2 import PdfReader
import os
import pickle

PDF_DIR = "/Users/jun/Downloads/pdfs"

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

@st.cache_data
def prepare_pdfs():
    pgbar = st.progress(0, "Preparing Knowledges...")
    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    handles = [open(os.path.join(PDF_DIR, pdf), "rb") for pdf in pdfs]
    data = get_pdf_text(handles)
    # Close the files
    for pdf in handles:
        pdf.close()

    # get the text chunks
    text_chunks = get_text_chunks(data)

    # create vector store
    vectorstore = get_vectorstore(pgbar, text_chunks)

    return vectorstore


class CustomVectorStoreQAWithSourcesTool(BaseVectorStoreTool, BaseTool):
    """Tool for the VectorDBQAWithSources chain."""

    @staticmethod
    def get_description(name: str, description: str) -> str:
        template: str = (
            "Useful for when you need to answer questions about {name} and the sources "
            "used to construct the answer. "
            "Whenever you need information about {description} "
            "you should ALWAYS use this. "
            " Input should be a fully formed question. "
            "Output is a json serialized dictionary with keys `answer` and `sources`. "
            "Only use this tool if the user explicitly asks for sources."
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

        from langchain.chains.qa_with_sources.retrieval import (
            RetrievalQAWithSourcesChain,
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            self.llm, retriever=self.vectorstore.as_retriever(search_kwargs={"k": 10})
        )
        return json.dumps(
            chain(
                {chain.question_key: query},
                return_only_outputs=True,
                callbacks=run_manager.get_child() if run_manager else None,
            )
        )


class CustomToolkit(BaseToolkit):
    """Toolkit for interacting with a Vector Store."""

    vectorstore_info: VectorStoreInfo = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: OpenAI(temperature=0))

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        description = "Semantic query into the knowledge base. THINK STEP BY STEP. ALWAYS PROVIDE QUOTES AND SOURCE CITIATIONS. BREAK COMPLEX QUESTIONS INTO SEVERAL QUERIES."
        qa_with_sources_tool = CustomVectorStoreQAWithSourcesTool(
            name=f"{self.vectorstore_info.name}_with_sources",
            description=description,
            vectorstore=self.vectorstore_info.vectorstore,
            llm=self.llm,
        )
        return [qa_with_sources_tool]


vectorstore_info = VectorStoreInfo(
    name="knowledge base",
    description="extra knowledge and context for fact checking",
    vectorstore=prepare_pdfs(),
)
toolkit = CustomToolkit(
    vectorstore_info=vectorstore_info,
    llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
)

llm = ChatOpenAI(temperature=0.5, streaming=True, model_name="gpt-3.5-turbo")
agent_executor = create_vectorstore_agent(
    llm=llm, 
    toolkit=toolkit,
    agent_executor_kwargs={
        "handle_parsing_errors": True,
    },
    prefix="System: You are an agent to interactive chats with grounded knowledge, complete with source references for fact checking.")

st.write("### Ask anything with your own documents")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(prompt, callbacks=[st_callback])
        st.write(response)
