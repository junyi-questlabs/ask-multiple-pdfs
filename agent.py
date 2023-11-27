from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema.messages import SystemMessage
from langchain.tools import DuckDuckGoSearchResults

import streamlit as st
from datetime import date

st.write("### Ask anything")

# set up the agent
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
search = DuckDuckGoSearchResults()
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="memory", chat_memory=msgs, return_messages=True)
msgs.add_message(SystemMessage(content="You are an Analyst to help Qatar decision maker on geo-location related topics. You are dutiful and do research at best. You are time-sensitive and always answer questions with most up-to-date information. Today's date is " + date.today().strftime("%b-%d-%Y") + "\n"))

# initialize the agent
agent = initialize_agent(
    [Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions. You MUST specify time in query to access most recent results.",
    )],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    },
    memory=memory,
)

roles = {"human": "user", "ai": "assistant"}
avatars = {
    "human": "https://yestherapyhelps.com/images/frases-y-reflexiones/339/75-frases-y-reflexiones-de-michel-foucault-3.jpg", 
    "ai": "https://questlabs.io/favicon.ico"
}
for msg in msgs.messages:
    if isinstance(msg, SystemMessage):
        continue
    st.chat_message(roles[msg.type], avatar=avatars[msg.type]).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user", avatar=avatars["human"]).write(prompt)
    with st.chat_message("assistant", avatar=avatars["ai"]):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
