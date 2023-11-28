from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain.agents import AgentType, initialize_agent, Tool, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema.messages import SystemMessage
from langchain.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv
import streamlit as st

from datetime import date

load_dotenv()

st.write("### Ask anything")

# set up the agent
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
search = DuckDuckGoSearchRun()
msgs = StreamlitChatMessageHistory()
chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=msgs, return_messages=True)

# initialize the agent
agent = initialize_agent(
    [Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={
        "prefix": "Respond to the human as an Qatar Analyst to help Qatar policy maker on geo-political related topics. You are dutiful and do research at best. You are time-sensitive and always answer questions with most up-to-date information. Today's date is " + date.today().strftime("%b-%d-%Y") + "\n" + ". You have access to the following tools, but remember that these tools may not that smart, and try rephrasing when the results aren't ideal: ",
        "memory_prompts": [chat_history],
        "input_variables": ["input", "agent_scratchpad", "chat_history", ]
    }
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
