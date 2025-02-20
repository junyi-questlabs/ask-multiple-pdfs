from typing import List
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema.messages import SystemMessage
from langchain.utilities import GoogleSearchAPIWrapper

from dotenv import load_dotenv
import streamlit as st
import json

from datetime import date

load_dotenv()

st.set_page_config(page_title="Saharaa AI Chat", page_icon="https://saharaa.ai/images/app-logo.png")
st.markdown("<h2 style='text-align: center; color: white;'><img style=\"margin-right: .5em; height: 1em;\" src='https://saharaa.ai/images/app-logo.png'>Saharaa AI Chat</h1>", unsafe_allow_html=True)

from langchain.tools import BaseTool

QUESTIONS = [
    "List the products manufactured in UAE and KSA exported to USA and Europe linked to similar products manufactured in China",
    "List the movement of officials from the MEMA region around the world",
    "Will there be another blockade on Qatar? How soon?"
]

# Sorry for the selectors but Streamlit sucks
styl = f"""
<style>
    div[aria-label="Chat message from assistant"]::before {{
        content: "Saharaa AI Chat";
        line-height: 2.2em;
        font-weight: bold;
    }}

    div[aria-label="Chat message from user"]::before {{
        content: "You";
        line-height: 2.2em;
        font-weight: bold;
    }}

    :nth-child(odd of .stChatMessage) {{
        background-color: #3A44A6;
        padding-right: 1em;
        border-radius: 8px;
    }}

    :nth-child(even of .stChatMessage) {{
        background-color: #2D2F31;;
        padding-right: 1em;
        border-radius: 8px;
    }}

    button > div > p {{
        text-align: left;
    }}

    .stChatInputContainer {{
        background-color: inherit;

    }}

    .stChatInputContainer > div > div {{
        border-bottom-color: grey;
        border-top-color: grey;
        border-left-color: grey;
        border-right-color: grey;
        border-radius: 50px;
    }}

    .stChatInputContainer > div > div:focus-within {{
        border-bottom-color: white;
        border-top-color: white;
        border-left-color: white;
        border-right-color: white;
    }}

    .st-ch::placeholder {{
        color: lightgrey;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

class KSAExportUSData(BaseTool):
    name = "KSA_Export_Data"
    description = "use this tool when you need export data of KSA to US and Europe"

    _ret = """
Following is a list of the top exports of KSA to US of 2021:
Mineral fuels, oils, distillation products 	$2.06B
Organic chemicals 	$533.09M
Fertilizers 	$431.25M
Aluminum 	$335.64M
Aircraft, spacecraft 	$197.96M
Inorganic chemicals, precious metal compound, isotope 	$193.84M
Miscellaneous chemical products 	$177.52M
Plastics 	$162.41M
Machinery, nuclear reactors, boilers 	$156.02M
Articles of iron or steel 	$122.69M
Pearls, precious stones, metals, coins 	$118.01M
Copper 	$92.45M
Manmade filaments 	$68.02M
Electrical, electronic equipment 	$37.87M
Wadding, felt, nonwovens, yarns, twine, cordage 	$33.63M
Iron and steel 	$33.11M
Glass and glassware 	$30.71M
Manmade staple fibers 	$24.63M
Optical, photo, technical, medical apparatus 	$22.78M

Following is a list of the top exports of KSA to Europe of 2022:
Mineral products - €36,489M
Products of the chemical or allied industries - €3,145M
Plastics, rubber and articles thereof - €2,668M
Base metals and articles thereof - €636M
Pearls, precious metals and articles thereof - €604M
"""
    def _run(self, destination: List[str]):
        return self._ret
    
    def _arun(self, destination: List[str]):
        return self._ret

class UAEExportUSData(BaseTool):
    name = "UAE_Export_Data"
    description = "use this tool when you need highlevel export data of UAE to US and Europe, in format of category, revenue"

    _ret = """
Following is a list of the top exports of UAE to US of 2021:
Pearls, precious stones, metals, coins 	$1.84B
Aluminum 	$1.23B
Electrical, electronic equipment 	$1.03B
Machinery, nuclear reactors, boilers 	$580.34M
Aircraft, spacecraft 	$544.34M
Essential oils, perfumes, cosmetics, toileteries 	$386.60M
Iron and steel 	$311.85M
Articles of iron or steel 	$302.15M
Printed books, newspapers, pictures 	$143.52M
Vehicles other than railway, tramway 	$138.50M
Plastics 	$97.08M
Optical, photo, technical, medical apparatus 	$87.26M
Tobacco and manufactures tobacco substitutes 	$86.97M
Clocks and watches 	$82.56M
Works of art, collectors' pieces and antiques 	$79.84M
Furniture, lighting signs, prefabricated buildings 	$54.23M
Footwear, gaiters and the like, 	$45.90M
Glass and glassware 	$42.82M
Paper and paperboard, articles of pulp, paper and board 	$39.89M
Manmade filaments 	$34.64M
Stone, plaster, cement, asbestos, mica or similar materials 	$30.02M
Ships, boats, and other floating structures 	$29.66M
Toys, games, sports requisites 	$28.77M
Miscellaneous articles of base metal 	$23.35M
Articles of apparel, not knit or crocheted 	$23.26M
Miscellaneous chemical products 	$20.91M

Following is a list of the top exports of UAE to Europe of 2022:
Mineral fuels, lubricants, and related materials - €6,083M
Manufactured goods classified chiefly by material - €4,674M
Machinery and transport equipment - €1,649M
Chemicals and related products, n.e.s. - €593M
Miscellaneous manufactured articles - €423M
Mineral products - €6,092M
Base metals and articles thereof - €2,695M
Pearls, precious metals, and articles thereof - €2,217M
Transport equipment - €985M
Machinery and appliances - €691M
"""
    def _run(self, destination: List[str]):
        return self._ret
    
    def _arun(self, destination: List[str]):
        return self._ret

class ChinaManufactureData(BaseTool):
    name = "China_Manufacture_Data"
    description = "use this tool when you need highlevel Manufacture data of China, in format of category, revenue"

    _ret = """
Following is a list of the top Manufactures of China of 2022 (in Thousand USD):
Mechanical & Electrical Products 	165267928.50
Machinery & Transport Equipment 	139506088.89
High-&-new-tech Products 	74811732.00
Handled Wireless Phone and Its Parts 	18738917.05
Mobile Telephone 	18738917.00
Automatic Data Proc. Eq. & Components 	15600784.00
Integrated Circuit 	11178307.80
Motor Vehicles & Chassis 	10273898.81
Digital Automatic Data Proc. Equip. 	8799244.00
Agricultural Products 	8283537.36
Plastic Products 	7902291.00
Iron & Steel Products 	7283144.16
Auto Parts 	6840770.00
Steel Products 	6288031.00
Food & Live Animals 	6195023.41
Furniture & Related Products 	5256500.00
Iron & Steel 	5235163.39
Refined Petroleum Oil (value) 	4248021.00
"""
    def _run(self):
        return self._ret
    
    def _arun(self):
        return self._ret

class ChinaExportUSData(BaseTool):
    name = "China_Export_US_Data"
    description = "use this tool when you need highlevel export data of China to US, in format of category, revenue"

    _ret = """
Following is a list of the top exports of China to US of 2022:
Electrical, electronic equipment 	$142.56B
Machinery, nuclear reactors, boilers 	$109.64B
Toys, games, sports requisites 	$36.96B
Furniture, lighting signs, prefabricated buildings 	$34.54B
Plastics 	$27.37B
Articles of apparel, knit or crocheted 	$21.68B
Vehicles other than railway, tramway 	$19.67B
Articles of iron or steel 	$15.99B
Articles of apparel, not knit or crocheted 	$15.17B
Footwear, gaiters and the like, 	$13.28B
Optical, photo, technical, medical apparatus 	$12.63B
Organic chemicals 	$11.87B
Other made textile articles, sets, worn clothing 	$11.33B
Commodities not specified according to kind 	$11.08B
Articles of leather, animal gut, harness, travel good 	$6.97B
Miscellanneous manufactured articles 	$5.77B
Miscellaneous articles of base metal 	$5.56B
Bird skin, feathers, artificial flowers, human hair 	$5.52B 
"""
    def _run(self):
        return self._ret
    
    def _arun(self):
        return self._ret
    

RESPONSE_KEYS = ["title", "link", "snippet"]
class GoogleSearchWithRefWrapper(GoogleSearchAPIWrapper):
    def run(self, query: str) -> str:
        """Run query through GoogleSearch and parse result."""
        snippets = []
        results = self._google_search_results(query, num=self.k)
        if len(results) == 0:
            return "No good Google Search Result was found"
        for result in results:
            if "snippet" in result:
                snippets.append(json.dumps({k: result[k] for k in RESPONSE_KEYS}, indent=4))

        return "\n\n".join(snippets)


# set up the agent
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, streaming=True)
search = GoogleSearchWithRefWrapper()
msgs = StreamlitChatMessageHistory()
chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = ConversationBufferMemory(
    memory_key="chat_history", k=6, chat_memory=msgs, return_messages=True)

from knowledge import prepare_pdfs

vs = prepare_pdfs()
knowledge_retriever = vs.as_retriever(search_kwargs={"k": 10})

# initialize the agent
agent = initialize_agent(
    [Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events with source references for fact checking, try rephrasing tool_input when the result isn't ideal e.g. attaching word recent",
    ),
    KSAExportUSData(), UAEExportUSData(), ChinaManufactureData(),
    Tool(
        name="Annual Report",
        func=knowledge_retriever.invoke,
        description="useful for when you need to answer questions about Annual Reports of Motherson with source references for fact checking, try rephrasing tool_input when the result isn't ideal",
    )],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_kwargs={
        "prefix": "Respond to the human as an Qatar Analyst to help human policy maker on geo-political related topics, especially China-US tension and trade war since 2018. You are dutiful and do research at best. You are time-sensitive and always answer questions with most up-to-date information. Your sole purpose is to write well written, critically acclaimed, objective and structured reports to human policy maker, ALWAYS with source references for fact checking. Today's date is " + date.today().strftime("%b-%d-%Y") + "\n" + ". You have access to the following tools: ",
        "format_instructions": """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```
```
{{{{
  "action": "Search",
  "action_input": {{{{'tool_input': 'Recent diplomatic visits by Gulf officials'}}}}
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```""",
        "memory_prompts": [chat_history],
        "input_variables": ["input", "agent_scratchpad", "chat_history", ]
    }
)

roles = {"human": "user", "ai": "assistant"}
avatars = {
    "human": "https://yestherapyhelps.com/images/frases-y-reflexiones/339/75-frases-y-reflexiones-de-michel-foucault-3.jpg", 
    "ai": "https://saharaa.ai/images/app-logo.png"
}

for msg in msgs.messages:
    if isinstance(msg, SystemMessage):
        continue
    st.chat_message(roles[msg.type], avatar=avatars[msg.type]).write(msg.content)

prompt = st.session_state.get("prompt", None)
# Onboarding Box
if "onboarded" not in st.session_state:
    st.markdown("Ask anything, or start from questions you may like to ask:")
    if st.button(QUESTIONS[0]):
        prompt = QUESTIONS[0]
    if st.button(QUESTIONS[1]):
        prompt = QUESTIONS[1]
    if st.button(QUESTIONS[2]):
        prompt = QUESTIONS[2]

prompt = st.chat_input(placeholder="Ask any question") or prompt
st.session_state["prompt"] = prompt

if prompt:
    if "onboarded" not in st.session_state:
        st.session_state["onboarded"] = True
        st.rerun()
    st.empty() # Magic hack to hide the onboarding box instantly
    st.chat_message("user", avatar=avatars["human"]).write(prompt)
    # To hide the onboarding box
    with st.chat_message("assistant", avatar=avatars["ai"]):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
        # proposing further questions ... 
