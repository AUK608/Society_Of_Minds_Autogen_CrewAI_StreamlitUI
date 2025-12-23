
import os
import chromadb
import autogen
from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent
import torch
import ast
from dotenv import load_dotenv

###load environment variables
load_dotenv()

# API Key Configuration 
groq_api = os.getenv('GROQ_API_KEY')
ag_docker = os.getenv('AUTOGEN_USE_DOCKER')

torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch', '_classes.py')]

# Configure
config_list = [{
    "model": "llama3-70b-8192",
    "api_key": groq_api,
    "api_type": "groq"
}]

llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent(
    "inner-assistant",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

code_interpreter = autogen.UserProxyAgent(
    "inner-code-interpreter",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    default_auto_reply="",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

groupchat = autogen.GroupChat(
    agents=[assistant, code_interpreter],
    messages=[],
    speaker_selection_method="round_robin",  # With two agents, this is equivalent to a 1:1 conversation.
    allow_repeat_speaker=False,
    max_round=8,
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
)


society_of_mind_agent = SocietyOfMindAgent(
    "society_of_mind",
    chat_manager=manager,
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="",
    is_termination_msg=lambda x: True,
)


def rag_chat(prompt):

    user_proxy.initiate_chat(society_of_mind_agent, message=prompt)

    messages = user_proxy.chat_messages
    messages = [messages[k] for k in messages.keys()][0]
    result = [m["content"] for m in messages if m["role"] == "user"]
    result = ast.literal_eval(result[0])['content']

    print("messages = = = = ", messages)

    return result



import streamlit as st

st.title("Society Of Minds AutoGen Chatbot")
st.markdown(
    "Ask Question about Coding Content"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Questions about Coding Content Which will be Implemented and Checked..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = rag_chat(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
