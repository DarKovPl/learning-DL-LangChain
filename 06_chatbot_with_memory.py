import streamlit as st
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.globals import get_verbose
from dotenv import load_dotenv, find_dotenv

get_verbose()
load_dotenv(find_dotenv())

template = """You are an AI chatbot having a conversation with a human.

{history}
Human: {human_input}
AI: """

prompt_template = PromptTemplate(
    input_variables=["history", "human_input"], template=template
)

memory = ConversationBufferMemory(memory_key="history")


def load_chain():
    llm_chat_openai = ChatOpenAI(model="gpt-3.5-turbo")
    llm_chain = LLMChain(llm=llm_chat_openai, prompt=prompt_template, memory=memory)
    return llm_chain


def initialize_session_state():
    if "chain" not in st.session_state:
        st.session_state.chain = load_chain()

    if "generated" not in st.session_state:
        st.session_state.generated = []

    if "past" not in st.session_state:
        st.session_state.past = []


initialize_session_state()

st.set_page_config(page_title="LangChain ChatBot Demo", page_icon=":robot:")
st.header("LangChain ChatBot Demo")
user_input = st.chat_input("You:")

if user_input:
    output = st.session_state.chain.invoke(user_input)["text"]
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

for user_msg, bot_msg in zip(st.session_state.past, st.session_state.generated):
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("bot"):
        st.markdown(bot_msg)

if len(st.session_state.past) > len(st.session_state.generated):
    with st.chat_message("user"):
        st.markdown(st.session_state.past[-1])
