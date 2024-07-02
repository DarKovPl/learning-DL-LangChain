from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.callbacks.manager import get_openai_callback
from langchain.callbacks import StdOutCallbackHandler

load_dotenv(find_dotenv())

llm_chat_openai = ChatOpenAI(model="gpt-3.5-turbo")
prompt_template = PromptTemplate(input_variables=["input"], template="Tell me a joke about {input}")
# print("***************************************************************************************************************")
#
# handler = StdOutCallbackHandler()
#
# chain = LLMChain(llm=llm_chat_openai, prompt=prompt_template, callbacks=[handler])
# print(f"chain.invoke: \n{chain.invoke(input='a rabbit')}")
# print("***************************************************************************************************************")
#
#
# class MyCustomHandler(BaseCallbackHandler):
#     def on_llm_end(self, response, **kwargs) -> None:
#         print(f"RESPONSE: ", response)
#
#
# chain_1 = LLMChain(llm=llm_chat_openai, prompt=prompt_template)
# print(f"chain.invoke 1: {chain_1.invoke({'input': 'a rabbit'}, {'callbacks': [MyCustomHandler()]})}")
print("***************************************************************************************************************")

with get_openai_callback() as callback:
    chain_2 = LLMChain(llm=llm_chat_openai, prompt=prompt_template)
    print(f"chain.invoke 2: {chain_2.invoke({'input': 'a rabbit'})}")
print(callback)
print(callback.total_cost)
