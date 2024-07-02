from langchain_openai import OpenAI, ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv(find_dotenv())

# openai_single = OpenAI(model="gpt-3.5-turbo-instruct")
# # print(openai_single.invoke("Tell me a joke."))
print("***************************************************************************************************************")

# openai_batch = OpenAI(model="gpt-3.5-turbo-instruct")
# result = openai_batch.generate(["Tell me a joke about cows.", "Tell me a joke about parrots."])
#
# print(result)
# print()
# print(result.llm_output)
print("***************************************************************************************************************")

# openai_chat = ChatOpenAI(model="gpt-3.5-turbo")
# result = openai_chat.invoke("Tell me a joke about cows")
#
# print(result)
# print()
# print(result.response_metadata)
print("***************************************************************************************************************")

# messages = [
#     SystemMessage(content="You are a helpful assistant specialized in providing information about BellaVista Italian "
#                           "Restaurant."),
#     HumanMessage(content="What's on the menu?"),
#     AIMessage(content="BellaVista offers a variety of Italian dishes including pasta, pizza, and seafood."),
#     HumanMessage(content="Do you have vegan options?")
# ]
#
# openai_chat = ChatOpenAI(model="gpt-3.5-turbo")
# result = openai_chat.invoke(messages)
# print(result)
print("***************************************************************************************************************")

# batch_messages = [
#     [
#         SystemMessage(content="You are a helpful assistant that translates English to Polish."),
#         HumanMessage(content="Do you have vegan options?")
#     ],
#     [
#         SystemMessage(content="You are a helpful assistant that translates the English to Spanish."),
#         HumanMessage(content="Do you have vegan options?")
#     ],
# ]
#
# openai_chat_batch = ChatOpenAI(model="gpt-3.5-turbo")
# batch_result = openai_chat_batch.generate(batch_messages)
# print(batch_result)
#
# translations = [generation[0].text for generation in batch_result.generations]
# print(translations)
print("***************************************************************************************************************")
