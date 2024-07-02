from dotenv import load_dotenv, find_dotenv
import openai
import json
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.openai_functions import create_openai_fn_chain


load_dotenv(find_dotenv())

# def chat(query):
#     response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are an AI chatbot having a conversation with a human."},
#             {"role": "user", "content": query},
#         ],
#     )
#     messages = response.choices[0].message.content
#     return messages
#
#
# query = "How much does pizza salami cost?"true story
# message = chat(query)
# print(f"message: \n{message}")
print("***************************************************************************************************************")

print("Function calling in OpenAI")


def get_pizza_info(pizza_name: str):
    pizza_info = {
        "pizza_salami": {
            "price": 10,
            "ingredients": ["tomato sauce", "mozzarella", "salami"],
            "size": "medium",
        },
        "pizza_margherita": {
            "price": 8,
            "ingredients": ["tomato sauce", "mozzarella", "basil"],
            "size": "medium",
        },
        "pizza_hawaii": {
            "price": 12,
            "ingredients": ["tomato sauce", "mozzarella", "ham", "pineapple"],
            "size": "medium",
        },
    }
    return json.dumps(pizza_info[pizza_name])


functions = [
    {
        "name": "get_pizza_info",
        "description": "Get name and price of a pizza of the restaurant. Check description for the pizza name format.",
        "parameters": {
            "type": "object",
            "properties": {
                "pizza_name": {
                    "type": "string",
                    "description": "The name of the pizza, e.g. Salami and format it as pizza_* e.g 'pizza_salami'."
                                   "Pizza name's always lowercase. Correct typos in the pizza name."
                },
            },
            "required": ["pizza_name"],
        },
    }
]
#
#
# def chat(query):
#     response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": query}],
#         functions=functions
#     )
#     message = response.choices[0].message
#     return message
#
#
# query = "Pizza barbecue??"
# message = chat(query)
# print(f"message: {message}")
#
# if message.function_call:
#     pizza_name = json.loads(message.function_call.arguments).get("pizza_name")
#     print(f"pizza_name: {pizza_name}")
#     function_response = get_pizza_info(
#         pizza_name=pizza_name
#     )
#     print(f"function_response: {function_response}")
#
#     second_response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "user", "content": query},
#             message,
#             {
#                 "role": "function",
#                 "name": "get_pizza_info",
#                 "content": function_response,
#             },
#         ],
#     )
#     print(f"second_response: {second_response}")
print("***************************************************************************************************************")

# print("The same can be achieved with LangChain")
#
# llm_chat_openai = ChatOpenAI(model="gpt-3.5-turbo")
# template = """You are an AI chatbot having a conversation with a human.
#
# Human: {human_input}
# AI: """
# prompt_template = PromptTemplate(
#     input_variables=["human_input"], template=template
# )
# chain = create_openai_fn_chain(functions=functions, llm=llm_chat_openai, prompt=prompt_template, verbose=True)
# query = "How much does pizza salami cost?"
# message = chain.invoke({"human_input": query})
# print(f"message: {message}")
#
# if 'function' in message:
#     function_call = message['function']
#     pizza_name = function_call.get("pizza_name")
#     print(f"pizza_name: {pizza_name}")
#     function_response = get_pizza_info(
#         pizza_name=pizza_name
#     )
#     print(f"function_response: {function_response}")
#
#     second_response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "user", "content": query},
#             message,
#             {
#                 "role": "function",
#                 "name": "get_pizza_info",
#                 "content": function_response,
#             },
#         ],
#     )
#     print(f"second_response: {second_response}")
# else:
#     print("No function call in the response")
# print("***************************************************************************************************************")

print("Pydantic Classes instead of JSON Schemas")

