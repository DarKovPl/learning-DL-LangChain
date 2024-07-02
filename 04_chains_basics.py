from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

load_dotenv(find_dotenv())

print("Chains Basics")

# TEMPLATE = """
# Interprete the text and evaluate the text.
# sentiment: is the text in a positive, neutral or negative sentiment? Sentiment is required.
# subject: What subject is the text about? Use exactly one word. Use 'None' if no subject was provided.
# price: How much did the customer pay? Use 'None' if no price was provided.
#
# Format the output as JSON with the following keys:
# sentiment
# subject
# price
#
# text: {input}
# """

llm_chat_openai = ChatOpenAI(model="gpt-3.5-turbo")
# prompt_template = ChatPromptTemplate.from_template(template=TEMPLATE)
#
# chain = LLMChain(llm=llm_chat_openai, prompt=prompt_template)
# chain.invoke(input="I ordered pizza salami from the restaurant Bellavista. "
#                    "It was ok, but the dough could have been a bit more crisp."
#              )
print("***************************************************************************************************************")
print("Response Schemas")

response_schemas = [
    ResponseSchema(
        name="sentiment",
        description="is the text in a positive, neutral or negative sentiment? Sentiment is required."
    ),
    ResponseSchema(
        name="subject",
        description="What subject is the text about? Use exactly one word. Use None if no price was provided."
    ),
    ResponseSchema(
        name="price",
        description="How much did the customer pay? Use None if no price was provided.",
        type="float"
    )
]
print(f"response_schemas: \n{response_schemas}")

structured_output_parser = StructuredOutputParser(response_schemas=response_schemas)
format_instructions = structured_output_parser.get_format_instructions()
print(f"format_instructions: \n{format_instructions}")
print("***************************************************************************************************************")

print("Create prompt template")

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Interprete the text and evaluate the text. "
            "sentiment: is the text in a positive, neutral or negative sentiment? "
            "subject: What subject is the text about? Use exactly one word. "
            "Just return the JSON, do not add ANYTHING, NO INTERPRETATION! "
            "text: {input}\n"
            "{format_instructions}\n"
        )
    ],
    input_variables=["input"],
    partial_variables={"format_instructions": format_instructions}
)

_input = prompt.format_prompt(input="I ordered pizza salami from the restaurant Bellavista. "
                                    "It was ok, but the dough could have been a bit more crisp."
                              )
print(f"_input.to_messages(): \n{_input.to_messages()}")

output = llm_chat_openai.invoke(_input.to_messages())
print(f"output: \n{output.content}")

json_output = structured_output_parser.parse(output.content)
print(f"json_output: \n{json_output}")
print(f"json_output.get('sentiment'): \n{json_output.get('sentiment')}")
