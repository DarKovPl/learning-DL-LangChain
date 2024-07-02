from dotenv import load_dotenv, find_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain, MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE


load_dotenv(find_dotenv())

llm_chat_openai = ChatOpenAI(model="gpt-3.5-turbo")
print("***************************************************************************************************************")

# print("One 'input' variable")
#
# prompt_template = PromptTemplate(input_variables=["input"], template="Tell me a joke about {input}")
# chain = LLMChain(llm=llm_chat_openai, prompt=prompt_template)
# print(f"chain.invoke: \n{chain.invoke(input='a parrot')}")
print("***************************************************************************************************************")

# print("Multiple 'input' variables")
#
# prompt_template = PromptTemplate(
#     input_variables=["input", "language"], template="Tell me a joke about {input} in {language}"
# )
#
# chain = LLMChain(llm=llm_chat_openai, prompt=prompt_template)
# print(f"chain.invoke: \n{chain.invoke({'input': 'a parrot', 'language': 'polish'})}")
# print("***************************************************************************************************************")
#
# print("Sequential chains")
#
# prompt_review = PromptTemplate.from_template(
#     template="You ordered {dish_name} and your experience was {experience}. Write a review: "
# )
# chain_review = LLMChain(llm=llm_chat_openai, prompt=prompt_review, output_key="review")
# print("_____________________________________")
#
# prompt_comment = PromptTemplate.from_template(
#     template="Given the restaurant review: {review}, write a follow-up comment: "
# )
# chain_comment = LLMChain(llm=llm_chat_openai, prompt=prompt_comment, output_key="comment")
# print("_____________________________________")
#
# prompt_summary = PromptTemplate.from_template(
#     template="Summarise the review in one short sentence: \n\n {comment}"
# )
# chain_summary = LLMChain(llm=llm_chat_openai, prompt=prompt_summary, output_key="summary")
# print("_____________________________________")
#
# prompt_translation = PromptTemplate.from_template(
#     template="Translate the summary to polish: \n\n {summary}"
# )
# chain_translation = LLMChain(
#     llm=llm_chat_openai, prompt=prompt_translation, output_key="polish_translation"
# )
#
# overall_chain = SequentialChain(
#     chains=[chain_review, chain_comment, chain_summary, chain_translation],
#     input_variables=["dish_name", "experience"],
#     output_variables=["review", "comment", "summary", "polish_translation"],
#
# )
# print(f"overall_chain.invoke: \n{overall_chain.invoke({'dish_name': 'Pizza Salami', 'experience': 'It was awful!'})}")
print("***************************************************************************************************************")

print("...LLM to decide which follow up chain is being used")

positive_template = """You are an AI that focuses on the positive side of things. \
Whenever you analyze a text, you look for the positive aspects and highlight them. \
Here is the text:
{input}"""

neutral_template = """You are an AI that has a neutral perspective. You just provide a balanced analysis of the text, \
not favoring any positive or negative aspects. Here is the text:
{input}"""

negative_template = """You are an AI that is designed to find the negative aspects in a text. \
You analyze a text and show the potential downsides. Here is the text:
{input}"""

prompt_infos = [
    {
        "name": "positive",
        "description": "Good for analyzing positive sentiments",
        "prompt_template": positive_template,
    },
    {
        "name": "neutral",
        "description": "Good for analyzing neutral sentiments",
        "prompt_template": neutral_template,
    },
    {
        "name": "negative",
        "description": "Good for analyzing negative sentiments",
        "prompt_template": negative_template,
    },
]

destination_chains = {}
for prompt_info in prompt_infos:
    prompt = PromptTemplate(template=prompt_info["prompt_template"], input_variables=["input"])
    chain = LLMChain(llm=llm_chat_openai, prompt=prompt)
    destination_chains[prompt_info["name"]] = chain
print(f"destination_chains: \n{destination_chains}")

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
print(f"destinations_str: \n {destinations_str}")

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
print(f"router_template: \n{router_template}")

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)
print(f"router_prompt: \n{router_prompt}")

router_chain = LLMRouterChain.from_llm(
    llm_chat_openai,
    router_prompt
)

multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=destination_chains["neutral"],
    verbose=True
)

print(f"multi_prompt_chain.invoke: "
      f"\n{multi_prompt_chain.invoke(input='I ordered Pizza Salami for 9.99$ and it was awesome!')}")
