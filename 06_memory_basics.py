from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

load_dotenv(find_dotenv())

history = ChatMessageHistory()
history.add_user_message("Hello")
history.add_ai_message("Hi")
print(f"history.get_messages: {history.messages}")
print("***************************************************************************************************************")

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("Hello")
memory.chat_memory.add_ai_message("Hi")
memory.load_memory_variables({})
print(f"memory.chat_memory.get_messages: {memory.chat_memory.messages}")
print("***************************************************************************************************************")

llm_chat_openai = ChatOpenAI(model="gpt-3.5-turbo")
conversation = ConversationChain(
    llm=llm_chat_openai,
    verbose=True,
    memory=ConversationBufferMemory()
)
print(conversation.invoke("What is the capital of France?"))
print(conversation.invoke("Whats the best food there?"))
print("***************************************************************************************************************")

print("ConversationSummaryMemory")

review = "I ordered Pizza Salami for 9.99$ and it was awesome! \
The pizza was delivered on time and was still hot when I received it. \
The crust was thin and crispy, and the toppings were fresh and flavorful. \
The Salami was well-cooked and complemented the cheese perfectly. \
The price was reasonable and I believe I got my money's worth. \
Overall, I am very satisfied with my order and I would recommend this pizza place to others."

summary_memory = ConversationSummaryBufferMemory(
    llm=llm_chat_openai, max_token_limit=2000
)
summary_memory.save_context(
    {"input": "Hello, how can I help you today?"},
    {"output": "Could you analyze a review for me?"},
)
summary_memory.save_context(
    {"input": "Sure, I'd be happy to. Could you provide the review?"},
    {"output": f"{review}"},
)

conversation = ConversationChain(
    llm=llm_chat_openai,
    verbose=True,
    memory=summary_memory
)
print(conversation.invoke("Thank you very much!"))
print(memory.load_memory_variables({}))
