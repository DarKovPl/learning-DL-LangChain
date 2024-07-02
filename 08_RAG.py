from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import numpy as np
from langchain_community.vectorstores import FAISS
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


load_dotenv(find_dotenv())

loader = TextLoader("./data/bella_vista.txt")
docs = loader.load()
print(f"docs: \n{docs}")
print(len(docs))
print("***************************************************************************************************************")

example_doc = Document(page_content="This is an example document", metadata={"important_info": "Example Document"})
print(f"example_doc: \n{example_doc}")
print("***************************************************************************************************************")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)
documents = text_splitter.split_documents(docs)
for doc in documents:
    print(f"doc: \n{doc}")
print(f"len(documents): \n{len(documents)}")
print("***************************************************************************************************************")

embeddings = OpenAIEmbeddings()

embedding1 = embeddings.embed_query(text="The solar system consists of the Sun and the objects that orbit it")
print(f"embedding1: \n{embedding1}")
print(f"len(embedding1): \n{len(embedding1)}")

embedding2 = embeddings.embed_query(text="The solar system consists of the Sun and the objects that orbit it")
embedding3 = embeddings.embed_query(text="Planets, asteroids, and comets are part of our solar system.")
embedding4 = embeddings.embed_query(text="I love baking chocolate chip cookies on weekends.")


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)


sim_1_2 = cosine_similarity(embedding1, embedding2)
sim_1_3 = cosine_similarity(embedding1, embedding3)
sim_3_4 = cosine_similarity(embedding3, embedding4)

print(sim_1_2, sim_1_3, sim_3_4)
print("***************************************************************************************************************")

print("Loading the database")

folder_path = "./data/bella_vista.faiss"
if not os.path.exists(folder_path):
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(folder_path)

retriever = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True).as_retriever()
print(f"retriever: \n{retriever}")

docs = retriever.invoke("When are the opening hours??") #retriever.get_relevant_documents(query="When are the opening hours??")

for doc in docs:
    print(f"doc: \n{doc}")
print("***************************************************************************************************************")

docs = retriever.invoke(
    input="When are the opening hours?",
    filter={'source': folder_path}, k=3
) ##retriever.get_relevant_documents(query="When are the opening hours??")
for doc_1 in docs:
    print(f"doc_1: \n{doc_1}")  # does not work!
print("___________________________")

vectorstore = FAISS.from_documents(documents, embeddings)
# vectorstore = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"filter": {'source': './data/bella_vista.txt'}, "k": 2})

docs = retriever.invoke(input="When are the opening hours??") #retriever.get_relevant_documents(query="When are the opening hours??")
for doc_2 in docs:
    print(f"doc_2: \n{doc_2}")
print("***************************************************************************************************************")

prompt_template = """You are a helpful assistant for our restaurant.

{context}

Question: {question}
Answer here:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

chain_type_kwargs = {"prompt": prompt}
llm_chat_openai = ChatOpenAI(model="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    chain_type="stuff",
    chain_type_kwargs=chain_type_kwargs,
    retriever=retriever,
    llm=llm_chat_openai
)

result = qa.invoke(input="When are the opening hours on sunday??")
print(f"result: \n{result}")
