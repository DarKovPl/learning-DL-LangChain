from dotenv import load_dotenv, find_dotenv
import openai
import os

load_dotenv(find_dotenv())

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant specialized in providing information about "
                                      "BellaVista Italian Restaurant."},
        {"role": "user", "content": "What's on the menu?"},
        {"role": "assistant", "content": "BellaVista offers a variety of Italian dishes including pasta, pizza, "
                                         "and seafood."},
        {"role": "user", "content": "Do you have vegan options?"}
    ]
)


print(response)
print()
print(response.choices[0].message.content)
