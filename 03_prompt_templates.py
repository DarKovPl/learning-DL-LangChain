from dotenv import load_dotenv, find_dotenv
from langchain.prompts import load_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

load_dotenv(find_dotenv())

print("Prompt Templates")

TEMPLATE = """
You are a helpful assistant that translates the {input_language} to {output_language}
"""

prompt_template = PromptTemplate(template=TEMPLATE, input_variables=["input_language", "output_language"])
print(prompt_template.format(input_language="English", output_language="Polish"))
print("***************************************************************************************************************")

print("Few Shot Prompt - provide a few examples in the template - I")
TEMPLATE = """
Interprete the text and evaluate the text.
sentiment: is the text in a positive, neutral or negative sentiment?
subject: What subject is the text about? Use exactly one word.

Format the output as JSON with the following keys:
sentiment
subject

text: {input}

Examples: text: The BellaVista restaurant offers an exquisite dining experience. 
The flavors are rich and the presentation is impeccable. 
sentiment: positive 
subject: BellaVista

text: BellaVista restaurant was alright. The food was decent, but nothing stood out.
sentiment: neutral
subject: BellaVista

text: I was disappointed with BellaVista. The service was slow and the dishes lacked flavor.
sentiment: negative
subject: BellaVista

text: SeoulSavor offered the most authentic Korean flavors I've tasted outside of Seoul.
The kimchi was perfectly fermented and spicy.
sentiment: positive
subject: SeoulSavor

text: SeoulSavor was okay. The bibimbap was good but the bulgogi was a bit too sweet for my taste.
sentiment: neutral
subject: SeoulSavor

text: I didn't enjoy my meal at SeoulSavor. The tteokbokki was too mushy and the service was not attentive.
sentiment: negative
subject: SeoulSavor

text: MunichMeals has the best bratwurst and sauerkraut I've tasted outside of Bavaria. 
Their beer garden ambiance is truly authentic.
sentiment: positive
subject: MunichMeals

text: MunichMeals was alright. The weisswurst was okay, but I've had better elsewhere.
sentiment: neutral
subject: MunichMeals

text: I was let down by MunichMeals. The potato salad lacked flavor and the staff seemed uninterested.
sentiment: negative
subject: MunichMeals
"""

prompt_template = PromptTemplate(template=TEMPLATE, input_variables=["input"])
print(prompt_template.format(input="The MunichDeals experience was just awesome!"))
print("***************************************************************************************************************")

print("Few Shot Prompt - provide a few examples in the template - II")
examples = [
    {
        "text": "The BellaVista restaurant offers an exquisite dining experience. "
                "The flavors are rich and the presentation is impeccable.",
        "response": "sentiment: positive\nsubject: BellaVista"
    },
    {
        "text": "BellaVista restaurant was alright. The food was decent, but nothing stood out.",
        "response": "sentiment: neutral\nsubject: BellaVista"
    }
]

new_example = {
    "text": "SeoulSavor was okay. The bibimbap was good but the bulgogi was a bit too sweet for my taste.",
    "response": "sentiment: neutral\nsubject: SeoulSavor"
}
examples.append(new_example)
example_prompt = PromptTemplate(template="text: {text}\n{response}", input_variables=["text", "response"])

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="text: {input}",
    input_variables=["input"]
)
print(prompt.format(input="The MunichDeals experience was just awesome!"))
print("***************************************************************************************************************")

print("Chain-of-thought Prompting")

TEMPLATE = """
Interprete the text and evaluate the text. Determine if the text has a positive, neutral, or negative sentiment. 
Also, identify the subject of the text in one word.

Format the output as JSON with the following keys:
sentiment
subject

text: {input}

Chain-of-Thought Prompts:
Let's start by evaluating a statement. Consider: "The BellaVista restaurant offers an exquisite dining experience. 
 The flavors are rich and the presentation is impeccable." How does this make you feel about BellaVista?
 It sounds like a positive review for BellaVista.

Based on the positive nature of that statement, how would you format your response?
 {{ "sentiment": "positive", "subject": "BellaVista" }}

Now, think about this: "SeoulSavor was okay. The bibimbap was good but the bulgogi was a bit too sweet for my taste." 
 Does this give a strong feeling either way?
 Not particularly. It seems like a mix of good and not-so-good elements, so it's neutral.

Given the neutral sentiment, how should this be presented?
 {{ "sentiment": "neutral", "subject": "SeoulSavor" }}

Lastly, ponder on this: "I was let down by MunichMeals. The potato salad lacked flavor and the staff seemed uninterested."
 What's the overall impression here?
 The statement is expressing disappointment and dissatisfaction.

And if you were to categorize this impression, what would it be?
 {{ "sentiment": "negative", "subject": "MunichMeals" }}
"""
prompt_template = PromptTemplate(template=TEMPLATE, input_variables=["input"])
formated_prompt = prompt_template.format(input="The MunichDeals experience was just awesome!")
print(formated_prompt)
print("***************************************************************************************************************")

print("Composition of Prompts")

introduction_template = """
Interprete the text and evaluate the text. Determine if the text has a positive, neutral, or negative sentiment. 
Also, identify the subject of the text in one word.
"""
introduction_prompt = PromptTemplate.from_template(introduction_template)
print(f"Introduction prompt: {introduction_prompt}")

example_template = """
Chain-of-Thought Prompts:
Let's start by evaluating a statement. Consider: "{example_text}". How does this make you feel about {example_subject}?
Response: {example_evaluation}

Based on the {example_sentiment} nature of that statement, how would you format your response?
Response: {example_format}
"""
example_prompt = PromptTemplate.from_template(example_template)
print(f"Example prompt: {example_prompt}")

execution_template = """
Now, execute this process for the text: "{input}".
"""
execution_prompt = PromptTemplate.from_template(execution_template)
print(f"Execution prompt: {execution_prompt}")

full_template = """{introduction}{examples}{execution}"""
full_prompt = PromptTemplate.from_template(full_template)
print(f"Full prompt: {full_prompt}")

input_prompts = [
    ("introduction", introduction_prompt),
    ("examples", example_prompt),
    ("execution", execution_prompt)
]
print(f"Input prompts: {input_prompts}")

pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
print(f"Pipeline prompt: {pipeline_prompt}")
print()

print(pipeline_prompt.format(
    example_text="The BellaVista restaurant offers an exquisite dining experience. "
                 "The flavors are rich and the presentation is impeccable.",
    example_subject="BellaVista",
    example_evaluation="It sounds like a positive review for BellaVista.",
    example_sentiment="positive",
    example_format='{{ "sentiment": "positive", "subject": "BellaVista" }}',
    input="The new restaurant downtown has bland dishes and the wait time is too long."
))
print("***************************************************************************************************************")

print("Serializing prompts")

prompt = PromptTemplate(input_variables=["input"], template="Tell me a joke about {input}")
prompt.save("./prompts/prompt.yaml")
prompt.save("./prompts/prompt.json")

loaded_prompt = load_prompt("./prompts/prompt.yaml")
chickens = loaded_prompt.format(input="chickens")
print(f"yaml: {chickens}")

loaded_prompt = load_prompt("./prompts/prompt.json")
cows = loaded_prompt.format(input="cows")
print(f"json: {cows}")
print("***************************************************************************************************************")
