from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["input"],
    template="you are a helpful assistant that can answer questions in one line. {input}?"
)

model = ChatOllama(model="phi3:latest")

response = model.invoke(prompt.format(input="What is the capital of France?"))
print(response.content)
print(type(response))

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

response = model.invoke(messages)
print(response.content)
print(type(response))

chain = prompt | model
response = chain.invoke(input="What is the capital of France?")
print(response.content)
print(type(response))