from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="mistral:7b-instruct", temperature=0)

def calculator(query: str)-> str:
    """ this is a function used for mathematics calculations"""
    return str(eval(query))

calculator_tool = Tool(
    name = "calculator tool",
    func = calculator,
    description = "tool to be used for mathematical calculation"
)

tools = [calculator_tool]

agent = create_agent(
    model = model,
    tools = tools,
    system_prompt = "You are a helpful research assistant."
)

chain = agent

response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the value of 2+2?"}]
})
print(response)