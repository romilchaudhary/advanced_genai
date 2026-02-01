from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

@tool("calculator_tool", description="perform mathematical operations, use this for any math problem")
def calculator(expression: str)-> str:
    """ funcction to use to calculate mathematical expressions """
    print(f"[TOOL CALLED] calculator with input: {expression}")
    return str(eval(expression))

tools = [calculator]

llm = ChatOllama(
    model="mistral:7b-instruct",
    temperature=0
).bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI agent. Use tools when needed."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(
    llm = llm,
    tools = tools,
    prompt = prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

result = agent_executor.invoke({
    "input": "What is 10 + 2 + (2 * 3) + (4 / 2)?"
})

print(result)
