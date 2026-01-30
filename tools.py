from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import Tool
# from langchain_classic.agents import initialize_agent, AgentType
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

model = ChatOllama(model="mistral:7b-instruct",temperature=0)

def custom_search(query: str) -> str:
    """Search the information using the custom search engine,don't use any other search engine"""
    return "This is a test response from the custom search engine"

custom_search_tool = Tool(
    name="custom_search",
    func=custom_search,
    description="Search the information using the custom search engine"
)

tools = [custom_search_tool]
agent = create_agent(model, tools)
result = agent.invoke({"messages": [("user", "What is the capital of France?")]})
print(result["messages"][-1].content)
# response = agent.invoke({"input": "What is the capital of France?"})
# print(response.content)