import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schema import AgentResponse

load_dotenv()

llm = ChatOllama(model="mistral:7b-instruct", temperature=0)
structured_llm = llm.with_structured_output(AgentResponse)

tavily_search_tool = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions="")

agent = create_react_agent(
    llm=llm, tools=[tavily_search_tool], prompt=react_prompt_with_format_instructions
)

extract_output = RunnableLambda(lambda x: x["output"])

agent_executor = AgentExecutor(
    agent=agent, tools=[tavily_search_tool], verbose=True
)

chain = agent_executor | extract_output | structured_llm

result = chain.invoke(
    input=
    {
        "input": "search for 3 job postings for an ai engineer using langchain in the Delhi NCR on linkedin and list their details",
    }
)
print(result)
