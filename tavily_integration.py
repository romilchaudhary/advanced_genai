from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os
load_dotenv()

tavily_search_tool = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"), max_results=3)

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: The final answer should be returned as a list of dictionaries/rows. The keys to be included are Title, Company, Location, Link. Example: [{{"Title":"","Company":"","Location":"","Link":""}}]

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

llm = ChatOllama(model="mistral:7b-instruct",temperature=0)

agent = create_react_agent(llm, [tavily_search_tool], prompt)

agent_executor = AgentExecutor(agent=agent, tools=[tavily_search_tool])

chain = agent_executor

result = chain.invoke({"input": "find AI engineer job openings for 2 years of experience in Delhi Ncr Area."})
print(result)