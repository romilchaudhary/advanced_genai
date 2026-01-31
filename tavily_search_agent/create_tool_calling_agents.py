from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_core.messages import AIMessage

llm = ChatOllama(
    model="mistral:7b-instruct",
    temperature=0
)

def extract_final_answer(messages):
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content.strip():
            return msg.content
    return None

@tool
def calculator(expression: str) -> str:
    """Use this tool for mathematical calculations."""
    try:
        print(f"[TOOL CALLED] calculator with input: {expression}")
        return str(eval(expression))
    except:
        return "Invalid expression"

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=300
    )
)

tools = [calculator, wiki_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system",
    "You MUST use a tool before answering. "
    "If math is involved, use calculator. "
    "If facts are asked, use Wikipedia. "
    "End with FINAL ANSWER."),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant, use tools to answer whenever required and response the final answer"
)

result = agent.invoke(
    {
        "messages":
            [
                {
                    "role": "user",
                    "content": "What is the value of 10+2+(2*3)+(4/2)?"
                }
            ]
    }
)
print("Answer: ", extract_final_answer(result["messages"][::-1]))

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "When was Tata Group founded?"
        }
    ]
})

print("Answer: ", extract_final_answer(result["messages"][::-1]))
