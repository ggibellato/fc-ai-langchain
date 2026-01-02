from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expresion: str) -> str:
    """Evaluate a simple mathematical expression and return the result."""
    try:
        result = eval(expresion) # be careful with this because it's a security risk
        return str(result)
    except Exception as e:
        return f"Error {e}"
    
@tool("websearchmock")
def websearchmock(query: str) -> str:
    """Mocked web search tool. Returns a hardcoded result."""
    
    data = {
        "Brazil": "Brasília",
        "France": "Paris",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Spain": "Madrid",
        "United States": "Washington, D.C."
    }

    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."
    return str("I don't know the capital of that country.")

llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq", temperature=0.5)
tools = [calculator, websearchmock]

# Define the ReAct prompt template
prompt = PromptTemplate.from_template(
"""
Answer the following questions as best you can. You have access to the following tools.
Only use the information you get from the tools, even if you know the answer.
If the information is not provided by the tools, say you don't know.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
- If you choose an Action, do NOT include Final Answer in the same step.
- After Action and Action Input, stop and wait for Observation.
- Never search the internet. Only use the tools provided.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# Create the ReAct agent using langchain-classic
agent = create_react_agent(llm, tools, prompt)

# Create the executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

print(agent_executor.invoke({"input": "What is the capital of Iran?"}))
#print(agent_executor.invoke({"input": "What is the capital of Brazil?"}))
#print(agent_executor.invoke({"input": "How much is 10 + 10?"}))
