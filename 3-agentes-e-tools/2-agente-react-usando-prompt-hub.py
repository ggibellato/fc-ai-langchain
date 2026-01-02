from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client
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

# Pull the ReAct prompt from LangSmith Hub
client = Client()
prompt = client.pull_prompt("hwchase17/react")

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
