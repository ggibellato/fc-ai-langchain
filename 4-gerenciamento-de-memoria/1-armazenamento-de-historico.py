from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
    
chat_model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq", temperature=0.5)

chain = prompt | chat_model

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_message_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "demo-session"}}
configb = {"configurable": {"session_id": "demo-session-1"}}

response1 = conversational_chain.invoke({"input": "Hello, my name is Wesley. how are you?"}, config=config)
print("Assistant: ", response1.content)
print("-"*30)

response1b = conversational_chain.invoke({"input": "Hello, my name is Gleison. how are you?"}, config=configb)
print("Assistant: ", response1b.content)
print("-"*30)

response2 = conversational_chain.invoke({"input": "Can you repeat my name?"}, config=config)
print("Assistant: ", response2.content)
print("-"*30)

response2b = conversational_chain.invoke({"input": "Can you repeat my name?"}, config=configb)
print("Assistant: ", response2b.content)
print("-"*30)

response3 = conversational_chain.invoke({"input": "Can you repeat my name in a motivation phrase?"}, config=config)
print("Assistant: ", response3.content)
print("-"*30)

response4b = conversational_chain.invoke({"input": "My surname is Silva"}, config=configb)
print("Assistant: ", response4b.content)
print("-"*30)

response5 = conversational_chain.invoke({"input": "Can you tell my full name?"}, config=config)
print("Assistant: ", response5.content)
print("-"*30)

response5b = conversational_chain.invoke({"input": "Can you tell my full name?"}, config=configb)
print("Assistant: ", response5b.content)
print("-"*30)
