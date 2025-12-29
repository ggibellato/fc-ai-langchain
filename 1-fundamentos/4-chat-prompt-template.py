from langchain_core.prompts import ChatPromptTemplate
#from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

system = ("system", "You are an assistant that answers questions in a {style} style.")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(style="funny", question="Who is Alan Turing?")

for msg in messages:
    print(f"{msg.type}: {msg.content}")

#model = ChatOpenAI(model_name="gpt-5-nano", temperature=0.5)    
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)    
result = model.invoke(messages)
print(result.content)