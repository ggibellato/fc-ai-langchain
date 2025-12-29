from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()


question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name"
)

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0.5)
                                      
chain = question_template | model

result = chain.invoke({"name": "Gleison"})
print(result.content)