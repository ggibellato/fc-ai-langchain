from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


# algo -> traduz ingles -> resumo data traducao

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n ```{initial_text}```"
)

template_summarize = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 5 words:\n ```{text}```\n\n"
)

llm_en = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

translate = template_translate | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summarize | llm_en | StrOutputParser()

result = pipeline.invoke({"initial_text": "LangChain é um framework para desenvolver aplicações de IA."})

print(result)