from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama # pyright: ignore[reportMissingImports]

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
FIFA 2026 will be the 23rd FIFA World Cup, the quadrennial international"""
    summary_template = """
given the information {information} about a topic I want you to create:
1. A short Summary
2. two interesting facts about the topic
"""
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    #llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm = ChatOllama(model="gemma3:270m", temperature=0)

    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
