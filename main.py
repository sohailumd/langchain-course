from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama # pyright: ignore[reportMissingImports]

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
Lionel Andrés "Leo" Messi[note 1] (born 24 June 1987) is an Argentine professional footballer who plays as a forward for and captains both the Major League Soccer club Inter Miami and the Argentina national team. Widely regarded as one of the greatest players in history, Messi has set numerous records for individual accolades won throughout his professional footballing career, including eight Ballons d'Or, six European Golden Shoes, and eight times being named the world's best player by FIFA.[note 2] In 2025, he was named the All Time Men's World Best Player by the IFFHS.

Messi is the most decorated player in the history of professional football, having won 46 team trophies.[note 3] His records include most goals in a calendar year (91), most goals for a single club (672 for Barcelona), most goals in La Liga (474), most assists in international football (61), most goal contributions in the FIFA World Cup (21), and most goal contributions in the Copa América (32). Messi has scored over 910 senior career goals and provided over 410 assists for club and country, resulting in over 1,320 goal contributions—the highest total in the sport's history.[25]
"""
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
