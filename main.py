from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """ Scheme for the source used by the agent"""
    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """ Scheme for the response of the agent"""
    answer:str = Field(description="The agent answer to the query")
    sources:list[Source] = Field(default_factory=list, description="The sources used by the agent to answer the query")

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain!")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 jobs posting in AI in Chicago area on linkedin and list their details?")})
    print(result)

if __name__ == "__main__":
    main()