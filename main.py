from dotenv import load_dotenv

load_dotenv()

from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import SecretStr

from config import settings


def get_llm(provider: Literal["openai", "ollama"]):
    if provider == "openai":
        return ChatOpenAI(
            model="gpt-6-luna", api_key=SecretStr(settings.OPENAI_API_KEY)
        )
    elif provider == "ollama":
        return ChatOllama(model="qwen3.5:2b", num_ctx=8192, num_predict=256)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


llm = get_llm("openai")
tools = [TavilySearch(max_results=3)]
agent = create_agent(llm, tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    )
    print(result)


if __name__ == "__main__":
    main()
