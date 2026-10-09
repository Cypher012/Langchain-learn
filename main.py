from typing import Literal

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from config import settings

SUMMARY_TEMPLATE = """
You are a patient computer science tutor.

Explain {topic} to a complete beginner.

Your explanation should:
- Start with a one-sentence definition in plain language
- Describe what problem it solves and why someone would use it
- Give one simple, real-world example
- End with a short summary of the key idea

Avoid jargon where possible. If you must use a technical term, define it the first time you use it.
"""

topic = "Agentic AI"

LLMProvider = Literal["openai", "ollama"]


def get_llm(provider: LLMProvider):
    match provider:
        case "openai":
            return ChatOpenAI(
                temperature=0.0,
                model="gpt-5-nano",
                api_key=SecretStr(settings.OPENAI_API_KEY),
            )
        case "ollama":
            return ChatOllama(
                model="qwen3.5:2b",
                temperature=0.0,
                reasoning=False,
            )
        case _:
            raise ValueError(f"Unknown LLM: {provider}")


def main():
    summary_prompt_template = PromptTemplate(
        input_variables=["topic"],
        template=SUMMARY_TEMPLATE,
    )

    llm = get_llm("ollama")

    chain = summary_prompt_template | llm

    result = chain.invoke({"topic": topic})
    print(result.content)


if __name__ == "__main__":
    main()
