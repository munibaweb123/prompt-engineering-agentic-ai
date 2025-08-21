import asyncio
import os
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner

_: bool = load_dotenv(find_dotenv())

# ONLY FOR TRACING
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

agent:Agent=Agent(
    name="Reasoning Assistant",
    instructions="You are a reasoning assistant. For every math question, always think step by step and explain your logic before giving the final answer",
    model=llm_model
)

async def main():
    result = await Runner.run(agent,"Determine if 42 is even or odd. Think step by step and explain your reasoning.")
    print(result.final_output)

asyncio.run(main())
