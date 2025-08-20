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
    model="gemini-2.5-flash",
    openai_client=external_client
)
agent = Agent(
    name="cost agent",
    instructions="You are a professional AI assistant. "
        "Always answer concisely in a professional tone. "
        "Never exceed 50 words.",
    model=llm_model
)
prompt="Respond to this query in a concise, professional tone: 'What are the ethical concerns of AI?' Limit to 50 words."

result=Runner.run_sync(agent,prompt)
print(result.final_output)