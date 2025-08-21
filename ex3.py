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
agent = Agent(
    name="AI Agent",
    instructions=(
        "You are an AI that always outputs structured data. "
        "When asked for project ideas, always return them in a Markdown table with two columns: Name and Description. "
        "Always follow the format: | Name | Description |"
    ),
    model=llm_model
)
prompt = (
    "Generate three project ideas for an AI app. "
    "Format the output as a table with columns 'Name' and 'Description.' "
    "Example: | Name | Description | | AI Chat | A chatbot for customer support |"
)
result=Runner.run_sync(agent,prompt)
print(result.final_output)