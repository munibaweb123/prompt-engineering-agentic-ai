import asyncio
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
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
# Disable tracing for clean output
set_tracing_disabled(True)

# Define the model
# model = OpenAIChatCompletionsModel("gpt-4o-mini")

# Build the agent
profile_agent = Agent(
    model=llm_model,
    name="Profile Summarizer",
    instructions=(
        "You are an assistant that summarizes user profiles into JSON. "
        "Always return output in JSON format starting with: {\"summary\": ...}. "
        "Do not include extra explanations."
    ),
)

async def main():
    # Example profile data
    profile_data = """
    Name: Muniba Ahmed
    Age: 29
    Location: Karachi, Pakistan
    Skills: Next.js, Tailwind CSS, TypeScript, Sanity, Stripe, ShipEngine, OpenAI Agents SDK
    Interests: Teaching, Freelancing, AI Agents, Graphic Design
    """

    # Effective Prompt
    user_prompt = f"Summarize this user profile in JSON: {profile_data}. Start with: {{'summary':"

    # Run the agent
    
    response = await Runner.run(profile_agent,user_prompt)
    print(response.final_output)

if __name__ == "__main__":
    asyncio.run(main())
