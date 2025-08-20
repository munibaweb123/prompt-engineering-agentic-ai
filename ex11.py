import asyncio
import os
from typing import List, Dict
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel

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

# ---- Tool for analyzing trends ----
class TrendResult(BaseModel):
    trends: List[Dict[str, str]]  # each trend has a 'trend' and 'impact'


@function_tool
def stats_tool(dataset: List[int]) -> TrendResult:
    """
    Analyze a dataset and return the top 3 trends in a table format.
    Keeps context small and structured.
    """
    if not dataset:
        return TrendResult(trends=[])

    avg = sum(dataset) / len(dataset)
    max_val = max(dataset)
    min_val = min(dataset)

    results = [
        {"trend": "Average Value", "impact": f"{avg:.2f}"},
        {"trend": "Maximum Value", "impact": str(max_val)},
        {"trend": "Minimum Value", "impact": str(min_val)},
    ]
    return TrendResult(trends=results)


# ---- Agent ----
agent = Agent(
    name="Trend Analysis Agent",
    instructions="you are a helpful trend analysis assistent, you help user with analysis the trends.",
    model=llm_model,
    tools=[stats_tool],
)


async def main():
    # Example dataset
    dataset = [12, 15, 20, 22, 18, 25, 30, 28]

    # Prompt 1 (vague)
    prompt1="Analyze trends in this dataset: " + str(dataset)
    resp1 = await Runner.run(agent,prompt1)

    # Prompt 2 (specific, optimized)
    prompt2=f"Analyze trends in this dataset using the stats tool: {str(dataset)} Limit to top 3 trends in a table, keeping context under 500 tokens."
    resp2 = await Runner.run(agent,
        prompt2
    )

    print("\n--- Prompt 1 (Vague) ---")
    print(resp1.final_output)
    print("\n--- Prompt 2 (Optimized) ---")
    print(resp2.final_output)


if __name__ == "__main__":
    asyncio.run(main())
