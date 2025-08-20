import asyncio
import os
from typing import Any, Dict
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

# Disable tracing (optional)
#set_tracing_disabled(True)
_ = load_dotenv(find_dotenv())

# API Keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

# 1. LLM Service
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. LLM Model
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

# -----------------------
# Define Tool Schemas
# -----------------------

class AnalyticsInput(BaseModel):
    target_audience: str
    competitors: list[str]

class BudgetInput(BaseModel):
    advertising: float
    influencers: float
    content_creation: float


# Fake analytics tool
@function_tool
def analytics_tool(data: AnalyticsInput):
    """Analyze target audience and competitors."""
    return {
        "audience_insights": f"Targeting {data.target_audience}",
        "competitor_summary": f"Main competitors: {', '.join(data.competitors)}"
    }

# Fake budget calculator tool
@function_tool
def budget_calculator_tool(budget: BudgetInput):
    """Calculate total budget for marketing."""
    total = budget.advertising + budget.influencers + budget.content_creation
    return {"total_cost": total}

# -----------------------
# Build the agent
# -----------------------
marketing_agent = Agent(
    model=llm_model,
    tools=[analytics_tool, budget_calculator_tool],
    name="Marketing Planner",
    instructions=(
        "You are a marketing strategist. Use the analytics_tool and budget_calculator_tool "
        "to build a detailed 3-month marketing campaign plan. "
        "Always include: strategy, timeline, and costs. "
        "Explain your reasoning step by step before giving the final structured plan."
    ),
)

# -----------------------
# Main Runner
# -----------------------
async def main():
    user_prompt = (
        "Develop a detailed marketing campaign plan using the analytics tool "
        "and budget calculator tool. Include strategy, timeline, and costs for a 3-month period. "
        "Explain your reasoning step by step."
    )

    response = await Runner.run(marketing_agent, user_prompt)
    print("\n📌 Final Marketing Plan:\n")
    print(response.final_output)

if __name__ == "__main__":
    asyncio.run(main())
