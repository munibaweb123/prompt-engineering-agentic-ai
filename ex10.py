# ex10.py
import asyncio
import os
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI

# ── Env ────────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Use Gemini (OpenAI-compatible) if provided, else default to OpenAI model id
if GEMINI_API_KEY:
    external_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = OpenAIChatCompletionsModel("gemini-2.5-flash", openai_client=external_client)
else:
    model = OpenAIChatCompletionsModel("gpt-4o-mini")

# ── Tool Schemas ───────────────────────────────────────────────────────────────
class SalesQuery(BaseModel):
    year: int = Field(..., description="Four-digit year, e.g., 2025")
    month: int = Field(..., ge=1, le=12, description="Month as number 1-12")
    region: Optional[str] = Field(None, description="Optional region filter")
    product: Optional[str] = Field(None, description="Optional product filter")

# ── Mock Sales Data Source (replace with real DB/API as needed) ────────────────
MOCK_MARCH_2025: List[Dict] = [
    {"date": "2025-03-01", "sales": 12450, "region": "NA", "product": "Core"},
    {"date": "2025-03-08", "sales": 13980, "region": "EU", "product": "Core"},
    {"date": "2025-03-15", "sales": 15210, "region": "APAC", "product": "Pro"},
    {"date": "2025-03-22", "sales": 16175, "region": "NA", "product": "Pro"},
    {"date": "2025-03-29", "sales": 17040, "region": "EU", "product": "Core"},
]

# ── Tool: sales_data_tool ──────────────────────────────────────────────────────
@function_tool
def sales_data_tool(query: SalesQuery) -> Dict[str, List[str]]:
    """
    Retrieve sales figures for a given month/year.
    Returns a flat list of strings for quick display.
    """
    if query.year == 2025 and query.month == 3:
        rows = MOCK_MARCH_2025
    else:
        rows = []  # in a real tool, query your DB/API here

    # Optional filters
    if query.region:
        rows = [r for r in rows if r["region"].lower() == query.region.lower()]
    if query.product:
        rows = [r for r in rows if r["product"].lower() == query.product.lower()]

    if not rows:
        return {"items": ["No sales data found for the requested period/filters."]}

    # Return quick list lines like: "2025-03-01 — $12,450"
    items = [f'{r["date"]} — ${r["sales"]:,}'for r in rows]
    return {"items": items}

# ── Agent ─────────────────────────────────────────────────────────────────────
agent = Agent(
    name="SalesDataAgent",
    model=model,
    tools=[sales_data_tool],
    instructions=(
        "You are a precise data assistant. "
        "When asked for sales figures, call `sales_data_tool` with the correct year and month. "
        "Return results quickly as a simple list (one item per line). "
        "Do not add extra commentary."
    ),
)

# ── Run ────────────────────────────────────────────────────────────────────────
async def main():
    corrected_prompt = (
        "Use the sales data tool to retrieve sales figures for March 2025. "
        "Return results quickly in a list."
        # "Use tool get data about sales fast."
    )

    # Optional: you can also nudge the tool args via few-shot in the prompt,
    # but the agent should infer (year=2025, month=3) from the text.
    result = await Runner.run(agent, corrected_prompt)
    # Print exactly what the agent returns (ideally a list)
    print(result.final_output if hasattr(result, "final_output") else result)

if __name__ == "__main__":
    asyncio.run(main())
