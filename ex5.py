import os
import sqlite3
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool

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

# -------------------------
# Step 1: Mock Database Tool
# -------------------------
@function_tool
def query_sales(q: str) -> dict:
    """
    Executes a SQL query on a mock SQLite database.
    Returns results as a list of rows (for demonstration).
    """

    # Connect to an in-memory SQLite database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create sample sales data
    cursor.execute("""
    CREATE TABLE sales (
        id INTEGER PRIMARY KEY,
        month TEXT,
        revenue INTEGER
    )
    """)
    sample_data = [
        ("Jan", 12000),
        ("Feb", 15000),
        ("Mar", 18000),
        ("Apr", 10000)
    ]
    cursor.executemany("INSERT INTO sales (month, revenue) VALUES (?, ?)", sample_data)

    # Run the provided query
    cursor.execute(q)
    rows = cursor.fetchall()

    conn.close()
    return {"results": rows}


# -------------------------
# Step 2: Define the Agent
# -------------------------
agent=Agent(
    name="Data Agent",
    instructions="You are a data analyst. Use the database query tool to analyze sales data "
        "and summarize findings in clear bullet points.",
    tools=[query_sales],
    model=llm_model
)
# -------------------------
# Step 3: Run Exercise 5 Prompt
# -------------------------
prompt = (
    "Act as a data analyst. Use the database query tool to analyze sales data "
    "and identify trends for Q1 2025. Return results in bullet points."
)

result = Runner.run_sync(agent,prompt)
print(result.final_output)