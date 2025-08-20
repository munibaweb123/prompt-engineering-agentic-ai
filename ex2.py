import os
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
import requests
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
    model="gemini-2.5-flash",
    openai_client=external_client
)

# Your OpenWeather API key (free tier works fine)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Define the weather tool
@function_tool
def get_weather(city: str, country: str = "UK") -> dict:
    """
    Fetches the current weather for a city using OpenWeather API.
    Returns temperature (°C) and weather condition.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()

    return {
        "temperature": response["main"]["temp"],
        "condition": response["weather"][0]["description"]
    }

# Define the model (GPT-4.1 or GPT-4o recommended)
#model = OpenAIChatCompletionsModel("gpt-4o-mini")

# Create the agent
agent = Agent(
    name="weather agent",
    model=llm_model,
    tools=[get_weather],  # Attach our weather tool
    instructions="You are a helpful agent. Always use tools when needed. Return concise answers."
)

# Run the agent with the Exercise 2 prompt

result = Runner.run_sync(agent,"Use the weather API tool to get the current weather in Karachi, Pakistan. Return the temperature and condition.")

print(result.final_output)
