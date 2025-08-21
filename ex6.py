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

agent=Agent(
    name="Article Summary Agent",
    instructions=(
        "You are a professional summarizer. "
        "Your job is to condense long text into short, accurate summaries. "
        "Never exceed 100 words. Focus only on the most important points."
    ),
    model=llm_model
)

article_text = (
    "The Future of Renewable Energy\n\n"
    "Renewable energy has rapidly transformed from a niche concept into one of the most critical "
    "drivers of global sustainability. Over the past two decades, the world has witnessed significant "
    "progress in clean energy technologies such as solar, wind, hydropower, and biomass. These sources "
    "are not only reshaping the global energy landscape but also playing a vital role in reducing "
    "greenhouse gas emissions. With climate change becoming an urgent global concern, governments, "
    "corporations, and communities are now investing more than ever in renewable energy solutions.\n\n"
    "Solar energy, in particular, has experienced exponential growth due to declining costs of "
    "photovoltaic panels and improved efficiency. Countries like China, India, and the United States are "
    "leading the solar revolution, installing massive solar farms and integrating rooftop solar systems "
    "into residential and commercial buildings. Similarly, wind energy has emerged as a powerful contender, "
    "with offshore wind farms gaining traction in Europe and Asia. Technological advancements in turbine "
    "design have increased energy output while reducing costs, making wind power a reliable and competitive option.\n\n"
    "Hydropower remains the largest source of renewable electricity worldwide. Although it faces challenges "
    "such as environmental impacts and geographical limitations, small-scale hydro projects and innovations "
    "in water turbine design continue to make it a valuable contributor to the renewable mix. Biomass energy, "
    "derived from organic materials such as agricultural waste and forestry products, also provides a "
    "sustainable alternative, particularly in rural regions where access to traditional energy infrastructure is limited.\n\n"
    "A key factor in the expansion of renewable energy has been supportive government policies and "
    "international collaboration. Agreements such as the Paris Climate Accord have pushed nations to "
    "commit to reducing carbon emissions, thereby accelerating investments in green technologies. Many "
    "countries now offer tax incentives, subsidies, and funding programs to encourage renewable energy "
    "adoption. Furthermore, private companies are increasingly pledging to transition to 100% renewable "
    "energy, demonstrating the growing importance of corporate sustainability.\n\n"
    "Despite these advances, the renewable energy sector faces challenges. One of the biggest hurdles is "
    "energy storage. Since solar and wind power are intermittent, storing excess energy in efficient "
    "batteries or other storage systems is critical to ensure consistent supply. Advances in lithium-ion "
    "and solid-state batteries are promising, but scaling these solutions remains expensive. Additionally, "
    "integrating renewable energy into existing power grids requires infrastructure upgrades and smart grid "
    "technologies to manage variable energy inputs effectively.\n\n"
    "Looking ahead, the future of renewable energy appears bright. Analysts predict that by 2050, the "
    "majority of the world’s electricity will come from renewable sources, drastically cutting dependency "
    "on fossil fuels. Innovations in hydrogen fuel, carbon capture, and next-generation solar cells will "
    "further accelerate this transition. Moreover, as awareness of climate change grows, public support for "
    "renewable energy initiatives is expected to rise, creating a powerful push toward a greener and more "
    "sustainable future.\n\n"
    "In conclusion, renewable energy is no longer an option but a necessity for the planet’s survival. "
    "With technological innovation, supportive policies, and global cooperation, the vision of a clean "
    "energy future is not only achievable but inevitable. The choices made today will shape the energy "
    "landscape for generations to come."
)


# Exercise 6 Prompt
prompt = f"Summarize this 500-word article in 100 words: {article_text}"

result = Runner.run_sync(agent,prompt)
print(result.final_output)