import asyncio
import os
import httpx
from typing import Dict, Any, Tuple
from pydantic import BaseModel
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool

# Load env vars
_ = load_dotenv(find_dotenv())

# (Optional) ONLY FOR TRACING IN SOME SETUPS
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# ---- LLM: Gemini via OpenAI-compatible endpoint ----
gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

# ---- ShipEngine ----
SHIPENGINE_API_KEY = os.getenv("SHIPENGINE_API_KEY")
if not SHIPENGINE_API_KEY:
    raise RuntimeError("Missing SHIPENGINE_API_KEY in environment.")

# ---------- Models ----------
class ShippingCostResponse(BaseModel):
    step1: str
    step2: str
    step3: str
    final_cost: float

# ---------- Helpers ----------
def resolve_location(city: str) -> Tuple[str, str]:
    """
    Resolve a simple city name to (country_code, postal_code) for demo purposes.
    Extend this mapping or accept structured input for production.
    """
    city_key = city.strip().lower()

    # Minimal demo mapping
    presets = {
        "new york": ("US", "10001"),
        "nyc": ("US", "10001"),
        "paris": ("FR", "75001"),
        # Add more as needed...
    }

    if city_key in presets:
        return presets[city_key]

    # Fallback: simple heuristics (very naive)
    # Expect "City, CC POSTAL" -> e.g., "Paris, FR 75001"
    # Use only if you want loose parsing:
    # parts = city.split(",")
    # if len(parts) == 2:
    #     cc_and_postal = parts[1].strip().split()
    #     if len(cc_and_postal) >= 2:
    #         return (cc_and_postal[0].upper(), cc_and_postal[-1])

    # If we cannot resolve, default to US 10001 to avoid 400 due to empty fields
    return ("US", "10001")

async def get_shipping_rate_estimate(
    package_weight_kg: float,
    origin_city: str,
    destination_city: str,
) -> Dict[str, Any]:
    """
    Calls ShipEngine /v1/rates/estimate with structured data.
    NOTE: We intentionally DO NOT send carrier_ids here to avoid invalid placeholder IDs.
    """
    url = "https://api.shipengine.com/v1/rates/estimate"
    headers = {
        "API-Key": SHIPENGINE_API_KEY,
        "Content-Type": "application/json",
    }

    from_country, from_postal = resolve_location(origin_city)
    to_country, to_postal = resolve_location(destination_city)

    payload: Dict[str, Any] = {
        #"carrier_ids": ["se-1646315","se-1646316","se-1646317","se-1646383","se-3004923"],  # Omit unless you have real carrier IDs connected
        "from_country_code": from_country,
        "from_postal_code": from_postal,
        "to_country_code": to_country,
        "to_postal_code": to_postal,
        "weight": {
            "value": float(package_weight_kg),
            "unit": "kilogram",
        },
        # Dimensions are optional but improve estimate quality
        "dimensions": {
            "unit": "centimeter",
            "length": 30.0,
            "width": 20.0,
            "height": 10.0,
        },
        "confirmation": "none",
        "address_residential_indicator": "no",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json=payload)
        # Raise for HTTP errors so we can surface the error details below
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Try to extract ShipEngine's errors payload
            try:
                err_json = resp.json()
            except Exception:
                err_json = {"raw": resp.text}
            raise RuntimeError(
                f"ShipEngine API error ({resp.status_code}): {err_json}"
            ) from e

        return resp.json()

# ---------- Tool ----------
@function_tool
async def calculate_shipping(
    package_weight: float,
    origin: str,
    destination: str
) -> ShippingCostResponse:
    """
    Tool: Calculate shipping using ShipEngine estimate endpoint.
    Returns step-by-step explanation + final cost.
    """
    # Step 1: Query API
    step1 = (
        f"Queried ShipEngine /v1/rates/estimate for {package_weight} kg "
        f"from '{origin}' to '{destination}', including weight and dimensions."
    )

    api_response = await get_shipping_rate_estimate(
        package_weight_kg=package_weight,
        origin_city=origin,
        destination_city=destination,
    )

    # Step 2: Process Data
    # Response format is an array of rate estimates or an object depending on API evolution.
    # Commonly you'll see a list under 'rate_response' for /v1/rates (NOT estimate).
    # For /v1/rates/estimate, expect a list of estimates (carrier/service/amount).
    # We'll handle both shapes defensively.

    cost = None
    currency = None

    if isinstance(api_response, dict) and "rate_response" in api_response:
        # Some responses might look like the /v1/rates shape
        rates = api_response.get("rate_response", {}).get("rates", [])
        if not rates:
            raise RuntimeError("No rates found in ShipEngine response.")
        first = rates[0]
        amt = first.get("shipping_amount", {})
        cost = float(amt.get("amount"))
        currency = amt.get("currency")
    elif isinstance(api_response, list):
        # Many examples show /v1/rates/estimate returning a list of estimates
        if not api_response:
            raise RuntimeError("No estimates returned by ShipEngine.")
        first = api_response[0]
        # Typical keys: amount, currency, service_code, carrier_id, carrier_friendly_name, etc.
        cost = float(first.get("amount"))
        currency = first.get("currency")
    else:
        raise RuntimeError(f"Unexpected ShipEngine response shape: {api_response}")

    step2 = f"Processed response: found cost={cost} {currency} (first available estimate)."

    # Step 3: Return structured result
    step3 = "Returned final shipping cost in structured format."

    return ShippingCostResponse(
        step1=step1,
        step2=step2,
        step3=step3,
        final_cost=cost if cost is not None else -1.0,
    )

# ---------- Agent ----------
agent = Agent(
    name="Shipping Agent",
    instructions=(
        "You are a shipping assistant. "
        "When the user asks for costs, call the tool and then present: "
        "1) query step, 2) processing step, 3) final cost. Be concise and clear."
    ),
    tools=[calculate_shipping],
    model=llm_model,
)

# ---------- Runner ----------
async def main():
    result = await Runner.run(
        agent,
        "Calculate shipping costs for a 5kg package from New York to Paris using the ShipEngine API. "
        "Show your steps: 1) query API, 2) process data, 3) return cost."
    )
    # Depending on your Agents SDK version, you might access either:
    # print(result) or print(result.final_output). We'll print both safely.
    try:
        print(result.final_output)
    except AttributeError:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
