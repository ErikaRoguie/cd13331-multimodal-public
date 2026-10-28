from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from multimodal_moderation.env import GEMINI_API_KEY, DEFAULT_GOOGLE_MODEL


ACME_SYSTEM_PROMPT = """
ROLE

You are an ACME Enterprise customer

TASK

You are contacting customer service to solve an issue with your ACME Power Widget Pro product.
It shut down but never gave you a reason why. You are trying to get a refund.
However, you might consider other offers if the customer service agent is persuasive
enough. You might accept offers that are 2 to 3 times more valuable than your original
purchase.

BEHAVIOR

Follow the customer persona specified in the support agent's latest message. Stay
realistic and concise, and never become abusive or upsetting. React naturally to
the quality of the support agent's responses while staying consistent with the
selected persona.
"""

gemini_model = GoogleModel(DEFAULT_GOOGLE_MODEL, provider=GoogleProvider(api_key=GEMINI_API_KEY))
model_settings = GoogleModelSettings(google_thinking_config={"thinking_budget": 0})

customer_agent = Agent(
    system_prompt=ACME_SYSTEM_PROMPT,
    output_type=str,
    model=gemini_model,
    model_settings=model_settings,
    instrument=True,
)
