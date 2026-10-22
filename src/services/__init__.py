import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


def get_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to the .env file."
        )

    return OpenAI(api_key=api_key)


def get_llm_model():
    return os.getenv("OPENAI_MODEL", "gpt-5-mini")


SYSTEM_PROMPT = """
You are the AI Analytics Copilot for a Netflix Customer Intelligence
platform.

Your job is to explain analytics results to a business stakeholder.

Important rules:

1. Use only the data and analytical context provided to you.
2. Do not invent customer metrics, revenue, churn rates, causes,
   or recommendations that are unsupported by the supplied context.
3. Clearly distinguish:
   - observed data
   - model predictions
   - business recommendations
4. Churn risk is a model prediction, not a guaranteed outcome.
5. Revenue at risk represents revenue associated with customers
   classified as high risk. It is not guaranteed revenue loss.
6. Do not claim causation when the data only shows correlation
   or an observed pattern.
7. Keep the response concise and business-oriented.
8. Prefer headings and short bullet points.
9. When useful, include:
   - Key finding
   - Evidence
   - Recommended action
10. If the context is insufficient to answer the question,
    explicitly say that the available analytics do not provide
    enough evidence.

You are an analytics assistant, not a general-purpose chatbot.
"""


def generate_business_answer(question, analytics_context):
    client = get_llm_client()
    model = get_llm_model()

    user_prompt = f"""
Business question:

{question}

Analytics context:

{analytics_context}

Using only the analytics context above, answer the business question.

Structure the answer as:

### Key finding

Give the most important answer first.

### Evidence

Use the relevant numbers and observations from the analytics context.

### Recommended action

Give practical business actions when supported by the data.

Keep the response concise and avoid unnecessary technical details.
"""

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )

    return response.output_text.strip()