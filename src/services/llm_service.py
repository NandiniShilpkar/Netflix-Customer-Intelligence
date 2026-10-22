import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


def generate_business_answer(question, analytics_context):
    prompt = f"""You are a business analytics assistant.

Question:
{question}

Analytics:
{analytics_context}

Answer briefly using only these analytics.
Do not invent numbers or customer IDs.
Do not change the customer list.
"""

    response = requests.post(
        OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 80,
            },
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    result = response.json()
    answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty response.")

    return answer
