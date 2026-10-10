import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in .env"
    )

client = OpenAI(
    api_key=api_key
)


def generate_answer(
    question,
    context,
    customer_info=None
):

    customer_info = customer_info or {}

    prompt = f"""
You are a Customer Churn AI Assistant.

Your job is to analyze the customer's churn
prediction and provide a practical retention
recommendation.

Customer Information:
{customer_info}

Knowledge Base Context:
{context}

User Question:
{question}

Rules:
- Give a clear and simple answer.
- Use the provided knowledge as the main source.
- Consider the customer's churn probability.
- Give practical retention suggestions.
- Do not invent customer information.
- Do not guarantee that the customer will churn.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text