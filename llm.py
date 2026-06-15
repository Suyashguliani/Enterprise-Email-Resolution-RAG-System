from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_api_key = os.environ.get("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
    )

client = OpenAI(
    api_key=_api_key,
    base_url="https://api.groq.com/openai/v1",
)

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def generate_answer(question, context):
    prompt = f"""
You are an enterprise support assistant that analyzes historical email conversations and issue resolution threads.

Your job is to:
1. Identify the issue mentioned in the user's question.
2. Search the provided email context for similar past incidents.
3. Provide the exact resolution steps mentioned in the emails whenever possible.
4. Mention the person and the team contacted previously for the issue if available.
5. Summarize the actions taken and the final resolution clearly.
6. If solution not clear, mention the person and team which needs to be contacted for this issue.
7. MENTION THE TEAM WHO RESOLVED THE ISSUE AND GIVE ITS DISTRIBUTION LIST EMAIL.
8. MENTION THE TIME OF THE EMAIL AND INFORM THERE MAY BE CHANGES.

Rules:
- Use ONLY the provided context.
- Do NOT make up solutions, teams, or troubleshooting steps.
- If the issue was not fully resolved in the emails, clearly state that.
- If insufficient information exists, say:
  "I don't know based on the provided documents."
- Keep the answer concise, structured, and support-oriented.
- Prefer actual resolutions from the emails over assumptions.

Context:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
