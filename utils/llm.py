from openai import OpenAI

client = OpenAI(
    api_key="YOUR_GROQ_API_KEY",
    base_url="https://api.groq.com/openai/v1"
)

def generate_answer(question, context):
    prompt = f"""
You are a helpful assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer in a clear and concise way.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content