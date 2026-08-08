import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

messages = [
    {
        "role": "user",
        "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    }
]

response = client.chat.completions.create(
    model="openrouter/free",
    messages=messages,
)
prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens

print(response.choices[0].message.content)

print(f"Prompt tokens: {prompt_tokens}")
print(f"Response tokens: {completion_tokens}")
