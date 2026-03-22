"""Generate answers using LLM via OpenRouter."""

import time

from openai import OpenAI

from benchmarks.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


def get_client() -> OpenAI:
    """Create OpenRouter client (OpenAI-compatible)."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set. Add it to .env file.")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


ANSWER_PROMPT = """You are a helpful assistant with access to a user's conversation history stored as memories.
Use the retrieved memories below to answer the user's question accurately.

## Retrieved Memories
{context}

## Question
{question}

## Question Date
This question was asked on: {question_date}

## Instructions
- Answer based ONLY on the information in the retrieved memories
- If the memories don't contain enough information to answer, say "I don't have enough information to answer this question"
- Be concise and direct
- For temporal questions, pay attention to dates
- If information was updated/corrected in later conversations, use the most recent version"""


def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    question_date: str = "",
    max_retries: int = 3,
) -> str:
    """Generate an answer using the specified model via OpenRouter."""
    prompt = ANSWER_PROMPT.format(
        context=context,
        question=question,
        question_date=question_date,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                return f"ERROR: {e}"
