import os
import time
from typing import Any, Callable

PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start
    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, latency, usage


def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start
    text = response.text
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return text, latency, usage


def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start
    text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage


def compare_models(prompt: str = "") -> dict:
    gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    gpt4o_mini_text, gpt4o_mini_lat, gpt4o_mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    gemini_text, gemini_lat, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)

    gpt4o_cost = (
        gpt4o_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"]
        + gpt4o_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000

    gpt4o_mini_cost = (
        gpt4o_mini_usage["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"]
        + gpt4o_mini_usage["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000

    gemini_cost = (
        gemini_usage["input_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"]
        + gemini_usage["output_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": gpt4o_cost,
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": gpt4o_mini_text,
            "latency": gpt4o_mini_lat,
            "cost": gpt4o_mini_cost,
            "input_tokens": gpt4o_mini_usage["input_tokens"],
            "output_tokens": gpt4o_mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": gemini_cost,
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


def streaming_chatbot() -> None:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        history.append({"role": "user", "parts": [{"text": user_input}]})
        active_history = history[-6:]
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=active_history,
        )
        full_response = ""
        for chunk in response_stream:
            text = chunk.text or ""
            print(text, end="", flush=True)
            full_response += text
        print()
        history.append({"role": "model", "parts": [{"text": full_response}]})


def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt >= max_retries:
                raise e
            time.sleep(base_delay * (2 ** attempt))


def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        try:
            res = compare_models(prompt)
        except TypeError:
            res = compare_models()
        res["prompt"] = prompt
        results.append(res)
    return results


def format_comparison_table(results: list[dict]) -> str:
    headers = ["Prompt", "Model", "Response (truncated)", "Latency", "Tokens (In/Out)", "Cost (USD)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    model_labels = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }
    for res in results:
        prompt = res.get("prompt", "")
        for key, model_label in model_labels.items():
            if key in res:
                info = res[key]
                resp = info.get("response", "")
                if len(resp) > 50:
                    resp = resp[:50]
                latency = f"{info.get('latency', 0.0):.2f}s"
                tokens = f"{info.get('input_tokens', 0)}/{info.get('output_tokens', 0)}"
                cost = f"${info.get('cost', 0.0):.6f}"
                lines.append(f"| {prompt} | {model_label} | {resp} | {latency} | {tokens} | {cost} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
