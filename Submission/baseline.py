import os
import json
import openai

from client import SupportEnv
from models import SupportAction
from server.tasks import get_grader_reward


MAX_STEPS = 20
MODEL_NAME = "gpt-4o-mini"


def get_openai_client():
    """Initialize OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return openai.OpenAI(api_key=api_key)


def build_prompt(task_id, obs):
    """Construct prompt for the LLM."""
    return f"""Task: {task_id}
Ticket ID: {obs.ticket_id}
Description: {obs.description}
Current Category: {getattr(obs, 'current_category', 'None')}
KB Results: {getattr(obs, 'kb_results', [])}
Status: {getattr(obs, 'status', 'open')}
Message: {obs.message}

You are a support agent. Choose the next action.

Output ONLY JSON:
{{"action_type": "categorize|search_kb|resolve",
  "query": "optional",
  "category": "optional",
  "resolution": "optional"}}

Reason briefly, then output JSON.
"""


def parse_action(response_text):
    """Extract JSON action safely from model response."""
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end] if start != -1 else "{}"
        return json.loads(json_str)
    except Exception:
        return {
            "action_type": "resolve",
            "resolution": "Please try restarting the service."
        }


def sanitize_action(action_dict):
    """Ensure only valid fields are passed to SupportAction."""
    valid_fields = {"action_type", "query", "category", "resolution", "metadata"}
    filtered = {k: v for k, v in action_dict.items() if k in valid_fields}

    if "metadata" not in filtered or not isinstance(filtered.get("metadata"), dict):
        filtered["metadata"] = {}

    return filtered


def run_task(env, client, task_id):
    """Run a single task episode."""
    print(f"\n--- Running task: {task_id} ---")

    obs = env.reset().observation
    state = env.state()

    total_reward = 0.0
    steps = 0

    while not obs.done and steps < MAX_STEPS:
        prompt = build_prompt(task_id, obs)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a customer support agent. "
                            "Respond with reasoning followed by valid JSON action."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                seed=42,
            )

            action_text = (response.choices[0].message.content or "").strip()
            action_dict = parse_action(action_text)

        except Exception as e:
            print(f"API error: {e} → using fallback action")
            action_dict = {
                "action_type": "resolve",
                "resolution": "Please try restarting the service."
            }

        action = SupportAction(**sanitize_action(action_dict))
        result = env.step(action)

        obs = result.observation
        state = env.state()
        total_reward += result.reward
        steps += 1

        print(
            f"Step {steps}: {action.action_type} | "
            f"Reward: {result.reward} | Total: {total_reward}"
        )

    final_score = get_grader_reward(state)
    avg_reward = total_reward / steps if steps else 0

    print(
        f"{task_id} Score: {final_score} "
        f"(steps: {steps}, avg reward: {avg_reward:.2f})"
    )

    return final_score


def run_baseline():
    """Run baseline across all tasks."""
    client = get_openai_client()
    tasks = ["easy", "medium", "hard"]
    scores = {}

    with SupportEnv(base_url="http://localhost:8000").sync() as env:
        for task_id in tasks:
            scores[task_id] = run_task(env, client, task_id)

    avg_score = sum(scores.values()) / len(tasks)

    print("\n=== OpenAI Baseline Results ===")
    print("Scores:", scores)
    print("Average:", round(avg_score, 2))

    return scores, avg_score


if __name__ == "__main__":
    print(
        "Running OpenAI Baseline...\n"
        "Ensure server is running at http://localhost:8000 "
        "and OPENAI_API_KEY is set."
    )
    run_baseline()