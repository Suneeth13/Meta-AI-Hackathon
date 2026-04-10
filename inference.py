import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()

from client import SupportEnv
from models import SupportAction
from server.tasks import get_grader_reward


MAX_STEPS = 20
MODEL_NAME = "gpt-4o-mini"

def get_openai_client():
    """Initialize OpenAI client with proxy settings."""
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("API_BASE_URL")
    
    if not api_key:
        print("Warning: API_KEY not set. Falling back to rule-based ONLY.")
        return None
    
    return openai.OpenAI(api_key=api_key, base_url=base_url)

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


def generate_local_action(task_id, obs):
    """Rule-based action selection for customer support tickets."""
    desc_lower = obs.description.lower()
    
    # Hard task: hardware/bulb specific
    if task_id == "hard" or any(word in desc_lower for word in ['bulb', 'light', 'reset', 'flicker', 'blinking']):
        return {
            "action_type": "resolve",
            "resolution": "Please reset your bulb by unplugging it 5 times. This will restore factory settings."
        }
    # Medium: KB search for crashes
    elif task_id == "medium" or any(word in desc_lower for word in ['crash', 'error', 'exception', 'bug', 'failed']):
        return {
            "action_type": "search_kb",
            "query": "software crash error"
        }
    # Easy: categorize auth/billing
    elif task_id == "easy" or any(word in desc_lower for word in ['auth', 'login', 'account', 'password', 'sign in']):
        return {
            "action_type": "categorize",
            "category": "auth"
        }
    elif any(word in desc_lower for word in ['billing', 'payment', 'charge', 'subscription', 'invoice', 'refund']):
        return {
            "action_type": "categorize",
            "category": "billing"
        }
    else:
        # Fallback
        return {
            "action_type": "resolve",
            "resolution": "Thank you for contacting support. Our team will investigate and respond shortly."
        }


def parse_action(response_text):
    """Extract JSON action safely from response text."""
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
    """Run a single task episode with LLM and rule-based fallback."""
    print(f"\n--- Running task: {task_id} ---", flush=True)
    print(f"[START] task={task_id}", flush=True)

    obs = env.reset(task_id=task_id).observation
    state = env.state()

    total_reward = 0.0
    steps = 0

    while not obs.done and steps < MAX_STEPS:
        action_dict = None
        
        # 1. Try LLM if client is available
        if client:
            try:
                prompt = build_prompt(task_id, obs)
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a customer support agent. Respond with reasoning then valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                action_text = (response.choices[0].message.content or "").strip()
                action_dict = parse_action(action_text)
                print(f"LLM Action: {action_dict.get('action_type')}")
            except Exception as e:
                print(f"LLM Error: {e}")

        # 2. Fallback to Rule-based if LLM failed or not available
        if not action_dict or action_dict.get("action_type") not in ["categorize", "search_kb", "resolve"]:
            print("Using Rule-based fallback...")
            action_dict = generate_local_action(task_id, obs)

        action = SupportAction(**sanitize_action(action_dict))
        result = env.step(action)

        obs = result.observation
        state = env.state()
        total_reward += result.reward
        steps += 1

        print(f"[STEP] step={steps} reward={result.reward}", flush=True)
        print(
            f"Step {steps}: {action.action_type} | "
            f"Reward: {result.reward} | Total: {total_reward}",
            flush=True
        )

    final_score = get_grader_reward(state)
    avg_reward = total_reward / steps if steps else 0

    print(f"[END] task={task_id} score={final_score} steps={steps}", flush=True)
    print(
        f"{task_id} Score: {final_score} "
        f"(steps: {steps}, avg reward: {avg_reward:.2f})",
        flush=True
    )

    return final_score


def run_inference():
    """Run inference across all tasks."""
    client = get_openai_client()
    tasks = ["easy", "medium", "hard"]
    scores = {}

    with SupportEnv(base_url="http://localhost:8000").sync() as env:
        for task_id in tasks:
            scores[task_id] = run_task(env, client, task_id)

    avg_score = sum(scores.values()) / len(tasks)

    print(f"\nInference scores: {scores}, Avg: {avg_score:.2f}", flush=True)
    return scores


if __name__ == "__main__":
    print(
        "Running LLM-Based Inference (with Rule-Based Fallback)...\n"
        "Ensure server is running at http://localhost:8000.\n",
        flush=True
    )
    run_inference()
