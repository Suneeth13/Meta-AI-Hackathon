from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import SupportAction, SupportObservation, SupportState

class SupportEnv(EnvClient[SupportAction, SupportObservation, SupportState]):
    def _step_payload(self, action: SupportAction) -> dict:

        return {
            "action_type": action.action_type,
            "query": action.query,
            "category": action.category,
            "resolution": action.resolution
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", payload)
        obs = SupportObservation(
            done=payload["done"],
            reward=payload.get("reward"),
            ticket_id=obs_data["ticket_id"],
            description=obs_data["description"],
            current_category=obs_data.get("current_category"),
            kb_results=obs_data.get("kb_results", []),
            status=obs_data.get("status", "open"),
            message=obs_data.get("message", "")
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload["done"]
        )

    def _parse_state(self, payload: dict) -> SupportState:
        return SupportState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            ticket_data=payload.get("ticket_data", {}),
            current_category=payload.get("current_category"),
            kb_history=payload.get("kb_history", []),
            resolved=payload.get("resolved", False),
            task_id=payload.get("task_id", "easy"),
            last_action_type=payload.get("last_action_type"),
            last_query=payload.get("last_query"),
            last_category=payload.get("last_category"),
            last_resolution=payload.get("last_resolution")
        )

