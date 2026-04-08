import uuid
import random
from typing import Optional
from openenv.core.env_server import Environment
from models import SupportAction, SupportObservation, SupportState

TICKETS = {
    "easy": [
        {"id": "T-101", "desc": "I can't log in to my account. It says 'Invalid password'.", "category": "Auth"},
        {"id": "T-102", "desc": "How do I change my billing address?", "category": "Billing"},
    ],
    "medium": [
        {"id": "T-201", "desc": "The app crashes when I click 'Submit'.", "answer_key": "Check for updates or reinstall the app.", "kb_query": "app crash submit"},
    ],
    "hard": [
        {"id": "T-301", "desc": "My smart bulb is flashing red 3 times and won't connect.", "answer_key": "Reset the bulb by turning it off and on 5 times.", "kb_query": "flashing red 3 times"},
    ]
}

KB = {
    "app crash submit": "Known issue in v1.2. Check for updates or reinstall the app.",
    "flashing red 3 times": "Error 403: Hardware fault. Reset the bulb by turning it off and on 5 times.",
    "billing": "Go to Settings > Billing to update your address.",
    "auth": "Reset your password via the 'Forgot Password' link."
}

class CustomerSupportEnvironment(Environment):
    def __init__(self):
        self._state = SupportState()
        self.max_steps = 20


    def reset(self, task_id: str = "easy") -> SupportObservation:
        ticket = random.choice(TICKETS.get(task_id, TICKETS["easy"]))
        self._state = SupportState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            ticket_data=ticket,
            task_id=task_id
        )
        return SupportObservation(
            done=False,
            reward=None,
            ticket_id=ticket["id"],
            description=ticket["desc"],
            message=f"New ticket received. Task: {task_id}"
        )

    def step(self, action: SupportAction) -> SupportObservation:
        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        self._state.last_query = action.query
        self._state.last_category = action.category
        self._state.last_resolution = action.resolution

        if self._state.step_count >= self.max_steps:
            return SupportObservation(
                done=True,
                reward=-1.0,
                ticket_id=self._state.ticket_data["id"],
                description=self._state.ticket_data["desc"],
                current_category=self._state.current_category,
                status="timeout",
                message="Episode timeout - too many steps."
            )

        reward = 0.0
        message = ""
        done = False

        if action.action_type == "categorize":
            cat = action.category or ""
            self._state.current_category = cat
            if cat.lower() == self._state.ticket_data.get("category", "").lower():
                reward = 0.5
                message = "Correct category assigned."
            else:
                reward = -0.1
                message = "Incorrect category assigned."
        
        elif action.action_type == "search_kb":
            query = (action.query or "").lower()
            result = KB.get(query, "No relevant articles found.")
            self._state.kb_history.append(query)
            reward = 0.1 if result != "No relevant articles found." else -0.05
            message = f"KB Search Result: {result}"
            return SupportObservation(
                done=False,
                reward=reward,
                ticket_id=self._state.ticket_data["id"],
                description=self._state.ticket_data["desc"],
                current_category=self._state.current_category or "",
                kb_results=[result],
                status="open",
                message=message
            )

        elif action.action_type == "resolve":
            resolution = (action.resolution or "").lower()
            target = self._state.ticket_data.get("answer_key", self._state.ticket_data.get("category", "")).lower()
            
            if target and target in resolution:
                reward = 1.0
                message = "Ticket resolved successfully!"
                self._state.resolved = True
                done = True
            else:
                reward = -0.5
                message = "Incorrect resolution."
                done = self._state.step_count >= 10 # Max 10 steps

        return SupportObservation(
            done=done,
            reward=reward,
            ticket_id=self._state.ticket_data["id"],
            description=self._state.ticket_data["desc"],
            current_category=self._state.current_category,
            status="resolved" if self._state.resolved else "open",
            message=message
        )

    @property
    def state(self) -> SupportState:
        return self._state
