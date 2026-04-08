from typing import List, Dict
from pydantic import BaseModel

class Task(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str

TASKS = [
    Task(
        id="easy",
        name="Ticket Categorization",
        difficulty="easy",
        description="Categorize the incoming ticket into one of the predefined categories (Auth, Billing, Search)."
    ),
    Task(
        id="medium",
        name="KB Retrieval",
        difficulty="medium",
        description="Search the knowledge base for the correct answer to a simple user query."
    ),
    Task(
        id="hard",
        name="Complex Hardware Troubleshooting",
        difficulty="hard",
        description="Identify a complex hardware issue and provide the specific reset workaround from the knowledge base."
    )
]

from models import SupportState

def get_grader_reward(state: SupportState) -> float:
    """
    Programmatic grader that returns a score between 0.0 and 1.0.
    """
    ticket_data = getattr(state, 'ticket_data', {})
    last_category = getattr(state, 'last_category', None)
    last_resolution = getattr(state, 'last_resolution', None)
    resolved = getattr(state, 'resolved', False)
    
    if state.task_id == "easy":
        # Check categorization
        target = ticket_data.get("category", "").lower()
        if last_category and last_category.lower() == target:
            return 1.0
        return 0.0
    
    elif state.task_id == "medium":
        # Check if the last resolution contains the KB answer
        target_answer = ticket_data.get("answer_key", "").lower()
        res = (last_resolution or "").lower()
        if resolved and target_answer in res:
            return 1.0
        return 0.0
    
    elif state.task_id == "hard":
        # Check for the specific reset instructions in the last resolution
        target_answer = "reset the bulb by turning it off and on 5 times".lower()
        res = (last_resolution or "").lower()
        if resolved and target_answer in res:
            return 1.0
        return 0.0
    
    return 0.0
