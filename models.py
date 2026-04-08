from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from openenv.core.env_server import Action, Observation, State

class SupportAction(Action):
    action_type: str  # "categorize", "search_kb", "resolve"
    
    query: Optional[str] = None
    category: Optional[str] = None
    resolution: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class SupportObservation(Observation):
    done: bool # type: ignore
    reward: Optional[float] = None
    ticket_id: str
    description: str
    current_category: Optional[str] = None
    kb_results: List[str] = Field(default_factory=list)
    status: str = "open"
    message: str = ""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class SupportState(State):
    episode_id: Optional[str] = None
    step_count: int = 0
    ticket_data: Dict[str, Any] = Field(default_factory=dict)
    current_category: Optional[str] = None
    kb_history: List[str] = Field(default_factory=list)
    resolved: bool = False
    task_id: str = "easy"
    last_action_type: Optional[str] = None
    last_query: Optional[str] = None
    last_category: Optional[str] = None
    last_resolution: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
