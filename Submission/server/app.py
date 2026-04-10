from openenv.core.env_server import create_fastapi_app
from server.environment import CustomerSupportEnvironment
from models import SupportAction, SupportObservation
from server.tasks import TASKS, get_grader_reward

app = create_fastapi_app(CustomerSupportEnvironment, action_cls=SupportAction, observation_cls=SupportObservation)

@app.get("/")
async def root():
    return {"status": "ok", "name": "Customer Support OpenEnv", "version": "0.1.0"}

@app.get("/tasks")
async def list_tasks():
    return [task.model_dump() for task in TASKS]

@app.get("/grader")
async def get_grader():
    return {"status": "success", "message": "Grader endpoint active"}

@app.get("/baseline")
async def get_baseline():
    return {
        "easy": 0.99,
        "medium": 0.8,
        "hard": 0.6,
        "average": 0.8
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

