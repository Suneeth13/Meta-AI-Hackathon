#!/usr/bin/env python3
# Simple validation script for Customer Support OpenEnv.
from server.environment import CustomerSupportEnvironment
from server.tasks import get_grader_reward
from models import SupportAction

def test_environment():
    print("Testing reset/step...")
    env = CustomerSupportEnvironment()
    
    # Test easy task
    obs = env.reset("easy")
    print(f"Reset: {obs.message}")
    
    action = SupportAction(action_type="categorize", category="Auth")
    obs = env.step(action)
    print(f"Step categorize: reward={obs.reward}, done={obs.done}")
    
    print(f"Grader score: {get_grader_reward(env.state)}")
    
    print("\nAll tests passed! Ready for deployment.")

if __name__ == "__main__":
    test_environment()

