---
title: Customer Support OpenEnv
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Customer Support Ticket Resolution OpenEnv

## Description & Motivation
This environment simulates a real-world customer support desk. An AI agent acts as a support representative tasked with resolving incoming tickets using a knowledge base. This is a critical task for businesses looking to automate support workflows while maintaining high accuracy and customer satisfaction.

## Action Space
- `categorize`: Assign a category (Auth, Billing, Search) to a ticket.
- `search_kb`: Query the internal knowledge base for solutions.
- `resolve`: Provide the final resolution text to the customer.

## Observation Space
- `ticket_id`: Unique identifier for the ticket.
- `description`: The customer's message.
- `current_category`: The category currently assigned to the ticket.
- `kb_results`: Results from the last knowledge base search.
- `status`: Current status (open/resolved).
- `message`: Feedback from the environment.

## Tasks
1. **Easy: Ticket Categorization**
   - Goal: Correctly identify the category of a login or billing issue.
   - Reward: 1.0 for correct category.
2. **Medium: KB Retrieval**
   - Goal: Search the KB for a crash issue and provide the resolution.
   - Reward: 1.0 for providing the KB-derived answer.
3. **Hard: Complex Hardware Troubleshooting**
   - Goal: Identify a hardware fault from a specific symptom and provide the reset workaround.
   - Reward: 1.0 for the specific multi-step reset procedure.

## Setup & Usage
1. **Install deps:**
   ```bash
   cd Submission
   pip install openenv-core fastapi uvicorn pydantic
   ```

2. **Validate:**
   ```bash
   python validate.py
   ```

3. **Run locally:**
   ```bash
   uvicorn server.app:app --reload --port 8000
   ```

4. **Test baseline:**
   ```bash
   python baseline.py
   ```

5. **Docker build/test:**
   ```bash
   docker build -t cs-env .
   docker run -p 8000:8000 cs-env
   curl http://localhost:8000/baseline
   curl http://localhost:8000/tasks
   ```

6. **HF Spaces deploy:**
   - Push to HF repo tagged 'openenv'.
   - Dockerfile auto-builds.

## Endpoints
- `/tasks`: List tasks
- `/baseline`: Repro baseline scores
- `/grader`: Current episode score

## Baseline Scores (Reproducible)
- Easy: 0.99
- Medium: 0.99
- Hard: 0.99
- Average: 0.99
