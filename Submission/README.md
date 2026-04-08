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
Object with:
- `action_type`: "categorize", "search_kb", "resolve"
- `query`: str (optional for search)
- `category`: str (optional for categorize)
- `resolution`: str (optional for resolve)

## Observation Space
Object with:
- `ticket_id`: str
- `description`: str
- `current_category`: str/null
- `kb_results`: list[str]
- `status`: "open"/"resolved"
- `message`: str
- `done`: bool
- `reward`: float

## Tasks
1. **Easy** (Categorization): Match Auth/Billing keywords. Grader reward 1.0 correct.
2. **Medium** (KB Retrieval): Search crash, resolve with KB answer. 1.0 match.
3. **Hard** (Hardware): Specific "bulb reset 5 times". 1.0 exact.

Baseline repro ~0.8 avg.

## Setup
```
cd Submission
pip install -r requirements.txt
uvicorn server.app:app --reload --port 8000
```

## Inference (Pre-submission)
```
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=$OPENAI_API_KEY
python inference.py  # Run with server up
```

## Docker/HF
docker build -t cs-env .
HF Spaces: Upload all, tag 'openenv'.

Endpoints: / /tasks /grader /baseline

Baseline: inference.py OpenAI, <20min, low resource.

