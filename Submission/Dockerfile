FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install openenv-core
RUN pip install --no-cache-dir openenv-core fastapi uvicorn pydantic pyyaml openai

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
