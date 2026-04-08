import subprocess
import time
import os
import threading
from inference import run_inference

def run_server():
    """Run uvicorn server in subprocess."""
    cmd = [
        'uvicorn', 'server.app:app',
        '--host', '0.0.0.0',
        '--port', '8000',
        '--reload'
    ]
    proc = subprocess.Popen(cmd, cwd='Submission', env=os.environ.copy())
    return proc

def test_inference():
    """Test inference.py locally with server running."""
    print("Starting server...")
    server_proc = run_server()
    time.sleep(5)  # Wait for server startup
    
    try:
        print("Running inference...")
        os.environ['API_BASE_URL'] = 'http://localhost:8000'  # Override if needed
        scores = run_inference()
        print("Inference complete:", scores)
    except Exception as e:
        print(f"Inference error: {e}")
    finally:
        server_proc.terminate()
        server_proc.wait()

if __name__ == '__main__':
    test_inference()
