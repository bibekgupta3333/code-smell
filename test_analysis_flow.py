import requests
import time
import json
import os

BASE_URL = "http://localhost:8000/api/v1"

def test_flow():
    code = """def very_long_method(a, b, c, d, e, f, g, h):
    x = a + b
    y = c + d
    z = e + f
    w = g + h
    result = x + y + z + w
    return result
"""
    
    payload = {
        "code": code,
        "file_name": "code_snippet",
        "ground_truth_mode": True
    }
    
    print("Submitting code for analysis...")
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    if response.status_code not in [200, 202]:
        print(f"Error submitting analysis ({response.status_code}): {response.text}")
        return
    
    data = response.json()
    task_id = data.get("analysis_id") or data.get("task_id")
    print(f"Task ID: {task_id}")
    
    if not task_id:
        print("Failed to get task_id from response")
        return
    
    while True:
        status_response = requests.get(f"{BASE_URL}/progress/{task_id}")
        if status_response.status_code != 200:
            print(f"Error checking progress: {status_response.text}")
            time.sleep(2)
            continue
            
        status_data = status_response.json()
        status = status_data.get("status")
        print(f"Status: {status}")
        
        if status == "completed":
            break
        elif status == "failed":
            print(f"Analysis failed: {status_data.get('error')}")
            return
        
        time.sleep(2)
    
    results_response = requests.get(f"{BASE_URL}/results/{task_id}")
    results = results_response.json()
    
    # Print results summary
    print("\n--- Project Results ---")
    findings = results.get("findings", [])
    print(f"Number of findings detected: {len(findings)}")
    
    evaluation = results.get("evaluation", {})
    metrics = evaluation.get("metrics", {})
    
    print(f"F1 score: {metrics.get('f1_score', 'N/A')}")
    print(f"Evaluation mode: {evaluation.get('evaluation_mode', 'N/A')}")
    print(f"Metrics breakdown: {json.dumps(metrics, indent=2)}")

    print("\n--- Recent Log Entries ---")
    log_dir = "results/logs"
    if os.path.exists(log_dir):
        # List files and find the most recent log file
        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".log") or f.endswith(".json")]
        if log_files:
            latest_log = max(log_files, key=os.path.getmtime)
            print(f"Latest log: {latest_log}")
            with open(latest_log, 'r') as f:
                content = f.read()
                # Look for F1 or ground truth related lines
                relevant_lines = [line for line in content.split('\n') if "F1" in line or "ground truth" in line.lower() or "evaluation" in line.lower()]
                for line in relevant_lines[-15:]:
                    print(line)
        else:
            print("No logs found in results/logs")
    else:
        print(f"Log directory {log_dir} not found")

if __name__ == "__main__":
    test_flow()
