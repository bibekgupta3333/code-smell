import requests
import time

API_URL = "http://localhost:8000/api/v1"

code = """
class LargeClass:
    def __init__(self):
        self.data = []

    def long_method(self):
        print("Starting a very long method...")
        a = 1
        b = 2
        c = a + b
        d = c * 2
        e = d - 1
        f = e / 1
        g = f + a
        h = g - b
        i = h * c
        j = i / d
        k = j + e
        l = k - f
        m = l * g
        n = m / h
        o = n + i
        p = o - j
        q = p * k
        r = q / l
        s = r + m
        t = s - n
        u = t * o
        v = u / p
        w = v + q
        x = w - r
        y = x * s
        z = y / t
        # Duplicate code block
        print("Result is", z)
        a2 = 1
        b2 = 2
        c2 = a2 + b2
        d2 = c2 * 2
        e2 = d2 - 1
        f2 = e2 / 1
        g2 = f2 + a2
        h2 = g2 - b2
        i2 = h2 * c2
        j2 = i2 / d2
        k2 = j2 + e2
        l2 = k2 - f2
        m2 = l2 * g2
        n2 = m2 / h2
        o2 = n2 + i2
        p2 = o2 - j2
        q2 = q2 * k2 if 'q2' in locals() else 0
        r2 = q2 / l2 if 'l2' in locals() and l2 !=0 else 0
        # Dead code
        if False:
            print("This will never run")
        return z

def another_duplicate():
    a = 1
    b = 2
    c = a + b
    d = c * 2
    e = d - 1
    f = e / 1
    g = f + a
    h = g - b
    i = h * c
    j = i / d
"""

def run_test():
    payload = {
        "file_name": "smelly_code.py",
        "code": code,
        "language": "python",
        "use_rag": True,
        "model_provider": "openai",
        "model_name": "gpt-4"
    }
    
    response = requests.post(f"{API_URL}/analyze", json=payload)
    
    if response.status_code not in [200, 201, 202]:
        print(f"Error submitting analysis: {response.status_code} - {response.text}")
        return

    job_id = response.json().get("job_id") or response.json().get("analysis_id")
    print(f"Submitted job {job_id}")

    # Poll for results
    for _ in range(30):
        res = requests.get(f"{API_URL}/results/{job_id}")
        data = res.json()
        status = data.get("status")
        print(f"Status: {status}")
        
        if status == "completed":
            findings = data.get("findings", [])
            print(f"Number of findings detected: {len(findings)}")
            smells = set(f.get("type", "unknown") for f in findings)
            print(f"Types of smells found: {', '.join(smells)}")
            print(f"F1 Score Metrics: {data.get('metrics', {})}")
            print(f"Summary: {data.get('summary', '')[:100]}...")
            return
        elif status == "failed":
            print(f"Job failed: {data.get('error')}")
            return
        time.sleep(10)

if __name__ == "__main__":
    run_test()
