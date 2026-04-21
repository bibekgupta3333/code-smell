# Enhanced Multi-Agent Code Smell Detector

## Overview

The **Enhanced Code Smell Detector** now supports detection of **all 10 types of code smells**:

1. **Long Method** - Functions exceeding 50-100 lines of code
2. **Deep Nesting** - Nesting depth > 3-4 levels
3. **Duplicated Code** - Identical/similar blocks with >85% similarity
4. **High Cyclomatic Complexity** - >10-15 decision paths
5. **God Class** - Classes with >15 methods and multiple responsibilities
6. **Data Clump** - Same parameters repeated across 3+ functions
7. **Magic Numbers** - Hard-coded values without semantic meaning
8. **Unused Variables** - Dead code, unused imports, unreachable statements
9. **Inconsistent Naming** - Mixed naming conventions, unclear variable names
10. **Missing Error Handling** - Risky operations without try/except blocks

---

## Architecture

### Dual Detection Approach

The detector uses a **two-tier detection strategy**:

#### Tier 1: LLM-Based Detection (Primary)
- Uses selected LLM (llama3, mistral, codellama) with agentic reasoning
- Comprehensive prompt asks for all 10 smell types explicitly
- Leverages RAG context for pattern matching
- Provides semantic understanding and business logic awareness

#### Tier 2: Metric-Based Detection (Fallback)
- Automatically triggers if LLM parsing fails
- Computes metrics for each smell type:
  - **Lines of Code, Cyclomatic Complexity** → Long Method, High CC
  - **Nesting Depth** → Deep Nesting
  - **Token Similarity** → Duplicated Code
  - **Parameter Count** → Data Clump
  - **Magic Number Frequency** → Magic Numbers
  - **Variable Usage Analysis** → Unused Variables
  - **Naming Convention Analysis** → Inconsistent Naming
  - **Risky Operation Detection** → Missing Error Handling
  - **Method Count + LOC** → God Class

---

## Usage

### Via Web UI

1. **Navigate to** `http://localhost:8000`
2. **Select Language** (Java, Python, JavaScript)
3. **Paste Code** (single file or focused snippet, recommended < 500 LOC)
4. **Configure Options**:
   - Enable RAG: Provides similar pattern context
   - Model: Auto-select (agentic) or specify (llama3:8b, codellama:13b)
   - Timeout: 30-600 seconds
5. **Submit** and monitor the **agent graph** showing workflow progress
6. **Review Results** - All detected smells with:
   - Smell type and location (line range)
   - Severity level (CRITICAL, HIGH, MEDIUM, LOW)
   - Confidence score (0.0-1.0)
   - Explanation with specific metrics
   - Refactoring suggestions

### Via Python API

```python
import asyncio
from src.analysis.code_smell_detector import CodeSmellDetector

async def analyze():
    detector = CodeSmellDetector(
        specialization="Comprehensive analyzer",
        model="llama3:8b"
    )
    
    code = """
def process_data(user, config, cache):
    result = []
    for item in user.items:
        if item.valid:
            processed = item.process()
            result.append(processed)
            if len(result) > 1000:
                result = []
    return result
    """
    
    findings = await detector.detect_smells(code, use_rag=True)
    
    for finding in findings:
        print(f"✗ {finding.smell_type} ({finding.severity})")
        print(f"  Location: {finding.location}")
        print(f"  Confidence: {finding.confidence:.2f}")
        print(f"  Explanation: {finding.explanation}")
        print(f"  Fix: {finding.refactoring}\n")

asyncio.run(analyze())
```

### Via REST API

```bash
# Submit analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def long_func():\n    # 150+ lines\n    ...",
    "language": "python",
    "file_name": "example.py",
    "include_rag": true,
    "timeout": 300,
    "model": null
  }'

# Response: {"analysis_id": "abc123", "status": "queued"}

# Check progress
curl http://localhost:8000/api/v1/progress/abc123

# Get results
curl http://localhost:8000/api/v1/results/abc123
```

---

## Metric Thresholds

| Smell | Metric | Threshold | Severity |
|-------|--------|-----------|----------|
| **Long Method** | LOC | > 100 | CRITICAL |
| | | 50-100 | HIGH |
| | | 20-50 | MEDIUM |
| **Deep Nesting** | Max Depth | > 6 | HIGH |
| | | 4-6 | MEDIUM |
| **Duplicated Code** | Similarity | > 90% | HIGH |
| | | 85-90% | MEDIUM |
| **Cyclomatic Complexity** | CC | > 15 | HIGH |
| | | 10-15 | MEDIUM |
| **Data Clump** | Param Frequency | 3+ functions | MEDIUM |
| **Magic Numbers** | Count | > 3 | LOW |
| **Unused Variables** | Found | Any | MEDIUM |
| **Inconsistent Naming** | Entropy | > 0.60 | LOW |
| **Missing Error Handling** | Risky Ops | Any unhandled | HIGH |
| **God Class** | Methods + LOC | > 15 methods OR > 300 LOC | HIGH |

---

## Confidence Scoring

Each finding includes a **confidence score (0.0-1.0)**:

$$\text{Confidence} = \frac{\text{Metric Alignment} + \text{LLM Signal} + \text{RAG Similarity}}{3}$$

- **0.85-1.00** — Very High: Strong metric + LLM agreement, act immediately
- **0.70-0.84** — High: Good evidence, review context
- **0.55-0.69** — Medium: Some signals, investigate further
- **< 0.55** — Low: Weak signals, manual review recommended

---

## Examples

### Example 1: Long Method Detection

**Input Code (Python)**:
```python
def process_order(order_id, customer_id, payment_method):
    # 120+ lines of mixed logic
    validate_order(order_id)
    check_inventory(order_id)
    calculate_tax(customer_id)
    apply_discount(customer_id)
    process_payment(payment_method)
    send_confirmation(customer_id)
    update_shipping(order_id)
    log_transaction(order_id)
    # ... more business logic ...
```

**Detection Result**:
```json
{
  "smell_type": "Long Method",
  "location": "line 1-125",
  "severity": "CRITICAL",
  "confidence": 0.89,
  "explanation": "Method spans 125 LOC with 12 distinct responsibilities. Cyclomatic complexity: 8. Multiple business logic concerns mixed.",
  "refactoring": "Extract into focused functions: validate(), process_inventory(), calculate_costs(), charge_payment(), notify_customer(), update_shipping()."
}
```

### Example 2: Deep Nesting + High Complexity

**Input Code (JavaScript)**:
```javascript
function processData(users) {
  for (let i = 0; i < users.length; i++) {
    if (users[i].active) {
      for (let j = 0; j < users[i].orders.length; j++) {
        if (users[i].orders[j].status === 'pending') {
          for (let k = 0; k < users[i].orders[j].items.length; k++) {
            if (users[i].orders[j].items[k].quantity > 0) {
              // Process item
            }
          }
        }
      }
    }
  }
}
```

**Detection Results**:
```json
[
  {
    "smell_type": "Deep Nesting",
    "location": "line 2-14",
    "severity": "HIGH",
    "confidence": 0.92,
    "explanation": "Nesting depth: 5 levels. Severely impacts readability and testability.",
    "refactoring": "Use early returns and extract to helper functions. Example: for (user of users) { processUser(user); }"
  },
  {
    "smell_type": "High Cyclomatic Complexity",
    "location": "line 1-14",
    "severity": "HIGH",
    "confidence": 0.85,
    "explanation": "Cyclomatic Complexity: 11. Multiple nested conditionals.",
    "refactoring": "Extract conditional logic to separate functions: filterActive(), filterPending(), filterByQuantity()."
  }
]
```

### Example 3: Missing Error Handling + Magic Numbers

**Input Code (Python)**:
```python
def load_config(filepath):
    with open(filepath) as f:  # ← Risky: no try/except
        data = json.load(f)    # ← Risky: no try/except
    
    if data['version'] != 2:   # ← Magic number
        raise ValueError()
    
    timeout = data.get('timeout', 3600)  # ← Magic number
    max_retries = data.get('retries', 5) # ← Magic number
    
    return data
```

**Detection Results**:
```json
[
  {
    "smell_type": "Missing Error Handling",
    "location": "line 2-3",
    "severity": "HIGH",
    "confidence": 0.88,
    "explanation": "File I/O + JSON parsing without error handling. No try/except blocks.",
    "refactoring": "Wrap in try/except: try:\\n    with open(...) as f: data = json.load(f)\\nexcept FileNotFoundError: ...",
    "location": "line 2-3"
  },
  {
    "smell_type": "Magic Numbers",
    "location": "line 1-11",
    "severity": "LOW",
    "confidence": 0.72,
    "explanation": "Found 3 magic numbers: 2, 3600, 5 (without semantic meaning).",
    "refactoring": "Define constants: CONFIG_VERSION = 2, DEFAULT_TIMEOUT = 3600, DEFAULT_RETRIES = 5"
  }
]
```

---

## For Comprehensive Details

See the full guide with architecture, best practices, and troubleshooting:

**📖 [Multi-Agent Code Smell Investigation Guide](../docs/MULTIAGENT_CODE_SMELL_INVESTIGATION.md)**

---

## Stats & Monitoring

The detector tracks statistics for quality assurance:

```python
detector = CodeSmellDetector()
findings = await detector.detect_smells(code)
stats = detector.get_stats()

print(f"Detections: {stats['detections_count']}")
print(f"Avg Confidence: {stats['average_confidence']:.2f}")
print(f"Avg Latency: {stats['average_latency']:.2f}s")
print(f"Tools Invoked: {stats['tools_invoked_count']}")
```

---

## API Response Format

```json
{
  "analysis_id": "e4445e5f-50a6-4685-83d7-960745ccb33d",
  "status": "completed",
  "findings": [
    {
      "smell_type": "Long Method",
      "location": "line 45-150",
      "severity": "HIGH",
      "confidence": 0.82,
      "explanation": "Spans 106 lines with CC=11...",
      "refactoring": "Extract smaller functions...",
      "agent_name": "detector_general",
      "timestamp": "2026-04-21T10:30:45.123Z"
    }
  ],
  "metrics": {
    "critical_count": 1,
    "high_count": 3,
    "medium_count": 2,
    "low_count": 1
  },
  "analysis_time_ms": 2450,
  "model_used": "llama3:8b",
  "model_reasoning": "Selected llama3:8b for Python code (150 LOC); good balance of speed and accuracy."
}
```
