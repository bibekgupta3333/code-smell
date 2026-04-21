# Multi-Agent Code Smell Investigation Guide

## Overview

This guide explains how to use the **Multi-Agent Code Smell Detection System** to identify and differentiate code smells in small-to-medium codebases. The system uses LangGraph-based workflow orchestration with agentic LLM model selection to detect, validate, and classify code quality issues.

---

## 1. What is Multi-Agent Code Smell Detection?

### Core Concept

Multi-agent code smell detection is a workflow-based approach that decomposes code quality analysis into specialized agents, each responsible for a distinct stage of the detection pipeline:

1. **Parser Agent** — Validates syntax, extracts metrics
2. **Model Selection Agent** — Intelligently chooses the best LLM based on code characteristics
3. **Chunking Agent** — Breaks large code into manageable pieces
4. **RAG Agent** — Retrieves relevant examples from knowledge base
5. **Detection Agent** — Identifies potential code smells using the selected model
6. **Validation Agent** — Filters false positives and ranks findings
7. **Aggregation Agent** — Compiles final results with severity scores

### Why Multi-Agent?

- **Specialization**: Each agent focuses on one task
- **Agentic Reasoning**: Model selection is intelligent, not hard-coded
- **Resilience**: Failures in one stage don't crash the pipeline
- **Observability**: Live progress tracking of each workflow stage
- **Scalability**: Easy to add or modify agents

---

## 2. Supported Code Smell Types

The system detects and differentiates the following code smells:

### 2.1 Long Method

**Definition**: A method/function that does too much and is hard to understand.

**Indicators**:
- Total lines > 50–100 (context-dependent)
- Cyclomatic complexity > 10
- Multiple responsibilities
- Difficult to name succinctly

**Detection Method**:
- LLM analyzes function body, loop nesting, and responsibility count
- Compares against training data of well-factored code
- Confidence score based on metric alignment

**Refactoring Suggestion**:
```python
# Before: Long method
def process_data(data):
    # 150+ lines of various tasks
    validate_input(data)
    transform_data(data)
    save_results(data)
    send_notification(data)
    return results

# After: Extract methods
def process_data(data):
    validate_input(data)
    transform_data(data)
    save_and_notify(data)
    return results
```

### 2.2 Deep Nesting

**Definition**: Code with excessive nesting depth (nested loops, conditionals, etc.).

**Indicators**:
- Nesting depth > 3–4 levels
- Reduced readability
- Harder to test and maintain

**Detection Method**:
- Parse AST to compute max nesting depth
- LLM verifies complexity is justified
- Compares against codebase baseline

**Refactoring Suggestion**:
```python
# Before: Deep nesting
for item in items:
    if item.valid:
        for sub in item.subitems:
            if sub.active:
                for detail in sub.details:
                    if detail.important:
                        process(detail)

# After: Early return / helper functions
for item in items:
    if not item.valid:
        continue
    for sub in item.subitems:
        process_active_details(sub)
```

### 2.3 Duplicated Code

**Definition**: Identical or near-identical code blocks in multiple locations.

**Indicators**:
- Token-level similarity > 85%
- Same logic implemented multiple times
- Maintenance burden (changes required in multiple places)

**Detection Method**:
- Tokenize code blocks and compute similarity hashes
- LLM confirms semantic equivalence
- Suggests common extraction point

**Refactoring Suggestion**:
```python
# Before: Duplication
def validate_user(user):
    if not user.email or "@" not in user.email:
        raise ValueError("Invalid email")
    if not user.name or len(user.name) < 2:
        raise ValueError("Invalid name")

def validate_admin(admin):
    if not admin.email or "@" not in admin.email:
        raise ValueError("Invalid email")
    if not admin.name or len(admin.name) < 2:
        raise ValueError("Invalid name")

# After: Extract common logic
def validate_person(person):
    if not person.email or "@" not in person.email:
        raise ValueError("Invalid email")
    if not person.name or len(person.name) < 2:
        raise ValueError("Invalid name")

def validate_user(user):
    validate_person(user)

def validate_admin(admin):
    validate_person(admin)
```

### 2.4 High Cyclomatic Complexity

**Definition**: A function with too many independent decision paths.

**Indicators**:
- Complexity score > 10–15
- Many `if`, `else if`, `switch` branches
- Hard to understand control flow
- Difficult to test comprehensively

**Detection Method**:
- Compute cyclomatic complexity using AST
- LLM identifies unnecessarily complex decision trees
- Suggests simplification patterns (polymorphism, lookup tables, etc.)

**Refactoring Suggestion**:
```python
# Before: High complexity
def calculate_discount(customer_type, age, purchase_amount):
    if customer_type == "premium":
        if age > 65:
            return 0.20
        elif age > 18:
            return 0.15
    elif customer_type == "regular":
        if age > 65:
            return 0.10
        elif age > 18:
            return 0.05
    else:
        if purchase_amount > 1000:
            return 0.03
    return 0

# After: Lookup table / strategy pattern
DISCOUNT_MATRIX = {
    "premium": {"senior": 0.20, "adult": 0.15, "other": 0.0},
    "regular": {"senior": 0.10, "adult": 0.05, "other": 0.0},
    "guest": {"senior": 0.05, "adult": 0.03, "other": 0.01},
}

def calculate_discount(customer_type, age, purchase_amount):
    category = "senior" if age > 65 else ("adult" if age > 18 else "other")
    return DISCOUNT_MATRIX.get(customer_type, {}).get(category, 0)
```

### 2.5 God Class / God Object

**Definition**: A class that does too much and violates the Single Responsibility Principle.

**Indicators**:
- Class has > 10–15 public methods
- Multiple responsibilities (UI, business logic, persistence, etc.)
- Low cohesion: methods don't use shared state
- Large file size (> 500 lines)

**Detection Method**:
- Count methods, analyze interdependencies
- LLM groups methods by responsibility
- Scores single-responsibility adherence

**Refactoring Suggestion**:
```python
# Before: God Class
class UserManager:
    def create_user(self, data): pass
    def update_user(self, user_id, data): pass
    def validate_email(self, email): pass
    def send_email(self, recipient, subject, body): pass
    def hash_password(self, password): pass
    def save_to_database(self, user): pass
    def generate_report(self, user_id): pass
    def render_html(self, user): pass

# After: Separated concerns
class UserService:
    def create_user(self, data): pass
    def update_user(self, user_id, data): pass

class UserValidator:
    def validate_email(self, email): pass

class EmailService:
    def send_email(self, recipient, subject, body): pass

class PasswordService:
    def hash_password(self, password): pass

class UserRepository:
    def save(self, user): pass

class ReportGenerator:
    def generate_report(self, user_id): pass
```

### 2.6 Data Clump

**Definition**: Groups of variables that are commonly used together but not encapsulated.

**Indicators**:
- Same set of parameters passed repeatedly
- Related data scattered across multiple variables
- Could be encapsulated in a class/dataclass

**Detection Method**:
- Track parameter patterns across functions
- Identify recurring tuples of data
- LLM suggests encapsulation

**Refactoring Suggestion**:
```python
# Before: Data clump
def calculate_shipping(address_line1, address_line2, city, state, zip_code, country):
    pass

def validate_address(address_line1, address_line2, city, state, zip_code, country):
    pass

def format_address(address_line1, address_line2, city, state, zip_code, country):
    pass

# After: Encapsulate in dataclass
from dataclasses import dataclass

@dataclass
class Address:
    line1: str
    line2: str
    city: str
    state: str
    zip_code: str
    country: str

def calculate_shipping(address: Address): pass
def validate_address(address: Address): pass
def format_address(address: Address): pass
```

### 2.7 Magic Numbers / Magic Strings

**Definition**: Hard-coded literal values with no clear meaning.

**Indicators**:
- Unexplained numeric or string constants
- Repeated literals (e.g., 30, "admin", 0.5)
- No accompanying comment or named constant

**Detection Method**:
- Scan for literal values
- Check if similar values appear elsewhere
- LLM verifies lack of semantic clarity

**Refactoring Suggestion**:
```python
# Before: Magic numbers
def is_valid_age(age):
    return age >= 18 and age <= 120

def apply_senior_discount(age, price):
    if age >= 65:  # Magic number
        return price * 0.85  # Magic number

# After: Named constants
MIN_LEGAL_AGE = 18
MAX_EXPECTED_AGE = 120
SENIOR_THRESHOLD = 65
SENIOR_DISCOUNT = 0.15

def is_valid_age(age):
    return MIN_LEGAL_AGE <= age <= MAX_EXPECTED_AGE

def apply_senior_discount(age, price):
    if age >= SENIOR_THRESHOLD:
        return price * (1 - SENIOR_DISCOUNT)
```

### 2.8 Unused Variables / Dead Code

**Definition**: Variables or code that are declared but never used.

**Indicators**:
- Local variables assigned but not read
- Unused imports
- Unreachable code blocks
- Dead branches (e.g., after `return`)

**Detection Method**:
- SSA (Static Single Assignment) analysis
- Unused import detection
- Linter-based unreachable code detection

**Refactoring Suggestion**:
```python
# Before: Dead code
def process_order(order):
    total = 0  # ← Unused (value never read)
    discount_rate = 0.1  # ← Unused
    
    if order.status == "cancelled":
        return None
        print("Order cancelled")  # ← Unreachable
    
    calculate_total(order)

# After: Cleaned up
def process_order(order):
    if order.status == "cancelled":
        return None
    
    return calculate_total(order)
```

### 2.9 Inconsistent Naming

**Definition**: Variables, functions, or classes with unclear or inconsistent naming conventions.

**Indicators**:
- Single-letter variable names (except loop counters: `i`, `j`)
- Misleading names that don't match responsibility
- Inconsistent casing (`user_id` vs `userId`)
- Abbreviations without explanation

**Detection Method**:
- Analyze variable names and their usage
- Check naming consistency (camelCase vs snake_case)
- LLM verifies clarity and correctness

**Refactoring Suggestion**:
```python
# Before: Inconsistent naming
def calc(a, b, c):
    x = a * b
    y = x + c
    return y

# After: Clear naming
def calculate_total_cost(unit_price, quantity, tax_amount):
    subtotal = unit_price * quantity
    total = subtotal + tax_amount
    return total
```

### 2.10 Missing Error Handling

**Definition**: Code that doesn't handle exceptions or edge cases properly.

**Indicators**:
- No try/except blocks around risky operations
- No validation of input parameters
- No null/None checks
- Unhandled API failures or file I/O errors

**Detection Method**:
- Identify risky operations (file I/O, network, parsing, indexing)
- Check for exception handling
- LLM verifies completeness

**Refactoring Suggestion**:
```python
# Before: No error handling
def load_user_data(user_id):
    with open(f"users/{user_id}.json") as f:
        return json.load(f)

def get_user_age(user_id):
    data = load_user_data(user_id)
    return data["age"]

# After: Proper error handling
def load_user_data(user_id):
    try:
        with open(f"users/{user_id}.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"User {user_id} not found")
    except json.JSONDecodeError:
        raise ValueError(f"User data for {user_id} is corrupted")

def get_user_age(user_id):
    try:
        data = load_user_data(user_id)
        return data.get("age")
    except (ValueError, KeyError) as e:
        logger.error(f"Failed to get age for user {user_id}: {e}")
        raise
```

---

## 3. How the System Differentiates Code Smells

### 3.1 Metric-Based Differentiation

The system computes multiple metrics to distinguish smells:

| Smell | Key Metrics | Threshold |
|-------|------------|-----------|
| **Long Method** | LOC, Cyclomatic Complexity | LOC > 100 OR CC > 10 |
| **Deep Nesting** | Max Nesting Depth | Depth > 4 |
| **Duplication** | Token Similarity | Similarity > 85% |
| **High Complexity** | Cyclomatic Complexity | CC > 15 |
| **God Class** | Method Count, Cohesion | Methods > 15 OR Cohesion < 0.5 |
| **Data Clump** | Parameter Frequency | Same 3+ params in 3+ functions |
| **Magic Numbers** | Literal Count, Entropy | Unlabeled constants > 3 |
| **Unused Variables** | SSA Analysis | Variables never read |
| **Inconsistent Naming** | Naming Pattern Entropy | Inconsistency score > 0.6 |
| **Missing Error Handling** | Exception Coverage | Coverage < 80% for risky ops |

### 3.2 LLM-Based Differentiation

The selected LLM (based on code size and language) provides:

1. **Contextual Understanding**: Grasps domain-specific naming and patterns
2. **False Positive Filtering**: Eliminates metric anomalies
3. **Severity Assessment**: Rates impact on maintainability/performance
4. **Refactoring Suggestions**: Proposes concrete fixes

### 3.3 Confidence Scoring

Each finding includes a confidence score (0.0–1.0):

$$\text{Confidence} = \frac{\text{Metric Alignment} + \text{LLM Confidence} + \text{Training Data Similarity}}{3}$$

- **High Confidence** (≥ 0.75): Metric agreement + strong LLM signal
- **Medium Confidence** (0.50–0.75): Some ambiguity, but likely valid
- **Low Confidence** (< 0.50): Reported but flagged for manual review

---

## 4. Using the System

### 4.1 Web Interface

Access the analysis dashboard at `http://localhost:8000`:

1. **Select Language**: Java, Python, JavaScript
2. **Paste Code**: Single file or focused snippet (recommended < 500 LOC)
3. **Configure Options**:
   - **Enable RAG**: Retrieves similar examples for better context
   - **Model Selection**: Auto-select (agentic) or specify (e.g., `llama3:8b`)
   - **Timeout**: 30–600 seconds (default: 300s)
4. **Submit**: Click "Analyze Code"
5. **Monitor Progress**: Watch the agent graph show each workflow stage
6. **Review Results**: View detected smells with severity, confidence, and suggestions

### 4.2 API Usage

#### Request
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def long_method():\n    # 150+ lines\n    ...",
    "language": "python",
    "file_name": "example.py",
    "include_rag": true,
    "timeout": 300,
    "model": null
  }'
```

#### Response (Async, returns `analysis_id`)
```json
{
  "analysis_id": "e4445e5f-50a6-4685-83d7-960745ccb33d",
  "status": "queued"
}
```

#### Poll Progress
```bash
curl http://localhost:8000/api/v1/progress/{analysis_id}
```

#### Get Results
```bash
curl http://localhost:8000/api/v1/results/{analysis_id}
```

### 4.3 Command-Line Usage

```bash
# Analyze a file
python -m src.analysis.code_smell_detector \
  --file path/to/code.py \
  --language python \
  --include-rag

# Output: findings.json with all detected smells
```

---

## 5. Interpreting Results

### 5.1 Finding Structure

```json
{
  "smell_type": "Long Method",
  "severity": "high",
  "confidence": 0.89,
  "location": {
    "file": "services/user_service.py",
    "line": 45,
    "end_line": 180
  },
  "explanation": "Function 'process_user_data' spans 135 lines with 8 decision points...",
  "metrics": {
    "lines_of_code": 135,
    "cyclomatic_complexity": 8,
    "functions_called": 12
  },
  "suggested_refactoring": "Extract nested loops into separate methods: 'validate_step', 'transform_step', 'persist_step'.",
  "model_used": "llama3:8b",
  "model_reasoning": "Selected llama3:8b for Python code (135 lines); good balance of speed and accuracy."
}
```

### 5.2 Severity Levels

| Severity | Meaning | Action |
|----------|---------|--------|
| **Critical** | High impact on maintenance; likely bugs | Fix immediately |
| **High** | Significant code quality issue | Fix in next sprint |
| **Medium** | Noticeable but not urgent | Consider refactoring |
| **Low** | Minor style or convention issue | Nice to have |

### 5.3 Confidence Interpretation

- **0.90–1.00**: Highly reliable, act on finding
- **0.70–0.89**: Likely valid, review context
- **0.50–0.69**: Possible false positive, investigate further
- **< 0.50**: Low confidence, manual review recommended

---

## 6. Best Practices

### 6.1 For Small Codebases (< 1000 LOC)

1. **Analyze incrementally**: Submit one file or function at a time
2. **Use auto model selection**: System chooses `orca-mini:7b` or `neural-chat:7b` for speed
3. **Enable RAG**: Helps for small samples with limited context
4. **Review high-confidence findings first**: Focus on > 0.80 confidence

### 6.2 For Different Languages

| Language | Recommended Model | Notes |
|----------|------------------|-------|
| **Python** | `llama3:8b` | Excellent code understanding |
| **Java** | `codellama:13b` | Code-specialized; slower but accurate |
| **JavaScript** | `neural-chat:7b` | Fast; good for web code |

### 6.3 Iterative Refinement

1. **First Pass**: Run full analysis, note high-confidence findings
2. **Refactor**: Fix top 3–5 issues
3. **Re-analyze**: Run again to verify fixes and uncover secondary issues
4. **Validate**: Check that suggested refactorings don't break tests

---

## 7. Troubleshooting

### Issue: "Parse error, using fallback"

**Cause**: Code has syntax errors or unusual patterns.

**Solution**:
- Ensure code is syntactically valid
- For non-standard DSLs, provide minimal reproducible example
- Check language selection is correct

### Issue: Low confidence scores across all findings

**Cause**: Model unfamiliar with code style; insufficient context.

**Solution**:
- Enable RAG to provide examples
- Use explicit model selection (e.g., `codellama:13b`)
- Submit smaller, more focused code samples

### Issue: No findings detected

**Cause**: Code is genuinely clean, or model too conservative.

**Solution**:
- Review manually for obvious smells
- Try different model (e.g., `llama3:8b` vs `mistral:7b`)
- Disable RAG and re-run

### Issue: Timeout exceeded

**Cause**: Complex code or slow model.

**Solution**:
- Reduce code size (break into smaller files)
- Increase timeout value (up to 600s)
- Use faster model (e.g., `orca-mini:7b`)

---

## 8. Example: End-to-End Investigation

### Scenario

Investigate code smells in a small user authentication module (250 LOC, Python).

### Workflow

1. **Copy code to web interface**
   - Language: Python
   - Enable RAG: Yes
   - Model: Auto-select (agentic)
   - Timeout: 300s

2. **Submit and monitor**
   - Watch agent graph: `START → Parse → Select Model (llama3:8b) → Chunk → RAG → Detect → Validate → Aggregate → END`
   - Progress: 0% → 100% (~15 seconds for 250 LOC)

3. **Review results** (example output)

   | Smell | Severity | Confidence | Action |
   |-------|----------|-----------|--------|
   | Missing Error Handling (line 45–60) | High | 0.92 | Add try/except for password validation |
   | Magic Numbers (line 78) | Medium | 0.71 | Replace `86400` with `SECONDS_PER_DAY` |
   | Deep Nesting (line 120) | Medium | 0.68 | Refactor nested loops with early returns |

4. **Implement fixes**
   - Extract password validation into separate function with error handling
   - Define constants for magic numbers
   - Simplify control flow in validation loop

5. **Re-analyze**
   - High-confidence findings resolved ✓
   - Confidence improved on remaining low-priority items

---

## 9. Architecture & Workflow Diagram

```
User Submission
    ↓
[1] Parse Code Node
    ↓ (Extract metrics, validate syntax)
[2] Select Model Node (AGENTIC)
    ↓ (Choose llama3, mistral, codellama based on size/language)
[3] Chunk Code Node
    ↓ (Split into functions for parallel processing)
[4] RAG Retrieval Node
    ↓ (Fetch similar examples from ChromaDB)
[5] Detect Smells Node
    ↓ (Run detection with selected LLM + context)
[6] Validate Findings Node
    ↓ (Filter false positives, rank by severity)
[7] Aggregate Results Node
    ↓ (Compile metrics, confidence scores, suggestions)
Results Display
    ↓
Frontend (Live Agent Graph + Findings List)
```

---

## 10. References

- **LangGraph Documentation**: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- **Code Smell Catalog** (Refactoring, Fowler): [https://refactoring.guru/refactoring](https://refactoring.guru/refactoring)
- **Cyclomatic Complexity**: [https://en.wikipedia.org/wiki/Cyclomatic_complexity](https://en.wikipedia.org/wiki/Cyclomatic_complexity)
- **SonarQube Smells**: [https://www.sonarqube.org/](https://www.sonarqube.org/)

---

## Appendix: Severity Rating Formula

$$\text{Severity} = \frac{\text{Impact} + \text{Confidence}}{2}$$

Where:

- **Impact** (0–1): Estimated effect on maintainability, performance, or correctness
  - 1.0: Critical (likely bugs, major refactoring needed)
  - 0.75: High (noticeable quality issue)
  - 0.5: Medium (minor issue, good to fix)
  - 0.25: Low (cosmetic)

- **Confidence** (0–1): Model + metric agreement (see Section 3.3)

**Severity Mapping**:
- **Score ≥ 0.80** → Critical
- **Score 0.60–0.80** → High
- **Score 0.40–0.60** → Medium
- **Score < 0.40** → Low
