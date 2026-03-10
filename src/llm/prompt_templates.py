"""
Prompt Templates for Code Smell Detection
Defines system prompts, few-shot examples, and RAG context templates.

Architecture: Implements Gap #12 (Production code) and Gap #11 (AI-generated code support)
"""

import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are an expert code smell detector specializing in identifying code quality issues.

Your task is to analyze code snippets and detect code smells - patterns indicating deeper design problems.

## Code Smell Categories

**Bloaters** - Code that is too long or too complex:
- Long Method: Methods with too many lines or responsibilities
- God Class: Classes that do too much
- Large Class: Classes with too many instance variables
- Long Parameter List: Methods with too many parameters
- Primitive Obsession: Overuse of primitives instead of small objects

**Couplers** - Excessive dependencies between modules:
- Feature Envy: Methods using data from other objects too much
- Inappropriate Intimacy: Classes that know too much about each other
- Message Chains: Long chains of method calls

**Change Preventers** - Code structure makes changes difficult:
- Divergent Change: Classes that change for different reasons
- Shotgun Surgery: Changes scattered across multiple small classes

**Dispensables** - Unnecessary code:
- Duplicate Code: Code repeated multiple times
- Lazy Class: Classes that contribute little value
- Data Class: Classes with only fields, no methods
- Dead Code: Unreachable or unused code

**Object-Oriented Abusers** - Misuse of OO principles:
- Switch Statements: Complex conditional logic
- Refused Bequest: Unused inherited methods
- Parallel Inheritance Hierarchies: Related inheritance structures

## Instructions

1. Analyze the provided code carefully
2. Identify all code smells present
3. For each smell:
   - Name the type of smell
   - Location (line numbers, method names)
   - Severity: LOW, MEDIUM, HIGH, CRITICAL
   - Brief explanation why it's a problem
   - Suggested refactoring (if applicable)

4. Support both human-written and AI-generated code
5. Be specific and actionable with feedback
6. Output ONLY valid JSON, no markdown or extra text

## Output Format

Return ONLY a JSON object with this structure - no other text:
{
  "code_smells": [
    {
      "type": "Long Method",
      "location": "function_name, lines 10-45",
      "severity": "HIGH",
      "explanation": "Method exceeds 50 lines",
      "refactoring": "Extract inner logic into separate methods"
    }
  ],
  "summary": "Found X code smell(s)",
  "is_valid_code": true,
  "notes": "Optional notes about the code"
}
"""

# ============================================================================
# FEW-SHOT EXAMPLES
# ============================================================================

EXAMPLE_LONG_METHOD = """
def process_user_data(user_data):
    # Validate input
    if not user_data:
        return None
    if 'name' not in user_data:
        raise ValueError("Name required")
    if 'email' not in user_data:
        raise ValueError("Email required")
    if 'age' not in user_data:
        raise ValueError("Age required")

    # Process name
    name = user_data['name'].strip().lower()
    name_parts = name.split()
    if len(name_parts) < 2:
        raise ValueError("Full name required")

    # Process email
    email = user_data['email'].strip().lower()
    if '@' not in email:
        raise ValueError("Invalid email")

    # Process age
    age = int(user_data['age'])
    if age < 0 or age > 150:
        raise ValueError("Invalid age")

    # Validate against database
    existing = database.find_user(email)
    if existing:
        return {"status": "exists", "user": existing}

    # Create new user
    user = User(name=name, email=email, age=age)
    database.save(user)
    logger.info(f"Created user: {user.id}")
    return {"status": "created", "user": user}
"""

EXAMPLE_GOD_CLASS = """
class UserManager:
    def __init__(self):
        self.users = []
        self.emails = set()
        self.admins = []
        self.audit_log = []

    def create_user(self, name, email):
        if email in self.emails:
            raise ValueError("Email exists")
        user = User(name, email)
        self.users.append(user)
        self.emails.add(email)
        self._log(f"Created user {user.id}")
        return user

    def update_user(self, user_id, **kwargs):
        user = self.find_user(user_id)
        if not user:
            raise ValueError("User not found")
        for key, value in kwargs.items():
            setattr(user, key, value)
        self._log(f"Updated user {user_id}")

    def send_email(self, user_id, subject, body):
        user = self.find_user(user_id)
        smtp.send(user.email, subject, body)
        self._log(f"Sent email to {user_id}")

    def generate_report(self):
        active = [u for u in self.users if u.is_active]
        inactive = [u for u in self.users if not u.is_active]
        return {"active": len(active), "inactive": len(inactive)}

    def _log(self, message):
        self.audit_log.append({"time": time.time(), "message": message})
"""

EXAMPLE_FEATURE_ENVY = """
class Order:
    def __init__(self, customer, items):
        self.customer = customer
        self.items = items

    def calculate_discount(self):
        # Accessing too much customer data
        if self.customer.membership_type == "gold":
            return 0.15
        elif self.customer.membership_type == "silver":
            return 0.10
        elif self.customer.membership_type == "bronze":
            return 0.05

        if self.customer.total_purchases > 1000:
            return 0.05

        if self.customer.account_age_years > 5:
            return 0.02

        return 0

class Item:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def apply_discount(self, discount):
        # Directly accessing order discounts
        self.price = self.price * (1 - discount)
"""

FEW_SHOT_EXAMPLES = {
    "Long Method": EXAMPLE_LONG_METHOD,
    "God Class": EXAMPLE_GOD_CLASS,
    "Feature Envy": EXAMPLE_FEATURE_ENVY,
}

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

PRODUCTION_CODE_ANALYSIS_PROMPT = """Analyze the following code for code smells. Focus on production code quality.

Code:
```
{code}
```

Analyze this code and return ONLY valid JSON with detected code smells."""

HUMAN_WRITTEN_CODE_ANALYSIS_PROMPT = """Analyze the following human-written code for code smells.

Code:
```
{code}
```

Analyze this code and return ONLY valid JSON with detected code smells."""

AI_GENERATED_CODE_ANALYSIS_PROMPT = """Analyze the following AI-generated code for code smells.
Note how AI-generated code may have patterns different from human code.

Code:
```
{code}
```

Analyze this code and return ONLY valid JSON with detected code smells."""

RAG_CONTEXT_TEMPLATE = """Here are similar code examples with detected code smells for reference:

{examples}

Now analyze the input code:
```
{code}
```

Return ONLY valid JSON with detected code smells."""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_system_prompt() -> str:
    """Get the system prompt for code smell detection."""
    return SYSTEM_PROMPT


def create_few_shot_example(smell_type: str) -> Optional[str]:
    """
    Get a few-shot example for a specific code smell type.

    Args:
        smell_type: Type of code smell (e.g., "Long Method")

    Returns:
        Example code or None if not found
    """
    return FEW_SHOT_EXAMPLES.get(smell_type)


def create_production_analysis_prompt(code: str) -> str:
    """
    Create prompt for analyzing production code.

    Args:
        code: Code to analyze

    Returns:
        Formatted prompt
    """
    return PRODUCTION_CODE_ANALYSIS_PROMPT.format(code=code)


def create_human_code_analysis_prompt(code: str) -> str:
    """
    Create prompt for analyzing human-written code.

    Args:
        code: Code to analyze

    Returns:
        Formatted prompt
    """
    return HUMAN_WRITTEN_CODE_ANALYSIS_PROMPT.format(code=code)


def create_ai_generated_code_analysis_prompt(code: str) -> str:
    """
    Create prompt for analyzing AI-generated code.

    Args:
        code: Code to analyze

    Returns:
        Formatted prompt
    """
    return AI_GENERATED_CODE_ANALYSIS_PROMPT.format(code=code)


def create_rag_prompt(
    code: str,
    examples: List[Dict[str, str]],
) -> str:
    """
    Create prompt with RAG context.

    Args:
        code: Code to analyze
        examples: List of example code smells with context

    Returns:
        Formatted prompt with context
    """
    examples_text = "\n\n".join([
        f"### Example: {ex['smell_type']}\n\nCode:\n```\n{ex['code']}\n```\n\nSmells: {ex['smells']}"
        for ex in examples
    ])

    return RAG_CONTEXT_TEMPLATE.format(code=code, examples=examples_text)


def create_prompt_with_few_shot(
    code: str,
    smell_types: Optional[List[str]] = None,
    use_rag: bool = False,
) -> str:
    """
    Create prompt with few-shot examples.

    Args:
        code: Code to analyze
        smell_types: Specific smells to focus on
        use_rag: Include RAG context

    Returns:
        Formatted prompt
    """
    if use_rag:
        # Would be enhanced with RAG context
        return create_production_analysis_prompt(code)

    prompt = PRODUCTION_CODE_ANALYSIS_PROMPT.format(code=code)

    if smell_types:
        examples = []
        for smell_type in smell_types:
            example = create_few_shot_example(smell_type)
            if example:
                examples.append(example)

        if examples:
            prompt += "\n\nReference examples:\n" + "\n\n".join(examples)

    return prompt


def format_analysis_for_llm(code: str) -> str:
    """
    Format code for LLM analysis with length limits for M4 Pro optimization.

    Args:
        code: Raw code string

    Returns:
        Formatted code for analysis
    """
    lines = code.split('\n')
    # Limit to 100 lines for M4 Pro efficiency
    if len(lines) > 100:
        lines = lines[:100]
        code = '\n'.join(lines) + "\n... (truncated)"

    return code


if __name__ == "__main__":
    # Test prompts
    test_code = """
    def process_user(user_data):
        if not user_data:
            return None
        name = user_data['name'].strip()
        email = user_data['email'].lower()
        age = int(user_data['age'])
        # ... more logic
        return User(name, email, age)
    """

    print("System Prompt:")
    print("=" * 60)
    print(get_system_prompt()[:200] + "...")
    print()

    print("Production Analysis Prompt:")
    print("=" * 60)
    print(create_production_analysis_prompt(test_code))
    print()

    print("Few-shot Example (Long Method):")
    print("=" * 60)
    print(create_few_shot_example("Long Method")[:200] + "...")
