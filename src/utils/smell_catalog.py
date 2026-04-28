"""
Code Smell Catalog - Single Source of Truth for All Supported Smells.

Aligns detection coverage with SonarQube, PMD, and classical Fowler taxonomies.
Provides:
  - CANONICAL_SMELLS: 28 smell types organized by category
  - SMELL_ALIASES: normalization map for LLM output variants
  - SMELL_DEFINITIONS: short definitions used in LLM prompt and UI
  - normalize_smell_type(): canonicalize any incoming label
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ============================================================================
# Canonical Smell Catalog (35 types)
# ============================================================================
# Grouped for LLM prompt clarity; flat list exposed as CANONICAL_SMELLS.
SMELL_CATALOG: Dict[str, List[str]] = {
    "Size-Related Smells": [
        "Long Method",
        "God Class",
        "Large Class",
        "Long Parameter List",
        "Primitive Obsession",
    ],
    "Object-Oriented Design Smells": [
        "Duplicate Code",
        "Feature Envy",
        "Inappropriate Intimacy",
        "Refused Bequest",
        "Speculative Generality",
    ],
    "Change Preventers": [
        "Divergent Change",
        "Shotgun Surgery",
        "Parallel Inheritance Hierarchies",
    ],
    "Dispensables": [
        "Dead Code",
        "Lazy Class",
        "Data Clumps",
        "Comments",
    ],
    "Coupling & Dependency Smells": [
        "Message Chains",
        "Middle Man",
        "Inappropriate Coupling",
    ],
    "Naming & Readability Smells": [
        "Poor Naming",
        "Magic Numbers",
        "Inconsistent Naming",
        "Inconsistent Style",
    ],
    "Control Flow Smells": [
        "Deep Nesting",
        "Switch Statements",
        "Flag Arguments",
    ],
    "Exception Handling Smells": [
        "Empty Catch Block",
        "Generic Exceptions",
        "Hidden Errors",
    ],
    "Legacy Categories": [
        "Temporary Field",
        "Alternative Classes with Different Interfaces",
        "Data Class",
        "High Cyclomatic Complexity",
        "Missing Error Handling",
    ],
}

# Flat canonical list
CANONICAL_SMELLS: List[str] = [
    smell for group in SMELL_CATALOG.values() for smell in group
]

# Category header names. LLMs occasionally return these (e.g. "Change
# Preventers", "Dispensables") as the smell_type instead of a specific smell.
# Callers can use this set to filter such responses without treating them as
# errors or spurious "unknown smell" warnings.
SMELL_CATEGORIES: set = set(SMELL_CATALOG.keys())

# Lowercased lookup for fast canonicalization
_CANONICAL_LOWER: Dict[str, str] = {s.lower(): s for s in CANONICAL_SMELLS}
_CATEGORY_LOWER: set = {c.lower() for c in SMELL_CATEGORIES}


def is_smell_category(label: Optional[str]) -> bool:
    """Return True if `label` is a category header rather than a specific smell."""
    if not label:
        return False
    return label.strip().lower() in _CATEGORY_LOWER


# ============================================================================
# Alias Map -> Canonical Name
# ============================================================================
# Accommodates LLM output variants, plurals, British/US spellings, etc.
SMELL_ALIASES: Dict[str, str] = {
    # Duplicate Code variants
    "duplicated code": "Duplicate Code",
    "code duplication": "Duplicate Code",
    "copy-paste code": "Duplicate Code",
    "copied code": "Duplicate Code",
    # Data Clumps
    "data clump": "Data Clumps",
    "parameter clump": "Data Clumps",
    # Long Parameter List
    "too many parameters": "Long Parameter List",
    "long parameters": "Long Parameter List",
    "excessive parameters": "Long Parameter List",
    # Dead Code / Unused
    "unused code": "Dead Code",
    "unused variables": "Dead Code",
    "unused variable": "Dead Code",
    "unused imports": "Dead Code",
    "unreachable code": "Dead Code",
    # Deep Nesting
    "nested conditionals": "Deep Nesting",
    "deeply nested code": "Deep Nesting",
    "excessive nesting": "Deep Nesting",
    # Cyclomatic Complexity
    "cyclomatic complexity": "High Cyclomatic Complexity",
    "complex method": "High Cyclomatic Complexity",
    "complex conditional": "High Cyclomatic Complexity",
    "complex function": "High Cyclomatic Complexity",
    # God Class / Large Class
    "god object": "God Class",
    "blob class": "God Class",
    # Magic Numbers
    "magic number": "Magic Numbers",
    "magic strings": "Magic Numbers",
    "magic string": "Magic Numbers",
    "hard-coded values": "Magic Numbers",
    "hardcoded values": "Magic Numbers",
    # Naming
    "naming convention": "Inconsistent Naming",
    "inconsistent naming convention": "Inconsistent Naming",
    "poor naming": "Poor Naming",
    "unclear names": "Poor Naming",
    "bad naming": "Poor Naming",
    "confusing names": "Poor Naming",
    "inconsistent style": "Inconsistent Style",
    "inconsistent formatting": "Inconsistent Style",
    "mixed formatting": "Inconsistent Style",
    "code style": "Inconsistent Style",
    # Error Handling
    "missing exception handling": "Missing Error Handling",
    "no error handling": "Missing Error Handling",
    "unhandled exceptions": "Missing Error Handling",
    # Empty Catch
    "empty catch": "Empty Catch Block",
    "swallowed exception": "Empty Catch Block",
    "empty except": "Empty Catch Block",
    "exception swallowing": "Empty Catch Block",
    # Generic Exceptions
    "catching all exceptions": "Generic Exceptions",
    "broad exception": "Generic Exceptions",
    "catch exception": "Generic Exceptions",
    "catching generic exception": "Generic Exceptions",
    # Hidden Errors
    "hidden errors": "Hidden Errors",
    "error hiding": "Hidden Errors",
    "swallowed errors": "Hidden Errors",
    "silently ignored": "Hidden Errors",
    "silent failure": "Hidden Errors",
    # Switch Statements
    "switch statement": "Switch Statements",
    "large switch": "Switch Statements",
    "long if-else chain": "Switch Statements",
    # Feature Envy
    "envy": "Feature Envy",
    # Message Chains
    "message chain": "Message Chains",
    "train wreck": "Message Chains",
    # Middle Man
    "middleman": "Middle Man",
    # Inappropriate Coupling
    "inappropriate coupling": "Inappropriate Coupling",
    "improper coupling": "Inappropriate Coupling",
    "unrelated coupling": "Inappropriate Coupling",
    "tight coupling": "Inappropriate Coupling",
    # Flag Arguments
    "flag argument": "Flag Arguments",
    "boolean flag": "Flag Arguments",
    "flag parameter": "Flag Arguments",
    "boolean parameter": "Flag Arguments",
    # Comments
    "comment smell": "Comments",
    "commented-out code": "Comments",
    "excessive comments": "Comments",
    # Temporary Field
    "temp field": "Temporary Field",
    # Primitive Obsession
    "primitive type obsession": "Primitive Obsession",
}


def normalize_smell_type(label: Optional[str]) -> Optional[str]:
    """Normalize an arbitrary smell label to the canonical catalog name.

    Returns ``None`` if the label cannot be normalized and is too ambiguous to
    keep (empty string, ``None``). Unknown but non-empty labels are returned as
    title-cased passthroughs so downstream code can still display them.
    """
    if not label:
        return None
    key = label.strip().lower()
    if not key:
        return None
    if key in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[key]
    if key in SMELL_ALIASES:
        return SMELL_ALIASES[key]
    # Drop trailing plural/singular to increase alias hit rate
    if key.endswith("s") and key[:-1] in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[key[:-1]]
    if (key + "s") in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[key + "s"]
    # Passthrough - keep original but title-cased
    return " ".join(w.capitalize() for w in label.strip().split())


# ============================================================================
# Short Definitions (used in LLM prompt and UI tooltips)
# ============================================================================
SMELL_DEFINITIONS: Dict[str, str] = {
    "Long Method": "Function/method longer than ~50-100 LOC or doing multiple things.",
    "God Class": "Class with too many responsibilities (>15 methods and/or >300 LOC).",
    "Large Class": "Class that has grown too large but may not yet qualify as god class.",
    "Long Parameter List": "Function signature with 5+ parameters (suggests poor abstraction).",
    "Primitive Obsession": "Overuse of primitives (str/int/dict) instead of dedicated types.",
    "Data Clumps": "Same group of 3+ variables/parameters appearing repeatedly together.",
    "Duplicate Code": "Two or more blocks of code that are identical or near-identical (>85% similar).",
    "Feature Envy": "Method that uses another class's data more than its own.",
    "Inappropriate Intimacy": "Classes that know too much about each other's internals.",
    "Refused Bequest": "Subclass that rejects or empties inherited behavior.",
    "Speculative Generality": "Unused abstractions added 'just in case'.",
    "Divergent Change": "One class changes for many unrelated reasons.",
    "Shotgun Surgery": "One change forces edits across many classes.",
    "Parallel Inheritance Hierarchies": "Every subclass of A requires a subclass of B.",
    "Dead Code": "Unused variables, functions, imports, or unreachable statements.",
    "Lazy Class": "Class that doesn't do enough to justify its existence.",
    "Data Class": "Class with fields/getters/setters and no behavior.",
    "Comments": "Comments that mask bad code or commented-out code left in place.",
    "Message Chains": "Long chain of calls like a.b().c().d().e().",
    "Middle Man": "Class that only delegates to another class.",
    "Inappropriate Coupling": "Unrelated modules or classes tightly linked with dependencies.",
    "Poor Naming": "Unclear, single-letter, or ambiguous variable/function/class names.",
    "Magic Numbers": "Unexplained numeric or string literals without named constants.",
    "Inconsistent Naming": "Mixed naming conventions (camelCase vs snake_case) or unclear identifiers.",
    "Inconsistent Style": "Mixed formatting, indentation, or code style conventions.",
    "Deep Nesting": "Nesting depth greater than 3-4 levels (if/else/for/while/try).",
    "Switch Statements": "Large switch/if-elif chains over type codes (replace with polymorphism).",
    "Flag Arguments": "Function parameters that are boolean flags controlling behavior paths.",
    "Empty Catch Block": "Exception handlers that swallow errors without logging/rethrowing.",
    "Generic Exceptions": "Catching broad exceptions like 'Exception' or 'Throwable' instead of specific types.",
    "Hidden Errors": "Exceptions or errors that are silently ignored or swallowed without visible handling.",
    "High Cyclomatic Complexity": "Method with >10-15 decision points (if/else/for/while/case).",
    "Missing Error Handling": "Risky operations (I/O, network, parsing) without try/except.",
    "Temporary Field": "Instance field only used under certain conditions.",
    "Alternative Classes with Different Interfaces": "Two classes doing the same thing with different APIs.",
}


def build_prompt_catalog_block() -> str:
    """Return a compact, grouped catalog block suitable for embedding in LLM prompts."""
    lines: List[str] = []
    for group, smells in SMELL_CATALOG.items():
        lines.append(f"[{group}]")
        for smell in smells:
            definition = SMELL_DEFINITIONS.get(smell, "")
            lines.append(f"  - {smell}: {definition}")
    return "\n".join(lines)
