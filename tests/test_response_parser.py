"""
Unit tests for src.llm.response_parser.ResponseParser.

L5: Lock down the alias-handling contract so silent regressions (LLMs start
emitting `type` instead of `smell_type`, dict locations, plural keys, bare
lists, etc.) can't pass CI undetected. These are hot paths that have
repeatedly broken in the past — see /memories/repo/runtime-fixes.md.

Run with: pytest tests/test_response_parser.py -q
"""

from __future__ import annotations

import json

import pytest

from src.llm.response_parser import AnalysisResult, ResponseParser, Severity


@pytest.fixture
def parser() -> ResponseParser:
    return ResponseParser(strict_mode=False)


# ---------------------------------------------------------------------------
# Object-shaped responses
# ---------------------------------------------------------------------------


def test_parse_object_with_canonical_keys(parser: ResponseParser) -> None:
    payload = {
        "code_smells": [
            {
                "smell_type": "God Class",
                "location": "line 10",
                "severity": "HIGH",
                "explanation": "Too many responsibilities",
            }
        ],
        "summary": "Found 1 smell",
        "is_valid_code": False,
    }
    result = parser.parse(json.dumps(payload))

    assert isinstance(result, AnalysisResult)
    assert len(result.code_smells) == 1
    smell = result.code_smells[0]
    assert smell.type == "God Class"
    assert smell.location == "line 10"
    assert smell.severity == Severity.HIGH
    assert result.is_valid_code is False


def test_parse_object_with_legacy_type_alias(parser: ResponseParser) -> None:
    """LLMs sometimes emit `type` instead of `smell_type`."""
    payload = {
        "code_smells": [
            {
                "type": "Long Method",
                "location": "line 5",
                "severity": "MEDIUM",
                "explanation": "Method body exceeds 100 lines",
            }
        ]
    }
    result = parser.parse(json.dumps(payload))
    assert [s.type for s in result.code_smells] == ["Long Method"]


def test_parse_object_with_dict_location(parser: ResponseParser) -> None:
    """Location can arrive as a {line, end_line} dict and must render to a string."""
    payload = {
        "code_smells": [
            {
                "smell_type": "Long Method",
                "location": {"line": 10, "end_line": 42},
                "severity": "HIGH",
                "explanation": "Spans 32 lines",
            },
            {
                "smell_type": "Long Method",
                "location": {"start_line": 7},
                "severity": "LOW",
                "explanation": "Starts at 7",
            },
        ]
    }
    result = parser.parse(json.dumps(payload))
    locations = [s.location for s in result.code_smells]
    assert locations == ["line 10-42", "line 7"]


def test_parse_object_with_findings_alias_key(parser: ResponseParser) -> None:
    """Accept `findings` or `smells` as alternate container keys."""
    payload = {
        "findings": [
            {
                "smell_type": "Duplicated Code",
                "location": "line 1",
                "severity": "LOW",
                "explanation": "Copy-pasted block",
            }
        ]
    }
    result = parser.parse(json.dumps(payload))
    assert len(result.code_smells) == 1
    # `Duplicated Code` is normalized to the canonical `Duplicate Code`.
    assert result.code_smells[0].type == "Duplicate Code"


# ---------------------------------------------------------------------------
# Bare-list responses (M5 contract)
# ---------------------------------------------------------------------------


def test_parse_bare_list_with_valid_entries(parser: ResponseParser) -> None:
    payload = [
        {
            "type": "God Class",
            "location": "line 1",
            "severity": "HIGH",
            "explanation": "Too many methods",
        }
    ]
    result = parser.parse(json.dumps(payload))
    assert len(result.code_smells) == 1
    # Bare list had findings -> is_valid_code resolves to False.
    assert result.is_valid_code is False


def test_parse_bare_empty_list_sets_valid_code_true(parser: ResponseParser) -> None:
    """Empty bare list means no smells -> is_valid_code True."""
    result = parser.parse(json.dumps([]))
    assert result.code_smells == []
    assert result.is_valid_code is True


# ---------------------------------------------------------------------------
# Malformed-finding filter (H2 contract)
# ---------------------------------------------------------------------------


def test_drops_findings_missing_required_fields(parser: ResponseParser) -> None:
    payload = {
        "code_smells": [
            # Missing severity AND location -> dropped
            {"smell_type": "God Class", "explanation": "bad"},
            # Missing smell_type/type/name -> dropped
            {"location": "line 3", "severity": "LOW", "explanation": "x"},
            # Valid (uses `line` alias for location, `type` alias for smell_type)
            {
                "type": "Long Method",
                "line": 9,
                "severity": "MEDIUM",
                "explanation": "Too long",
            },
        ]
    }
    result = parser.parse(json.dumps(payload))
    assert len(result.code_smells) == 1
    assert result.code_smells[0].type == "Long Method"


def test_drops_non_dict_entries(parser: ResponseParser) -> None:
    payload = {
        "code_smells": [
            "a string that slipped through",
            None,
            {
                "smell_type": "God Class",
                "location": "line 1",
                "severity": "HIGH",
                "explanation": "ok",
            },
        ]
    }
    result = parser.parse(json.dumps(payload))
    assert len(result.code_smells) == 1


# ---------------------------------------------------------------------------
# Refactoring / explanation aliases
# ---------------------------------------------------------------------------


def test_refactoring_alias_keys_resolved(parser: ResponseParser) -> None:
    payload = {
        "code_smells": [
            {
                "smell_type": "Long Method",
                "location": "line 5",
                "severity": "MEDIUM",
                "description": "Method too long",  # alias for explanation
                "suggestion": "Extract helpers",    # alias for refactoring
            }
        ]
    }
    result = parser.parse(json.dumps(payload))
    smell = result.code_smells[0]
    assert smell.explanation == "Method too long"
    assert smell.refactoring == "Extract helpers"


# ---------------------------------------------------------------------------
# Malformed JSON repair
# ---------------------------------------------------------------------------


def test_repair_trailing_text_around_json(parser: ResponseParser) -> None:
    raw = (
        "Here is the analysis:\n"
        '{"code_smells": [{"smell_type": "God Class", "location": "line 1", '
        '"severity": "HIGH", "explanation": "ok"}]}\n'
        "Hope that helps!"
    )
    result = parser.parse(raw)
    assert len(result.code_smells) == 1
    assert result.code_smells[0].type == "God Class"
