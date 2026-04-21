"""
Pydantic models for FastAPI endpoints.
Request and response schemas for code smell detection API.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class LanguageEnum(str, Enum):
    """Supported programming languages."""
    JAVA = "java"
    PYTHON = "python"
    JAVASCRIPT = "javascript"


# ============================================================================
# Request Models
# ============================================================================

class CodeSubmissionRequest(BaseModel):
    """Request to submit code for analysis."""
    code: str = Field(..., description="Code snippet to analyze (1KB - 100KB)", min_length=10, max_length=100000)
    language: Optional[LanguageEnum] = Field(None, description="Programming language (auto-detected if None)")
    file_name: Optional[str] = Field("code_snippet", description="Source file name for context")
    include_rag: bool = Field(True, description="Enable RAG context retrieval")
    model: Optional[str] = Field(None, description="Optional LLM model from Ollama (if None, agent auto-selects best model)")
    timeout_seconds: int = Field(300, description="Analysis timeout in seconds", ge=30, le=600)

    class Config:
        json_schema_extra = {
            "example": {
                "code": "public class VeryLongMethod {\n    public void doSomething() { ... }\n}",
                "language": "java",
                "file_name": "VeryLongMethod.java",
                "include_rag": True,
                "model": "llama3:8b",
                "timeout_seconds": 300
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class CodeSmellLocation(BaseModel):
    """Location of detected code smell."""
    line: int = Field(..., description="Start line number (1-indexed)")
    column: int = Field(0, description="Start column number (0-indexed)")
    end_line: Optional[int] = Field(None, description="End line number")
    end_column: Optional[int] = Field(None, description="End column number")


class CodeSmellFindingResponse(BaseModel):
    """A single code smell finding."""
    smell_type: str = Field(..., description="Type of smell (e.g., 'Long Method', 'God Class')")
    location: CodeSmellLocation = Field(..., description="Location in source code")
    severity: str = Field(..., description="Severity level: low, medium, high, critical")
    confidence: float = Field(..., description="Confidence score (0.0 - 1.0)", ge=0.0, le=1.0)
    explanation: str = Field(..., description="Why this smell was detected")
    rag_context: Optional[List[str]] = Field(None, description="Retrieved similar code examples")
    suggested_refactoring: Optional[str] = Field(None, description="Suggested refactoring approach")


class CodeMetricsResponse(BaseModel):
    """Code metrics extracted during analysis."""
    lines_of_code: int = Field(..., description="Total lines of code")
    cyclomatic_complexity: Optional[float] = Field(None, description="Cyclomatic complexity")
    halstead_complexity: Optional[float] = Field(None, description="Halstead complexity")
    maintainability_index: Optional[float] = Field(None, description="Maintainability index")


class AnalysisResultResponse(BaseModel):
    """Response containing analysis results."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    code_hash: str = Field(..., description="SHA256 hash of analyzed code")
    language: str = Field(..., description="Detected programming language")
    findings: List[CodeSmellFindingResponse] = Field(default_factory=list, description="Detected code smells")
    metrics: CodeMetricsResponse = Field(..., description="Code metrics")
    analysis_time_ms: float = Field(..., description="Total analysis time in milliseconds", ge=0)
    model_used: str = Field(..., description="LLM model used (llama3, codelama, mistral)")
    model_reasoning: Optional[str] = Field(None, description="Agent reasoning for model selection (if auto-selected)")
    cache_hit: bool = Field(False, description="Whether result was from cache")
    completed_at: datetime = Field(..., description="Completion timestamp")
    f1_score: Optional[float] = Field(None, description="F1 score against ground truth (0.0 - 1.0)", ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, description="Precision score (0.0 - 1.0)", ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, description="Recall score (0.0 - 1.0)", ge=0.0, le=1.0)
    ground_truth_count: Optional[int] = Field(None, description="Number of ground truth findings")


class CodeSubmissionResponse(BaseModel):
    """Response to code submission."""
    analysis_id: str = Field(..., description="Unique analysis identifier for tracking")
    status: str = Field(..., description="Current status: queued, processing, completed, failed")
    created_at: datetime = Field(..., description="When submission was created")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    message: str = Field("", description="Status message")


class ProgressResponse(BaseModel):
    """Real-time progress of analysis."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: str = Field(..., description="Current step: parsing, rag_retrieval, inference, validation")
    percentage: int = Field(..., description="Progress percentage (0-100)", ge=0, le=100)
    current_step: str = Field(..., description="Human-readable current step description")
    estimated_remaining_ms: Optional[int] = Field(None, description="Estimated remaining time in milliseconds")


class BaselineToolResult(BaseModel):
    """Results from a single baseline tool."""
    tool_name: str = Field(..., description="Tool name: SonarQube, PMD, Checkstyle, SpotBugs, IntelliJ")
    findings_count: int = Field(..., description="Number of findings")
    analysis_time_ms: float = Field(..., description="Analysis time in milliseconds")
    findings: List[CodeSmellFindingResponse] = Field(default_factory=list, description="Individual findings")


class ComparisonMetrics(BaseModel):
    """Metrics comparing LLM vs baseline tools."""
    precision: float = Field(..., description="Precision score (0.0 - 1.0)", ge=0.0, le=1.0)
    recall: float = Field(..., description="Recall score (0.0 - 1.0)", ge=0.0, le=1.0)
    f1_score: float = Field(..., description="F1 score (0.0 - 1.0)", ge=0.0, le=1.0)
    accuracy: float = Field(..., description="Accuracy score (0.0 - 1.0)", ge=0.0, le=1.0)
    true_positives: int = Field(..., description="Number of true positives")
    false_positives: int = Field(..., description="Number of false positives")
    false_negatives: int = Field(..., description="Number of false negatives")


class ComparisonResponse(BaseModel):
    """Response comparing LLM results with baseline tools."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    llm_results: AnalysisResultResponse = Field(..., description="LLM analysis results")
    baseline_results: Dict[str, BaselineToolResult] = Field(default_factory=dict, description="Results from baseline tools")
    metrics: ComparisonMetrics = Field(..., description="Comparison metrics")
    summary: str = Field("", description="Summary of comparison insights")


# ============================================================================
# Health Check Models
# ============================================================================

class ServiceStatus(BaseModel):
    """Status of a service component."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Status: healthy, degraded, unhealthy")
    message: str = Field("", description="Status message")


class HealthCheckResponse(BaseModel):
    """API health check response."""
    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, ServiceStatus] = Field(default_factory=dict, description="Individual service statuses")
    version: str = Field(..., description="API version")


class SystemStatusResponse(BaseModel):
    """Detailed system status."""
    status: str = Field(..., description="Overall status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    active_analyses: int = Field(..., description="Number of active analyses")
    completed_analyses: int = Field(..., description="Number of completed analyses")
    cache_size_mb: float = Field(..., description="Cache size in MB")
    services: Dict[str, ServiceStatus] = Field(default_factory=dict, description="Service statuses")


# ============================================================================
# Error Response Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail."""
    error_code: str = Field(..., description="Error code for categorization")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: ErrorDetail = Field(..., description="Error information")
    timestamp: datetime = Field(..., description="When error occurred")
    request_id: str = Field(..., description="Request ID for tracking")
