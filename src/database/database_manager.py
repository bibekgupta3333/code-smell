"""
Database Manager for Multi-Agent Code Smell Detection System
SQLAlchemy ORM-based database management for experiment tracking and agent monitoring.

Architecture:
  - SQLAlchemy 2.0+ ORM for type-safe database operations
  - SQLite for lightweight experiment tracking and reproducibility
  - Optimized for M4 Pro with scoped sessions and efficient queries
  - Supports agent performance analytics and ablation studies
  - Export capabilities for paper results and analysis

Models:
  - Agent: Agent metadata and configuration
  - AgentRequest: LLM request logging
  - AgentResponse: LLM response logging
  - AgentAction: Agent action logging
  - Process: Workflow step tracking
  - Experiment: Experiment metadata
  - AnalysisRun: Individual code analysis executions
  - CodeSmellFinding: Detected code smells per analysis
  - GroundTruth: Labeled data for evaluation

Performance:
  - Scoped sessions for thread-local database connections
  - Lazy loading and relationship management
  - Indexed queries for fast retrieval
  - Efficient JSON serialization for config storage
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Text, DateTime,
    ForeignKey, Index, func, JSON
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship, scoped_session
)
from sqlalchemy.pool import QueuePool
import pandas as pd

from config import PROJECT_ROOT, RESULTS_DIR

logger = logging.getLogger(__name__)

# SQLAlchemy base class for ORM models
Base = declarative_base()


# ============================================================================
# SEVERITY LEVEL ENUM
# ============================================================================

class SeverityLevel(str, Enum):
    """Code smell severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ============================================================================
# SQLALCHEMY ORM MODELS
# ============================================================================

class Agent(Base):
    """Agent metadata and configuration model."""
    __tablename__ = "agents"

    agent_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    system_prompt = Column(Text)
    framework = Column(String, default="LangChain")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    requests = relationship("AgentRequest", back_populates="agent", cascade="all, delete-orphan")
    actions = relationship("AgentAction", back_populates="agent", cascade="all, delete-orphan")
    processes = relationship("Process", back_populates="agent", cascade="all, delete-orphan")
    findings = relationship("CodeSmellFinding", back_populates="agent", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Agent(agent_id={self.agent_id}, name={self.name})>"


class AgentRequest(Base):
    """LLM request logging model."""
    __tablename__ = "agent_requests"

    request_id = Column(String, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"), nullable=False, index=True)
    user_prompt = Column(Text, nullable=False)
    model_used = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    agent = relationship("Agent", back_populates="requests")
    response = relationship("AgentResponse", back_populates="request", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (Index("idx_agent_requests_agent", "agent_id"),)

    def __repr__(self):
        return f"<AgentRequest(request_id={self.request_id}, agent_id={self.agent_id})>"


class AgentResponse(Base):
    """LLM response logging model with metrics."""
    __tablename__ = "agent_responses"

    response_id = Column(String, primary_key=True, index=True)
    request_id = Column(String, ForeignKey("agent_requests.request_id"), nullable=False, index=True)
    response_text = Column(Text, nullable=False)
    tokens_used = Column(Integer)
    latency = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    request = relationship("AgentRequest", back_populates="response")

    __table_args__ = (Index("idx_agent_responses_request", "request_id"),)

    def __repr__(self):
        return f"<AgentResponse(response_id={self.response_id}, tokens={self.tokens_used})>"


class AgentAction(Base):
    """Agent action execution logging model."""
    __tablename__ = "agent_actions"

    action_id = Column(String, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"), nullable=False, index=True)
    action_type = Column(String, nullable=False)
    action_content = Column(Text, nullable=False)
    status = Column(String, default="success")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    agent = relationship("Agent", back_populates="actions")

    __table_args__ = (Index("idx_agent_actions_agent", "agent_id"),)

    def __repr__(self):
        return f"<AgentAction(action_id={self.action_id}, type={self.action_type})>"


class Process(Base):
    """Workflow process step logging model."""
    __tablename__ = "processes"

    process_id = Column(String, primary_key=True, index=True)
    process_type = Column(String, nullable=False)
    agent_id = Column(String, ForeignKey("agents.agent_id"))
    task_id = Column(String)
    duration = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    agent = relationship("Agent", back_populates="processes")

    __table_args__ = (Index("idx_processes_agent", "agent_id"),)

    def __repr__(self):
        return f"<Process(process_id={self.process_id}, type={self.process_type})>"


class Experiment(Base):
    """Experiment metadata model."""
    __tablename__ = "experiments"

    exp_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    config = Column(JSON)
    status = Column(String, default="running")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)

    # Relationships
    runs = relationship("AnalysisRun", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Experiment(exp_id={self.exp_id}, name={self.name})>"


class AnalysisRun(Base):
    """Individual code analysis execution model."""
    __tablename__ = "analysis_runs"

    run_id = Column(String, primary_key=True, index=True)
    exp_id = Column(String, ForeignKey("experiments.exp_id"), nullable=False, index=True)
    code_snippet = Column(Text, nullable=False)
    language = Column(String)
    result = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    experiment = relationship("Experiment", back_populates="runs")
    findings = relationship("CodeSmellFinding", back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (Index("idx_analysis_runs_exp", "exp_id"),)

    def __repr__(self):
        return f"<AnalysisRun(run_id={self.run_id}, exp_id={self.exp_id})>"


class CodeSmellFinding(Base):
    """Code smell detection finding model."""
    __tablename__ = "code_smell_findings"

    finding_id = Column(String, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("analysis_runs.run_id"), nullable=False, index=True)
    smell_type = Column(String, nullable=False, index=True)
    severity = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    agent_id = Column(String, ForeignKey("agents.agent_id"), nullable=False, index=True)
    explanation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    run = relationship("AnalysisRun", back_populates="findings")
    agent = relationship("Agent", back_populates="findings")

    __table_args__ = (
        Index("idx_findings_run", "run_id"),
        Index("idx_findings_agent", "agent_id"),
        Index("idx_findings_smell", "smell_type"),
    )

    def __repr__(self):
        return f"<CodeSmellFinding(finding_id={self.finding_id}, type={self.smell_type})>"


class GroundTruth(Base):
    """Labeled ground truth data for evaluation model."""
    __tablename__ = "ground_truth"

    gt_id = Column(String, primary_key=True, index=True)
    code_snippet = Column(Text, nullable=False)
    smell_labels = Column(JSON)
    source = Column(String, nullable=False)
    language = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<GroundTruth(gt_id={self.gt_id}, source={self.source})>"


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """SQLAlchemy-based database manager for multi-agent system tracking."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database. Defaults to PROJECT_ROOT/cache/system.db
        """
        if db_path is None:
            db_path = PROJECT_ROOT / "cache" / "system.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # SQLAlchemy engine with connection pooling optimized for M4 Pro
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            db_url,
            connect_args={"timeout": 30.0, "check_same_thread": False},
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            echo=False,
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Scoped session for thread-local connections
        Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.session_maker = scoped_session(Session)

        logger.info("Database initialized at %s", self.db_path)  # noqa: G201

    def get_session(self):
        """Get a scoped session for database operations."""
        return self.session_maker

    # ========================================================================
    # AGENT OPERATIONS
    # ========================================================================

    def add_agent(self, agent_id: str, name: str, role: str, system_prompt: str,
                  framework: str = "LangChain") -> bool:
        """Register a new agent."""
        try:
            session = self.get_session()
            agent = Agent(
                agent_id=agent_id,
                name=name,
                role=role,
                system_prompt=system_prompt,
                framework=framework
            )
            session.add(agent)
            session.commit()
            logger.info("Agent registered: %s (%s)", name, agent_id)  # noqa: G201
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Error adding agent: %s", e)  # noqa: G201
            return False

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Retrieve agent by ID."""
        session = self.get_session()
        return session.query(Agent).filter(Agent.agent_id == agent_id).first()

    def get_all_agents(self) -> List[Agent]:
        """Get all registered agents."""
        session = self.get_session()
        return session.query(Agent).order_by(Agent.created_at.desc()).all()

    # ========================================================================
    # LLM REQUEST/RESPONSE LOGGING
    # ========================================================================

    def add_request(self, request_id: str, agent_id: str, user_prompt: str,
                   model_used: str) -> bool:
        """Log an LLM request from agent."""
        try:
            session = self.get_session()
            request = AgentRequest(
                request_id=request_id,
                agent_id=agent_id,
                user_prompt=user_prompt,
                model_used=model_used
            )
            session.add(request)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Error adding request: %s", e)  # noqa: G201
            return False

    def add_response(self, response_id: str, request_id: str, response_text: str,
                    tokens_used: int, latency: float) -> bool:
        """Log an LLM response."""
        try:
            session = self.get_session()
            response = AgentResponse(
                response_id=response_id,
                request_id=request_id,
                response_text=response_text,
                tokens_used=tokens_used,
                latency=latency
            )
            session.add(response)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Error adding response: %s", e)  # noqa: G201
            return False

    def get_requests_by_agent(self, agent_id: str, limit: int = 100) -> List[AgentRequest]:
        """Get recent requests from an agent."""
        session = self.get_session()
        return session.query(AgentRequest).filter(
            AgentRequest.agent_id == agent_id
        ).order_by(AgentRequest.timestamp.desc()).limit(limit).all()

    def get_responses_by_request(self, request_id: str) -> Optional[AgentResponse]:
        """Get response for a specific request."""
        session = self.get_session()
        return session.query(AgentResponse).filter(
            AgentResponse.request_id == request_id
        ).first()

    # ========================================================================
    # AGENT ACTIONS & PROCESSES
    # ========================================================================

    def add_action(self, action_id: str, agent_id: str, action_type: str,
                  action_content: str, status: str = "success") -> bool:
        """Log an agent action."""
        try:
            session = self.get_session()
            action = AgentAction(
                action_id=action_id,
                agent_id=agent_id,
                action_type=action_type,
                action_content=action_content,
                status=status
            )
            session.add(action)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Error adding action: %s", e)  # noqa: G201
            return False

    def add_process(self, process_id: str, process_type: str, agent_id: str,
                   task_id: str, duration: Optional[float] = None) -> bool:
        """Log a workflow process step."""
        try:
            session = self.get_session()
            process = Process(
                process_id=process_id,
                process_type=process_type,
                agent_id=agent_id,
                task_id=task_id,
                duration=duration
            )
            session.add(process)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Error adding process: %s", e)  # noqa: G201
            return False

    def get_actions_by_agent(self, agent_id: str, limit: int = 100) -> List[AgentAction]:
        """Get recent actions by an agent."""
        session = self.get_session()
        return session.query(AgentAction).filter(
            AgentAction.agent_id == agent_id
        ).order_by(AgentAction.timestamp.desc()).limit(limit).all()

    # ========================================================================
    # EXPERIMENT TRACKING
    # ========================================================================

    def add_experiment(self, exp_id: str, name: str, config: Dict[str, Any]) -> bool:
        """Create a new experiment."""
        try:
            session = self.get_session()
            experiment = Experiment(
                exp_id=exp_id,
                name=name,
                config=config
            )
            session.add(experiment)
            session.commit()
            logger.info("Experiment created: %s (%s)", name, exp_id)  # noqa: G201
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Error adding experiment: %s", e)  # noqa: G201
            return False

    def add_analysis_run(self, run_id: str, exp_id: str, code_snippet: str,
                        language: str, result: Optional[str] = None) -> bool:
        """Log a code analysis execution."""
        try:
            session = self.get_session()
            run = AnalysisRun(
                run_id=run_id,
                exp_id=exp_id,
                code_snippet=code_snippet,
                language=language,
                result=result
            )
            session.add(run)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Error adding analysis run: %s", e)  # noqa: G201
            return False

    def add_finding(self, finding_id: str, run_id: str, smell_type: str,
                   severity: str, confidence: float, agent_id: str,
                   explanation: Optional[str] = None) -> bool:
        """Log a detected code smell."""
        try:
            session = self.get_session()
            finding = CodeSmellFinding(
                finding_id=finding_id,
                run_id=run_id,
                smell_type=smell_type,
                severity=severity,
                confidence=confidence,
                agent_id=agent_id,
                explanation=explanation
            )
            session.add(finding)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Error adding finding: %s", e)  # noqa: G201
            return False

    def get_findings_by_run(self, run_id: str) -> List[CodeSmellFinding]:
        """Get all findings for an analysis run."""
        session = self.get_session()
        return session.query(CodeSmellFinding).filter(
            CodeSmellFinding.run_id == run_id
        ).order_by(CodeSmellFinding.confidence.desc()).all()

    # ========================================================================
    # GROUND TRUTH & EVALUATION
    # ========================================================================

    def add_ground_truth(self, gt_id: str, code_snippet: str, smell_labels: Dict[str, Any],
                        source: str, language: str) -> bool:
        """Add labeled ground truth data."""
        try:
            session = self.get_session()
            gt = GroundTruth(
                gt_id=gt_id,
                code_snippet=code_snippet,
                smell_labels=smell_labels,
                source=source,
                language=language
            )
            session.add(gt)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Error adding ground truth: %s", e)  # noqa: G201
            return False

    def get_ground_truth(self, limit: int = 100, source: Optional[str] = None) -> List[GroundTruth]:
        """Get ground truth samples."""
        session = self.get_session()
        query = session.query(GroundTruth)
        if source:
            query = query.filter(GroundTruth.source == source)
        return query.limit(limit).all()

    # ========================================================================
    # STATISTICS & ANALYTICS
    # ========================================================================

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an agent."""
        session = self.get_session()

        agent = session.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            return {}

        total_requests = session.query(func.count(AgentRequest.request_id)).filter(  # pylint: disable=not-callable
            AgentRequest.agent_id == agent_id
        ).scalar() or 0

        response_stats = session.query(
            func.count(AgentResponse.response_id),  # pylint: disable=not-callable
            func.avg(AgentResponse.tokens_used),  # pylint: disable=not-callable
            func.avg(AgentResponse.latency),  # pylint: disable=not-callable
            func.min(AgentResponse.latency),  # pylint: disable=not-callable
            func.max(AgentResponse.latency)  # pylint: disable=not-callable
        ).join(AgentRequest, AgentResponse.request_id == AgentRequest.request_id).filter(
            AgentRequest.agent_id == agent_id
        ).first()

        action_stats = session.query(
            func.count(AgentAction.action_id),  # pylint: disable=not-callable
            func.sum(func.cast(AgentAction.status == "success", Integer))  # pylint: disable=not-callable
        ).filter(AgentAction.agent_id == agent_id).first()

        detection_stats = session.query(
            func.count(CodeSmellFinding.finding_id),  # pylint: disable=not-callable
            func.avg(CodeSmellFinding.confidence),  # pylint: disable=not-callable
            func.count(func.distinct(CodeSmellFinding.smell_type))  # pylint: disable=not-callable
        ).filter(CodeSmellFinding.agent_id == agent_id).first()

        return {
            'agent': {
                'agent_id': agent.agent_id,
                'name': agent.name,
                'role': agent.role,
                'framework': agent.framework
            },
            'requests': {
                'total': total_requests,
                'avg_tokens': float(response_stats[1] or 0) if response_stats else 0,
                'avg_latency': float(response_stats[2] or 0) if response_stats else 0,
                'min_latency': float(response_stats[3] or 0) if response_stats else 0,
                'max_latency': float(response_stats[4] or 0) if response_stats else 0,
            },
            'actions': {
                'total': action_stats[0] or 0 if action_stats else 0,
                'successful': action_stats[1] or 0 if action_stats else 0,
                'success_rate': (
                    (action_stats[1] or 0) / (action_stats[0] or 1) * 100
                ) if action_stats and action_stats[0] else 0,
            },
            'detections': {
                'total': detection_stats[0] or 0 if detection_stats else 0,
                'avg_confidence': float(detection_stats[1] or 0) if detection_stats else 0,
                'unique_smell_types': detection_stats[2] or 0 if detection_stats else 0,
            },
        }

    def export_results(self, exp_id: str, output_dir: Optional[Path] = None) -> bool:
        """Export experiment results to CSV and JSON files."""
        if output_dir is None:
            output_dir = RESULTS_DIR / "exports"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            session = self.get_session()

            runs = session.query(AnalysisRun).filter(
                AnalysisRun.exp_id == exp_id
            ).all()
            if runs:
                df_runs = pd.DataFrame([{
                    'run_id': r.run_id,
                    'exp_id': r.exp_id,
                    'language': r.language,
                    'created_at': r.created_at.isoformat()
                } for r in runs])
                df_runs.to_csv(output_dir / f"{exp_id}_analysis_runs.csv", index=False)
                df_runs.to_json(output_dir / f"{exp_id}_analysis_runs.json", orient='records', indent=2)

            findings = session.query(CodeSmellFinding).join(
                AnalysisRun, CodeSmellFinding.run_id == AnalysisRun.run_id
            ).filter(AnalysisRun.exp_id == exp_id).all()
            if findings:
                df_findings = pd.DataFrame([{
                    'finding_id': f.finding_id,
                    'run_id': f.run_id,
                    'smell_type': f.smell_type,
                    'severity': f.severity,
                    'confidence': f.confidence,
                    'agent_id': f.agent_id,
                    'created_at': f.created_at.isoformat()
                } for f in findings])
                df_findings.to_csv(output_dir / f"{exp_id}_findings.csv", index=False)
                df_findings.to_json(output_dir / f"{exp_id}_findings.json", orient='records', indent=2)

            agents = session.query(Agent).all()
            if agents:
                df_agents = pd.DataFrame([{
                    'agent_id': a.agent_id,
                    'name': a.name,
                    'role': a.role,
                    'framework': a.framework,
                    'created_at': a.created_at.isoformat()
                } for a in agents])
                df_agents.to_csv(output_dir / f"{exp_id}_agents.csv", index=False)

            logger.info("Results exported to %s", output_dir)  # noqa: G201
            return True
        except (SQLAlchemyError, ValueError, TypeError, IOError) as e:  # noqa: B014
            logger.error("Export failed: %s", e)  # noqa: G201
            return False

    # ========================================================================
    # DATABASE UTILITIES
    # ========================================================================

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        session = self.get_session()

        stats = {
            'agents': session.query(func.count(Agent.agent_id)).scalar() or 0,  # pylint: disable=not-callable
            'agent_requests': session.query(func.count(AgentRequest.request_id)).scalar() or 0,  # pylint: disable=not-callable
            'agent_responses': session.query(func.count(AgentResponse.response_id)).scalar() or 0,  # pylint: disable=not-callable
            'agent_actions': session.query(func.count(AgentAction.action_id)).scalar() or 0,  # pylint: disable=not-callable
            'processes': session.query(func.count(Process.process_id)).scalar() or 0,  # pylint: disable=not-callable
            'experiments': session.query(func.count(Experiment.exp_id)).scalar() or 0,  # pylint: disable=not-callable
            'analysis_runs': session.query(func.count(AnalysisRun.run_id)).scalar() or 0,  # pylint: disable=not-callable
            'code_smell_findings': session.query(func.count(CodeSmellFinding.finding_id)).scalar() or 0,  # pylint: disable=not-callable
            'ground_truth': session.query(func.count(GroundTruth.gt_id)).scalar() or 0,  # pylint: disable=not-callable
        }

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        stats['database_size_mb'] = db_size / (1024 * 1024)

        return stats

    def cleanup(self, days_old: int = 30) -> int:
        """Clean up old records (older than N days)."""
        try:
            session = self.get_session()
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            deleted_count = 0

            deleted_count += session.query(AgentRequest).filter(
                AgentRequest.timestamp < cutoff_date
            ).delete()
            deleted_count += session.query(AgentResponse).filter(
                AgentResponse.timestamp < cutoff_date
            ).delete()
            deleted_count += session.query(AgentAction).filter(
                AgentAction.timestamp < cutoff_date
            ).delete()
            deleted_count += session.query(Process).filter(
                Process.timestamp < cutoff_date
            ).delete()
            deleted_count += session.query(AnalysisRun).filter(
                AnalysisRun.created_at < cutoff_date
            ).delete()
            deleted_count += session.query(CodeSmellFinding).filter(
                CodeSmellFinding.created_at < cutoff_date
            ).delete()

            session.commit()
            logger.info("Cleaned up %d old records", deleted_count)  # noqa: G201
            return deleted_count
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            session.rollback()
            logger.error("Cleanup failed: %s", e)  # noqa: G201
            return 0

    def vacuum(self) -> bool:
        """Optimize database by removing unused space."""
        try:
            session = self.get_session()
            session.execute("VACUUM")
            session.commit()
            logger.info("Database vacuumed successfully")
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:  # noqa: B014
            logger.error("Vacuum failed: %s", e)  # noqa: G201
            return False

    def close(self):
        """Close database connection and cleanup resources."""
        self.session_maker.remove()


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_db_manager_instance = None


def get_database_manager() -> DatabaseManager:  # pragma: no cover
    """Get or create the singleton database manager instance."""
    global _db_manager_instance  # pylint: disable=global-statement
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
    return _db_manager_instance
