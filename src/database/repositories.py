"""
Repository classes for splitting DatabaseManager God Class responsibilities.

Each repository handles a specific domain:
- AgentRepository: Agent registration and retrieval
- RequestRepository: LLM request/response logging
- ActionRepository: Agent action tracking
- ExperimentRepository: Experiment metadata and runs
- AnalysisRepository: Analysis findings
- GroundTruthRepository: Ground truth data management
"""

import logging
from typing import List, Optional
from sqlalchemy.orm import scoped_session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class AgentRepository:
    """Repository for agent operations."""

    def __init__(self, session_maker: scoped_session):
        """Initialize with SQLAlchemy session maker."""
        self.session_maker = session_maker

    def add_agent(self, agent_id: str, name: str, role: str, system_prompt: str,
                  framework: str = "LangChain") -> bool:
        """Register a new agent."""
        try:
            from src.database.database_manager import Agent

            session = self.session_maker()
            agent = Agent(
                agent_id=agent_id,
                name=name,
                role=role,
                system_prompt=system_prompt,
                framework=framework
            )
            session.add(agent)
            session.commit()
            logger.info("Agent registered: %s (%s)", name, agent_id)
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:
            session.rollback()
            logger.error("Error adding agent: %s", e)
            return False

    def get_agent(self, agent_id: str):
        """Retrieve agent by ID."""
        from src.database.database_manager import Agent

        session = self.session_maker()
        return session.query(Agent).filter(Agent.agent_id == agent_id).first()

    def get_all_agents(self) -> List:
        """Get all registered agents."""
        from src.database.database_manager import Agent

        session = self.session_maker()
        return session.query(Agent).order_by(Agent.created_at.desc()).all()


class RequestRepository:
    """Repository for LLM request/response logging."""

    def __init__(self, session_maker: scoped_session):
        """Initialize with SQLAlchemy session maker."""
        self.session_maker = session_maker

    def add_request(self, request_id: str, agent_id: str, user_prompt: str,
                   model_used: str) -> bool:
        """Log an LLM request from agent."""
        try:
            from src.database.database_manager import AgentRequest

            session = self.session_maker()
            request = AgentRequest(
                request_id=request_id,
                agent_id=agent_id,
                user_prompt=user_prompt,
                model_used=model_used
            )
            session.add(request)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:
            session.rollback()
            logger.error("Error adding request: %s", e)
            return False

    def add_response(self, response_id: str, request_id: str, response_text: str,
                    tokens_used: Optional[int] = None, latency: Optional[float] = None) -> bool:
        """Log LLM response for a request."""
        try:
            from src.database.database_manager import AgentResponse

            session = self.session_maker()
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
        except (SQLAlchemyError, ValueError, TypeError) as e:
            session.rollback()
            logger.error("Error adding response: %s", e)
            return False

    def get_requests_by_agent(self, agent_id: str, limit: int = 100) -> List:
        """Get all requests from an agent."""
        from src.database.database_manager import AgentRequest

        session = self.session_maker()
        return session.query(AgentRequest).filter(
            AgentRequest.agent_id == agent_id
        ).order_by(AgentRequest.timestamp.desc()).limit(limit).all()

    def get_responses_by_request(self, request_id: str):
        """Get response for a specific request."""
        from src.database.database_manager import AgentResponse

        session = self.session_maker()
        return session.query(AgentResponse).filter(
            AgentResponse.request_id == request_id
        ).first()


class ActionRepository:
    """Repository for agent action tracking."""

    def __init__(self, session_maker: scoped_session):
        """Initialize with SQLAlchemy session maker."""
        self.session_maker = session_maker

    def add_action(self, action_id: str, agent_id: str, action_type: str,
                  action_content: str, status: str = "success") -> bool:
        """Log an agent action."""
        try:
            from src.database.database_manager import AgentAction

            session = self.session_maker()
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
        except (SQLAlchemyError, ValueError, TypeError) as e:
            session.rollback()
            logger.error("Error adding action: %s", e)
            return False

    def get_actions_by_agent(self, agent_id: str, limit: int = 100) -> List:
        """Get all actions from an agent."""
        from src.database.database_manager import AgentAction

        session = self.session_maker()
        return session.query(AgentAction).filter(
            AgentAction.agent_id == agent_id
        ).order_by(AgentAction.timestamp.desc()).limit(limit).all()


class ProcessRepository:
    """Repository for process step logging."""

    def __init__(self, session_maker: scoped_session):
        """Initialize with SQLAlchemy session maker."""
        self.session_maker = session_maker

    def add_process(self, process_id: str, process_type: str, agent_id: str,
                   task_id: Optional[str] = None, duration: Optional[float] = None) -> bool:
        """Log a process step."""
        try:
            from src.database.database_manager import Process

            session = self.session_maker()
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
        except (SQLAlchemyError, ValueError, TypeError) as e:
            session.rollback()
            logger.error("Error adding process: %s", e)
            return False


class ExperimentRepository:
    """Repository for experiment metadata and runs."""

    def __init__(self, session_maker: scoped_session):
        """Initialize with SQLAlchemy session maker."""
        self.session_maker = session_maker

    def add_experiment(self, exp_id: str, name: str, config: dict) -> bool:
        """Create a new experiment."""
        try:
            from src.database.database_manager import Experiment

            session = self.session_maker()
            exp = Experiment(exp_id=exp_id, name=name, config=config)
            session.add(exp)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:
            session.rollback()
            logger.error("Error adding experiment: %s", e)
            return False

    def add_analysis_run(self, run_id: str, exp_id: str, code_snippet: str,
                        language: str) -> bool:
        """Log an analysis run."""
        try:
            from src.database.database_manager import AnalysisRun

            session = self.session_maker()
            run = AnalysisRun(
                run_id=run_id,
                exp_id=exp_id,
                code_snippet=code_snippet,
                language=language
            )
            session.add(run)
            session.commit()
            return True
        except (SQLAlchemyError, ValueError, TypeError) as e:
            session.rollback()
            logger.error("Error adding analysis run: %s", e)
            return False

    def get_experiment(self, exp_id: str):
        """Retrieve experiment by ID."""
        from src.database.database_manager import Experiment

        session = self.session_maker()
        return session.query(Experiment).filter(Experiment.exp_id == exp_id).first()

    def get_analysis_runs(self, exp_id: str, limit: int = 100) -> List:
        """Get all runs for an experiment."""
        from src.database.database_manager import AnalysisRun

        session = self.session_maker()
        return session.query(AnalysisRun).filter(
            AnalysisRun.exp_id == exp_id
        ).order_by(AnalysisRun.created_at.desc()).limit(limit).all()


class AnalysisRepository:
    """Repository for analysis findings."""

    def __init__(self, session_maker: scoped_session):
        """Initialize with SQLAlchemy session maker."""
        self.session_maker = session_maker

    def add_finding(self, finding_id: str, run_id: str, smell_type: str, severity: str,
                   confidence: float, agent_id: str, explanation: str = None) -> bool:
        """Log a code smell finding."""
        try:
            from src.database.database_manager import CodeSmellFinding

            session = self.session_maker()
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
        except (SQLAlchemyError, ValueError, TypeError) as e:
            session.rollback()
            logger.error("Error adding finding: %s", e)
            return False

    def get_findings_by_run(self, run_id: str) -> List:
        """Get all findings for an analysis run."""
        from src.database.database_manager import CodeSmellFinding

        session = self.session_maker()
        return session.query(CodeSmellFinding).filter(
            CodeSmellFinding.run_id == run_id
        ).order_by(CodeSmellFinding.created_at.desc()).all()

    def get_findings_by_type(self, smell_type: str, limit: int = 100) -> List:
        """Get findings by smell type."""
        from src.database.database_manager import CodeSmellFinding

        session = self.session_maker()
        return session.query(CodeSmellFinding).filter(
            CodeSmellFinding.smell_type == smell_type
        ).order_by(CodeSmellFinding.created_at.desc()).limit(limit).all()


class GroundTruthRepository:
    """Repository for ground truth data management."""

    def __init__(self, session_maker: scoped_session):
        """Initialize with SQLAlchemy session maker."""
        self.session_maker = session_maker

    def add_ground_truth(self, gt_id: str, code_snippet: str, smell_labels: dict,
                        source: str, language: str) -> bool:
        """Add ground truth labeled data."""
        try:
            from src.database.database_manager import GroundTruth

            session = self.session_maker()
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
        except (SQLAlchemyError, ValueError, TypeError) as e:
            session.rollback()
            logger.error("Error adding ground truth: %s", e)
            return False

    def get_ground_truth(self, gt_id: str):
        """Retrieve ground truth by ID."""
        from src.database.database_manager import GroundTruth

        session = self.session_maker()
        return session.query(GroundTruth).filter(GroundTruth.gt_id == gt_id).first()

    def get_ground_truth_by_language(self, language: str, limit: int = 100) -> List:
        """Get ground truth samples by language."""
        from src.database.database_manager import GroundTruth

        session = self.session_maker()
        return session.query(GroundTruth).filter(
            GroundTruth.language == language
        ).order_by(GroundTruth.created_at.desc()).limit(limit).all()
