"""Initial database schema from SQLAlchemy ORM models

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-03-02 16:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all database tables from SQLAlchemy ORM models."""

    # Create agents table
    op.create_table('agents',
        sa.Column('agent_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('framework', sa.String(), nullable=True, server_default='LangChain'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('agent_id')
    )
    op.create_index(op.f('ix_agents_agent_id'), 'agents', ['agent_id'], unique=False)
    op.create_index(op.f('ix_agents_created_at'), 'agents', ['created_at'], unique=False)

    # Create agent_requests table
    op.create_table('agent_requests',
        sa.Column('request_id', sa.String(), nullable=False),
        sa.Column('agent_id', sa.String(), nullable=False),
        sa.Column('user_prompt', sa.Text(), nullable=False),
        sa.Column('model_used', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.agent_id'], ),
        sa.PrimaryKeyConstraint('request_id')
    )
    op.create_index(op.f('ix_agent_requests_agent_id'), 'agent_requests', ['agent_id'], unique=False)
    op.create_index(op.f('ix_agent_requests_request_id'), 'agent_requests', ['request_id'], unique=False)
    op.create_index(op.f('ix_agent_requests_timestamp'), 'agent_requests', ['timestamp'], unique=False)
    op.create_index('idx_agent_requests_agent', 'agent_requests', ['agent_id'], unique=False)

    # Create agent_responses table
    op.create_table('agent_responses',
        sa.Column('response_id', sa.String(), nullable=False),
        sa.Column('request_id', sa.String(), nullable=False),
        sa.Column('response_text', sa.Text(), nullable=False),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('latency', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['request_id'], ['agent_requests.request_id'], ),
        sa.PrimaryKeyConstraint('response_id')
    )
    op.create_index(op.f('ix_agent_responses_request_id'), 'agent_responses', ['request_id'], unique=False)
    op.create_index(op.f('ix_agent_responses_response_id'), 'agent_responses', ['response_id'], unique=False)
    op.create_index(op.f('ix_agent_responses_timestamp'), 'agent_responses', ['timestamp'], unique=False)
    op.create_index('idx_agent_responses_request', 'agent_responses', ['request_id'], unique=False)

    # Create agent_actions table
    op.create_table('agent_actions',
        sa.Column('action_id', sa.String(), nullable=False),
        sa.Column('agent_id', sa.String(), nullable=False),
        sa.Column('action_type', sa.String(), nullable=False),
        sa.Column('action_content', sa.Text(), nullable=False),
        sa.Column('status', sa.String(), nullable=True, server_default='success'),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.agent_id'], ),
        sa.PrimaryKeyConstraint('action_id')
    )
    op.create_index(op.f('ix_agent_actions_action_id'), 'agent_actions', ['action_id'], unique=False)
    op.create_index(op.f('ix_agent_actions_agent_id'), 'agent_actions', ['agent_id'], unique=False)
    op.create_index(op.f('ix_agent_actions_timestamp'), 'agent_actions', ['timestamp'], unique=False)
    op.create_index('idx_agent_actions_agent', 'agent_actions', ['agent_id'], unique=False)

    # Create processes table
    op.create_table('processes',
        sa.Column('process_id', sa.String(), nullable=False),
        sa.Column('process_type', sa.String(), nullable=False),
        sa.Column('agent_id', sa.String(), nullable=True),
        sa.Column('task_id', sa.String(), nullable=True),
        sa.Column('duration', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.agent_id'], ),
        sa.PrimaryKeyConstraint('process_id')
    )
    op.create_index(op.f('ix_processes_agent_id'), 'processes', ['agent_id'], unique=False)
    op.create_index(op.f('ix_processes_process_id'), 'processes', ['process_id'], unique=False)
    op.create_index(op.f('ix_processes_timestamp'), 'processes', ['timestamp'], unique=False)
    op.create_index('idx_processes_agent', 'processes', ['agent_id'], unique=False)

    # Create experiments table
    op.create_table('experiments',
        sa.Column('exp_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(), nullable=True, server_default='running'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('exp_id')
    )
    op.create_index(op.f('ix_experiments_created_at'), 'experiments', ['created_at'], unique=False)
    op.create_index(op.f('ix_experiments_exp_id'), 'experiments', ['exp_id'], unique=False)

    # Create analysis_runs table
    op.create_table('analysis_runs',
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('exp_id', sa.String(), nullable=False),
        sa.Column('code_snippet', sa.Text(), nullable=False),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('result', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['exp_id'], ['experiments.exp_id'], ),
        sa.PrimaryKeyConstraint('run_id')
    )
    op.create_index(op.f('ix_analysis_runs_created_at'), 'analysis_runs', ['created_at'], unique=False)
    op.create_index(op.f('ix_analysis_runs_exp_id'), 'analysis_runs', ['exp_id'], unique=False)
    op.create_index(op.f('ix_analysis_runs_run_id'), 'analysis_runs', ['run_id'], unique=False)
    op.create_index('idx_analysis_runs_exp', 'analysis_runs', ['exp_id'], unique=False)

    # Create code_smell_findings table
    op.create_table('code_smell_findings',
        sa.Column('finding_id', sa.String(), nullable=False),
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('smell_type', sa.String(), nullable=False),
        sa.Column('severity', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('agent_id', sa.String(), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.agent_id'], ),
        sa.ForeignKeyConstraint(['run_id'], ['analysis_runs.run_id'], ),
        sa.PrimaryKeyConstraint('finding_id')
    )
    op.create_index(op.f('ix_code_smell_findings_agent_id'), 'code_smell_findings', ['agent_id'], unique=False)
    op.create_index(op.f('ix_code_smell_findings_created_at'), 'code_smell_findings', ['created_at'], unique=False)
    op.create_index(op.f('ix_code_smell_findings_finding_id'), 'code_smell_findings', ['finding_id'], unique=False)
    op.create_index(op.f('ix_code_smell_findings_run_id'), 'code_smell_findings', ['run_id'], unique=False)
    op.create_index(op.f('ix_code_smell_findings_smell_type'), 'code_smell_findings', ['smell_type'], unique=False)
    op.create_index('idx_findings_run', 'code_smell_findings', ['run_id'], unique=False)
    op.create_index('idx_findings_agent', 'code_smell_findings', ['agent_id'], unique=False)
    op.create_index('idx_findings_smell', 'code_smell_findings', ['smell_type'], unique=False)

    # Create ground_truth table
    op.create_table('ground_truth',
        sa.Column('gt_id', sa.String(), nullable=False),
        sa.Column('code_snippet', sa.Text(), nullable=False),
        sa.Column('smell_labels', sa.JSON(), nullable=True),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('gt_id')
    )
    op.create_index(op.f('ix_ground_truth_created_at'), 'ground_truth', ['created_at'], unique=False)
    op.create_index(op.f('ix_ground_truth_gt_id'), 'ground_truth', ['gt_id'], unique=False)


def downgrade() -> None:
    """Drop all database tables."""
    op.drop_table('ground_truth')
    op.drop_table('code_smell_findings')
    op.drop_table('analysis_runs')
    op.drop_table('experiments')
    op.drop_table('processes')
    op.drop_table('agent_actions')
    op.drop_table('agent_responses')
    op.drop_table('agent_requests')
    op.drop_table('agents')
