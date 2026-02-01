"""
Media Factory Data Contracts (MF-007)
======================================
Stable interfaces (schemas) for all data structures in the pipeline.

These contracts enable:
- Provider swapping
- Multi-server rendering
- Version compatibility
- Schema validation

All contracts are Pydantic models with built-in validation.
"""

from .trend_card import TrendCardSchema
from .cluster import ClusterSchema
from .content_brief import ContentBriefSchema
from .script import ScriptSchema
from .timeline import TimelineSchema
from .render_job import RenderJobSchema
from .publish_job import PublishJobSchema
from .validator import (
    ContractValidationError,
    validate_contract,
    validate_trend_card,
    validate_cluster,
    validate_content_brief,
    validate_script,
    validate_timeline,
    validate_render_job,
    validate_publish_job,
    get_contract_schema,
    validate_pipeline_sequence,
)

__all__ = [
    # Schemas
    "TrendCardSchema",
    "ClusterSchema",
    "ContentBriefSchema",
    "ScriptSchema",
    "TimelineSchema",
    "RenderJobSchema",
    "PublishJobSchema",
    # Validation
    "ContractValidationError",
    "validate_contract",
    "validate_trend_card",
    "validate_cluster",
    "validate_content_brief",
    "validate_script",
    "validate_timeline",
    "validate_render_job",
    "validate_publish_job",
    "get_contract_schema",
    "validate_pipeline_sequence",
]

