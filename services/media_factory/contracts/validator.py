"""
Contract Validator
==================
Utility for validating Media Factory contracts (MF-007).

This module provides validation functions for all Media Factory JSON contracts,
ensuring data integrity across the entire pipeline.
"""

from typing import Dict, Any, Type, Union
from pydantic import BaseModel, ValidationError
from loguru import logger

from . import (
    TrendCardSchema,
    ClusterSchema,
    ContentBriefSchema,
    ScriptSchema,
    TimelineSchema,
    RenderJobSchema,
    PublishJobSchema,
)


class ContractValidationError(Exception):
    """Raised when contract validation fails."""

    def __init__(self, contract_name: str, errors: list):
        self.contract_name = contract_name
        self.errors = errors
        super().__init__(f"Validation failed for {contract_name}: {errors}")


def validate_contract(
    data: Union[Dict[str, Any], str],
    contract_type: Type[BaseModel],
    strict: bool = True
) -> BaseModel:
    """
    Validate data against a Media Factory contract.

    Args:
        data: Dictionary or JSON string to validate
        contract_type: Pydantic model class (e.g., ScriptSchema)
        strict: If True, raise exception on validation error

    Returns:
        Validated Pydantic model instance

    Raises:
        ContractValidationError: If validation fails and strict=True

    Example:
        >>> script_data = {"brief_id": "123", "title": "Test", ...}
        >>> script = validate_contract(script_data, ScriptSchema)
    """
    try:
        # Parse JSON string if needed
        if isinstance(data, str):
            import json
            data = json.loads(data)

        # Validate with Pydantic
        validated = contract_type(**data)
        logger.debug(f"✓ Contract validation passed: {contract_type.__name__}")
        return validated

    except ValidationError as e:
        errors = e.errors()
        logger.error(f"✗ Contract validation failed: {contract_type.__name__}")
        logger.error(f"Errors: {errors}")

        if strict:
            raise ContractValidationError(
                contract_name=contract_type.__name__,
                errors=errors
            )
        return None


def validate_trend_card(data: Union[Dict, str], strict: bool = True) -> TrendCardSchema:
    """Validate TrendCard contract."""
    return validate_contract(data, TrendCardSchema, strict)


def validate_cluster(data: Union[Dict, str], strict: bool = True) -> ClusterSchema:
    """Validate Cluster contract."""
    return validate_contract(data, ClusterSchema, strict)


def validate_content_brief(data: Union[Dict, str], strict: bool = True) -> ContentBriefSchema:
    """Validate ContentBrief contract."""
    return validate_contract(data, ContentBriefSchema, strict)


def validate_script(data: Union[Dict, str], strict: bool = True) -> ScriptSchema:
    """Validate Script contract."""
    return validate_contract(data, ScriptSchema, strict)


def validate_timeline(data: Union[Dict, str], strict: bool = True) -> TimelineSchema:
    """Validate Timeline contract."""
    return validate_contract(data, TimelineSchema, strict)


def validate_render_job(data: Union[Dict, str], strict: bool = True) -> RenderJobSchema:
    """Validate RenderJob contract."""
    return validate_contract(data, RenderJobSchema, strict)


def validate_publish_job(data: Union[Dict, str], strict: bool = True) -> PublishJobSchema:
    """Validate PublishJob contract."""
    return validate_contract(data, PublishJobSchema, strict)


def get_contract_schema(contract_type: Type[BaseModel]) -> Dict[str, Any]:
    """
    Get JSON schema for a contract type.

    Args:
        contract_type: Pydantic model class

    Returns:
        JSON schema dictionary

    Example:
        >>> schema = get_contract_schema(ScriptSchema)
        >>> print(schema["properties"])
    """
    return contract_type.model_json_schema()


def validate_pipeline_sequence(
    trend_card: Dict,
    cluster: Dict,
    content_brief: Dict,
    script: Dict,
    timeline: Dict,
    render_job: Dict
) -> Dict[str, bool]:
    """
    Validate entire pipeline sequence from TrendCard to RenderJob.

    Args:
        trend_card: TrendCard data
        cluster: Cluster data
        content_brief: ContentBrief data
        script: Script data
        timeline: Timeline data
        render_job: RenderJob data

    Returns:
        Dictionary with validation results for each stage

    Example:
        >>> results = validate_pipeline_sequence(...)
        >>> print(results)
        {
            "trend_card": True,
            "cluster": True,
            "content_brief": True,
            "script": True,
            "timeline": True,
            "render_job": True
        }
    """
    results = {}

    stages = [
        ("trend_card", trend_card, TrendCardSchema),
        ("cluster", cluster, ClusterSchema),
        ("content_brief", content_brief, ContentBriefSchema),
        ("script", script, ScriptSchema),
        ("timeline", timeline, TimelineSchema),
        ("render_job", render_job, RenderJobSchema),
    ]

    for stage_name, stage_data, contract_type in stages:
        try:
            validate_contract(stage_data, contract_type, strict=True)
            results[stage_name] = True
        except ContractValidationError as e:
            logger.error(f"Pipeline validation failed at stage: {stage_name}")
            logger.error(f"Errors: {e.errors}")
            results[stage_name] = False

    return results


# Export all validators
__all__ = [
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
