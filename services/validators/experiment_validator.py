"""Experiment setup validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
from database.connection import async_session_maker
from sqlalchemy import text


async def validate_experiment_setup() -> ValidationResult:
    """Validate experiment setup before running"""
    result = ValidationResult(component="experiment_setup")
    
    try:
        async with async_session_maker() as db:
            # Check for active experiments using raw SQL
            experiments_query = text("""
                SELECT id, name, start_date, end_date, status
                FROM experiments
                WHERE status IN ('running', 'draft')
            """)
            experiments_result = await db.execute(experiments_query)
            experiments = experiments_result.fetchall()
            
            result.metadata["active_experiments"] = len(experiments)
            
            for experiment in experiments:
                exp_id, name, start_date, end_date, status = experiment
                
                # Check if experiment has hypotheses
                hypotheses_query = text("""
                    SELECT COUNT(*) FROM hypotheses
                    WHERE experiment_id = :experiment_id
                """)
                hypotheses_result = await db.execute(hypotheses_query, {"experiment_id": exp_id})
                hypothesis_count = hypotheses_result.scalar()
                
                if hypothesis_count == 0:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Experiment '{name if name else 'Unknown'}' has no hypotheses",
                        fix_suggestion="Add at least one hypothesis to the experiment"
                    )
                
                # Check if experiment has valid date range
                if end_date and start_date:
                    if end_date < start_date:
                        result.add_issue(
                            ValidationSeverity.CRITICAL,
                            f"Experiment '{name if name else 'Unknown'}' has invalid date range",
                            details="End date is before start date"
                        )
    
    except Exception as e:
        result.add_issue(
            ValidationSeverity.CRITICAL,
            f"Failed to validate experiments: {str(e)}",
            details=str(e)
        )
    
    return result

