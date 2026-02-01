"""Narrative builder validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
from database.connection import async_session_maker
from sqlalchemy import text
from datetime import datetime


async def validate_narrative_setup() -> ValidationResult:
    """Validate narrative builder setup before generating plans"""
    result = ValidationResult(component="narrative_setup")
    
    try:
        async with async_session_maker() as db:
            # Check for active goals using raw SQL
            goals_query = text("""
                SELECT id, goal_statement, start_date, end_date, status
                FROM narrative_goals
                WHERE status = 'active'
            """)
            goals_result = await db.execute(goals_query)
            goals = goals_result.fetchall()
            
            result.metadata["active_goals"] = len(goals)
            
            for goal in goals:
                goal_id, goal_statement, start_date, end_date, status = goal
                
                # Check if goal has pillars
                pillars_query = text("""
                    SELECT COUNT(*) FROM narrative_pillars
                    WHERE goal_id = :goal_id AND is_active = true
                """)
                pillars_result = await db.execute(pillars_query, {"goal_id": goal_id})
                pillar_count = pillars_result.scalar()
                
                if pillar_count == 0:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Goal '{goal_statement[:50] if goal_statement else 'Unknown'}...' has no active pillars",
                        fix_suggestion="Add at least one narrative pillar to the goal"
                    )
                
                # Check if goal has valid date range
                if end_date and start_date:
                    if end_date < start_date:
                        result.add_issue(
                            ValidationSeverity.CRITICAL,
                            f"Goal '{goal_statement[:50] if goal_statement else 'Unknown'}...' has invalid date range",
                            details="End date is before start date"
                        )
            
            if len(goals) == 0:
                result.add_issue(
                    ValidationSeverity.INFO,
                    "No active narrative goals found",
                    details="Create a narrative goal to start content planning"
                )
    
    except Exception as e:
        result.add_issue(
            ValidationSeverity.CRITICAL,
            f"Failed to validate narrative setup: {str(e)}",
            details=str(e)
        )
    
    return result

