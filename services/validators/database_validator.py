"""Database validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
from database.connection import async_session_maker
from sqlalchemy import text
from config import settings


async def validate_database() -> ValidationResult:
    """Validate database connectivity and schema"""
    result = ValidationResult(component="database")
    
    try:
        # Test connection
        async with async_session_maker() as db:
            await db.execute(text("SELECT 1"))
            result.metadata["connected"] = True
    except Exception as e:
        result.add_issue(
            ValidationSeverity.CRITICAL,
            "Cannot connect to database",
            details=str(e),
            fix_suggestion=f"Check DATABASE_URL: {settings.database_url[:50]}..."
        )
        result.metadata["connected"] = False
        return result
    
    # Check critical tables exist
    critical_tables = [
        "videos", "video_analysis", "scheduled_posts",
        "social_accounts", "platform_posts"
    ]
    
    missing_tables = []
    try:
        async with async_session_maker() as db:
            for table in critical_tables:
                try:
                    await db.execute(
                        text(f"SELECT 1 FROM {table} LIMIT 1")
                    )
                except Exception:
                    missing_tables.append(table)
    except Exception as e:
        result.add_issue(
            ValidationSeverity.WARNING,
            "Could not verify all tables",
            details=str(e)
        )
    
    if missing_tables:
        result.add_issue(
            ValidationSeverity.CRITICAL,
            f"Missing critical tables: {', '.join(missing_tables)}",
            fix_suggestion="Run database migrations"
        )
    
    # Check database size/health
    try:
        async with async_session_maker() as db:
            size_result = await db.execute(
                text("SELECT pg_database_size(current_database())")
            )
            db_size = size_result.scalar()
            result.metadata["database_size_bytes"] = db_size
            result.metadata["database_size_mb"] = db_size / (1024 * 1024)
            
            if db_size > 10 * 1024 * 1024 * 1024:  # > 10GB
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Database is large ({db_size / (1024**3):.2f} GB)",
                    fix_suggestion="Consider archiving old data"
                )
    except Exception:
        pass  # Not critical
    
    return result

