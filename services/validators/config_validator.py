"""Configuration validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
from config import settings
import os


async def validate_configuration() -> ValidationResult:
    """Validate application configuration and environment variables"""
    result = ValidationResult(component="configuration")
    
    # Check OpenAI API key
    if not settings.openai_api_key:
        result.add_issue(
            ValidationSeverity.WARNING,
            "OpenAI API key not configured",
            details="Analysis will use fallback mode without AI",
            fix_suggestion="Set OPENAI_API_KEY environment variable"
        )
    elif len(settings.openai_api_key) < 20:
        result.add_issue(
            ValidationSeverity.CRITICAL,
            "OpenAI API key appears invalid (too short)",
            fix_suggestion="Check OPENAI_API_KEY environment variable"
        )
    
    # Check Blotato API key
    if not settings.blotato_api_key:
        result.add_issue(
            ValidationSeverity.WARNING,
            "Blotato API key not configured",
            details="Publishing to social media will not work",
            fix_suggestion="Set BLOTATO_API_KEY environment variable"
        )
    
    # Check RapidAPI key
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key:
        result.add_issue(
            ValidationSeverity.WARNING,
            "RapidAPI key not configured",
            details="Social media analytics fetching will not work",
            fix_suggestion="Set RAPIDAPI_KEY environment variable"
        )
    
    # Check database URL
    if not settings.database_url or "postgres" not in settings.database_url:
        result.add_issue(
            ValidationSeverity.CRITICAL,
            "Database URL not properly configured",
            fix_suggestion="Set DATABASE_URL environment variable"
        )
    
    # Check Supabase configuration
    if not settings.supabase_url:
        result.add_issue(
            ValidationSeverity.WARNING,
            "Supabase URL not configured",
            details="Some features may not work",
            fix_suggestion="Set SUPABASE_URL environment variable"
        )
    
    # Check Google credentials (if needed)
    if not settings.google_client_id or not settings.google_client_secret:
        result.add_issue(
            ValidationSeverity.WARNING,
            "Google OAuth credentials not configured",
            details="YouTube integration will not work",
            fix_suggestion="Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET"
        )
    
    result.metadata = {
        "has_openai": bool(settings.openai_api_key),
        "has_blotato": bool(settings.blotato_api_key),
        "has_rapidapi": bool(rapidapi_key),
        "has_google": bool(settings.google_client_id and settings.google_client_secret)
    }
    
    return result

