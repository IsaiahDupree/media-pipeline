"""External API validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
from config import settings
import os
import httpx
from typing import Dict, Any


async def validate_external_apis() -> ValidationResult:
    """Validate external API connectivity and keys"""
    result = ValidationResult(component="external_apis")
    
    # Test OpenAI API
    if settings.openai_api_key:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"}
                )
                if response.status_code == 200:
                    result.metadata["openai_status"] = "connected"
                else:
                    result.add_issue(
                        ValidationSeverity.CRITICAL,
                        f"OpenAI API returned {response.status_code}",
                        details=response.text[:200]
                    )
        except Exception as e:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"OpenAI API test failed: {str(e)}",
                details="API key may be invalid or network issue"
            )
            result.metadata["openai_status"] = "error"
    else:
        result.metadata["openai_status"] = "not_configured"
    
    # Test Blotato API
    if settings.blotato_api_key:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Blotato health check endpoint (if available)
                response = await client.get(
                    "https://api.blotato.com/health",
                    headers={"Authorization": f"Bearer {settings.blotato_api_key}"},
                    follow_redirects=True
                )
                if response.status_code in [200, 404]:  # 404 might mean endpoint doesn't exist
                    result.metadata["blotato_status"] = "connected"
                else:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Blotato API returned {response.status_code}",
                        details="API key may be invalid"
                    )
        except Exception as e:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Blotato API test failed: {str(e)}",
                details="API key may be invalid or network issue"
            )
            result.metadata["blotato_status"] = "error"
    else:
        result.metadata["blotato_status"] = "not_configured"
    
    # Test RapidAPI
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if rapidapi_key:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Test with a simple endpoint
                response = await client.get(
                    "https://tiktok-scraper7.p.rapidapi.com/",
                    headers={
                        "X-RapidAPI-Key": rapidapi_key,
                        "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
                    }
                )
                # Any response means API key is being accepted
                result.metadata["rapidapi_status"] = "connected"
        except Exception as e:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"RapidAPI test failed: {str(e)}",
                details="API key may be invalid or rate limited"
            )
            result.metadata["rapidapi_status"] = "error"
    else:
        result.metadata["rapidapi_status"] = "not_configured"
    
    return result

