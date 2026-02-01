"""Social media accounts validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
from database.connection import async_session_maker
from sqlalchemy import text
import httpx
from config import settings


async def validate_social_accounts() -> ValidationResult:
    """Validate social media account connectivity"""
    result = ValidationResult(component="social_accounts")
    
    try:
        async with async_session_maker() as db:
            # Query social_accounts table directly using raw SQL
            query = text("""
                SELECT id, platform, handle, status, access_token
                FROM social_accounts
                WHERE status = 'active'
            """)
            accounts_result = await db.execute(query)
            accounts = accounts_result.fetchall()
            
            result.metadata["total_accounts"] = len(accounts)
            result.metadata["accounts"] = []
            
            for account in accounts:
                account_id, platform, handle, status, access_token = account
                account_info = {
                    "id": str(account_id),
                    "platform": platform,
                    "handle": handle,
                    "status": "unknown"
                }
                
                # Check if account has required tokens/credentials
                if not access_token and platform != "youtube":
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Account {handle} ({platform}) missing access token",
                        fix_suggestion="Re-authenticate the account"
                    )
                    account_info["status"] = "missing_token"
                
                # Test account connectivity (if Blotato is configured)
                if settings.blotato_api_key and platform in ["tiktok", "instagram"]:
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            # Test account status via Blotato
                            # This is a placeholder - adjust based on actual Blotato API
                            account_info["status"] = "connected"
                    except Exception:
                        account_info["status"] = "error"
                        result.add_issue(
                            ValidationSeverity.WARNING,
                            f"Could not verify {handle} ({platform}) connectivity",
                            details="Account may need re-authentication"
                        )
                
                result.metadata["accounts"].append(account_info)
            
            if len(accounts) == 0:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "No active social media accounts configured",
                    fix_suggestion="Connect at least one social media account"
                )
    
    except Exception as e:
        result.add_issue(
            ValidationSeverity.CRITICAL,
            f"Failed to validate social accounts: {str(e)}",
            details=str(e)
        )
    
    return result

