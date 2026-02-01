"""Scheduled posts validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
from database.connection import async_session_maker
from sqlalchemy import text
from datetime import datetime, timedelta


async def validate_scheduled_posts() -> ValidationResult:
    """Validate scheduled posts before they're published"""
    result = ValidationResult(component="scheduled_posts")
    
    try:
        async with async_session_maker() as db:
            # Get upcoming scheduled posts (next 7 days) using raw SQL
            upcoming_date = datetime.now() + timedelta(days=7)
            
            query = text("""
                SELECT id, video_id, account_id, scheduled_at, status, platform, title, caption
                FROM scheduled_posts
                WHERE scheduled_at <= :upcoming_date
                  AND status IN ('pending', 'queued')
            """)
            posts_result = await db.execute(query, {"upcoming_date": upcoming_date})
            posts = posts_result.fetchall()
            
            result.metadata["total_upcoming"] = len(posts)
            result.metadata["issues_found"] = 0
            
            for post in posts:
                post_id, video_id, account_id, scheduled_at, status, platform, title, caption = post
                issues = []
                
                # Check if video exists and is accessible
                if video_id:
                    video_query = text("""
                        SELECT id, source_uri FROM videos WHERE id = :video_id
                    """)
                    video_result = await db.execute(video_query, {"video_id": video_id})
                    video = video_result.fetchone()
                    
                    if not video:
                        issues.append("Video not found")
                        result.metadata["issues_found"] += 1
                    elif not video[1]:  # source_uri column
                        issues.append("Video has no source file")
                        result.metadata["issues_found"] += 1
                
                # Check if account exists and is active
                if account_id:
                    account_query = text("""
                        SELECT id, status FROM social_accounts WHERE id = :account_id
                    """)
                    account_result = await db.execute(account_query, {"account_id": account_id})
                    account = account_result.fetchone()
                    
                    if not account:
                        issues.append("Social account not found")
                        result.metadata["issues_found"] += 1
                    elif account[1] != 'active':  # status column
                        issues.append("Social account is inactive")
                        result.metadata["issues_found"] += 1
                
                # Check if scheduled time is in the past
                if scheduled_at and scheduled_at < datetime.now():
                    issues.append("Scheduled time is in the past")
                    result.metadata["issues_found"] += 1
                
                # Check if post has content
                if not caption and not title:
                    issues.append("Post has no content (title or caption)")
                    result.metadata["issues_found"] += 1
                
                if issues:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        f"Scheduled post {post_id} has issues: {', '.join(issues)}",
                        details=f"Scheduled for: {scheduled_at}, Platform: {platform}"
                    )
            
            if result.metadata["issues_found"] > 0:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"{result.metadata['issues_found']} scheduled post(s) have validation issues",
                    fix_suggestion="Review and fix scheduled posts before publishing"
                )
    
    except Exception as e:
        result.add_issue(
            ValidationSeverity.CRITICAL,
            f"Failed to validate scheduled posts: {str(e)}",
            details=str(e)
        )
    
    return result

