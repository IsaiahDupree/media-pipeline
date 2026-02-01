"""
Deduplication Guard Service
Prevents double-posting and ensures idempotency for scheduled posts.

This service provides multiple layers of protection:
1. Database constraints (unique indexes)
2. Pre-publish checks (already posted?)
3. Idempotency keys for API calls
4. Status tracking to prevent re-processing
"""
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, text
from loguru import logger
from uuid import uuid4

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")


class DeduplicationGuard:
    """
    Service to prevent duplicate posts and ensure idempotency
    """
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
    
    def ensure_deduplication_schema(self):
        """
        Create necessary tables and constraints for deduplication.
        Each operation runs in its own transaction to prevent cascading failures.
        """
        # 1. Add unique constraint to prevent duplicate scheduled posts
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_posts_unique_content_time
                    ON scheduled_posts(content_id, platform, account_id, scheduled_time)
                    WHERE status IN ('scheduled', 'posted')
                """))
                conn.commit()
                logger.info("✓ Created unique index for scheduled posts")
        except Exception as e:
            logger.debug(f"Unique index may already exist: {e}")
        
        # 2. Add unique constraint to posted_content to prevent duplicates
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_posted_content_unique_platform_post
                    ON posted_content(platform, platform_post_id)
                    WHERE platform_post_id IS NOT NULL
                """))
                conn.commit()
                logger.info("✓ Created unique index for posted content")
        except Exception as e:
            logger.debug(f"Unique index may already exist: {e}")
        
        # 3. Create idempotency tracking table
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS publish_idempotency (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        idempotency_key TEXT UNIQUE NOT NULL,
                        scheduled_post_id UUID,
                        content_id UUID,
                        platform TEXT NOT NULL,
                        account_id TEXT,
                        status TEXT NOT NULL,
                        platform_post_id TEXT,
                        platform_url TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        completed_at TIMESTAMP WITH TIME ZONE,
                        error_message TEXT,
                        metadata JSONB DEFAULT '{}'
                    )
                """))
                conn.commit()
                logger.info("✓ Created publish_idempotency table")
        except Exception as e:
            logger.debug(f"Idempotency table may already exist: {e}")
        
        # 4. Create indexes for idempotency table
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_publish_idempotency_key
                    ON publish_idempotency(idempotency_key)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_publish_idempotency_scheduled_post
                    ON publish_idempotency(scheduled_post_id)
                """))
                conn.commit()
                logger.info("✓ Created idempotency indexes")
        except Exception as e:
            logger.debug(f"Idempotency indexes may already exist: {e}")
        
        logger.info("✓ Deduplication schema ready")
    
    def generate_idempotency_key(
        self,
        content_id: str,
        platform: str,
        account_id: str,
        scheduled_time: Optional[datetime] = None
    ) -> str:
        """
        Generate a deterministic idempotency key for a publish operation.
        Same content + platform + account + time = same key
        """
        time_str = scheduled_time.isoformat() if scheduled_time else "immediate"
        return f"publish:{content_id}:{platform}:{account_id}:{time_str}"
    
    def check_idempotency(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """
        Check if this publish operation has already been attempted.
        
        Returns:
            - None if never attempted (safe to proceed)
            - Dict with status if already attempted (contains result)
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    id, status, platform_post_id, platform_url, 
                    error_message, created_at, completed_at
                FROM publish_idempotency
                WHERE idempotency_key = :key
            """), {"key": idempotency_key}).fetchone()
            
            if not result:
                return None
            
            return {
                "id": str(result[0]),
                "status": result[1],
                "platform_post_id": result[2],
                "platform_url": result[3],
                "error_message": result[4],
                "created_at": result[5].isoformat() if result[5] else None,
                "completed_at": result[6].isoformat() if result[6] else None
            }
    
    def record_publish_attempt(
        self,
        idempotency_key: str,
        scheduled_post_id: Optional[str],
        content_id: str,
        platform: str,
        account_id: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record that a publish attempt is starting.
        Returns the record ID.
        """
        with self.engine.connect() as conn:
            import json
            result = conn.execute(text("""
                INSERT INTO publish_idempotency 
                (idempotency_key, scheduled_post_id, content_id, platform, account_id, status, metadata)
                VALUES (:key, :scheduled_post_id, :content_id, :platform, :account_id, 'in_progress', CAST(:metadata AS jsonb))
                ON CONFLICT (idempotency_key) 
                DO UPDATE SET 
                    status = 'in_progress',
                    created_at = NOW(),
                    metadata = EXCLUDED.metadata
                RETURNING id
            """), {
                "key": idempotency_key,
                "scheduled_post_id": scheduled_post_id,
                "content_id": content_id,
                "platform": platform,
                "account_id": account_id,
                "metadata": json.dumps(metadata or {})
            }).fetchone()
            
            conn.commit()
            return str(result[0])
    
    def record_publish_success(
        self,
        idempotency_key: str,
        platform_post_id: str,
        platform_url: Optional[str] = None
    ):
        """
        Record successful publish completion.
        """
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE publish_idempotency
                SET 
                    status = 'completed',
                    platform_post_id = :platform_post_id,
                    platform_url = :platform_url,
                    completed_at = NOW()
                WHERE idempotency_key = :key
            """), {
                "key": idempotency_key,
                "platform_post_id": platform_post_id,
                "platform_url": platform_url
            })
            conn.commit()
    
    def record_publish_failure(
        self,
        idempotency_key: str,
        error_message: str
    ):
        """
        Record failed publish attempt.
        """
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE publish_idempotency
                SET 
                    status = 'failed',
                    error_message = :error,
                    completed_at = NOW()
                WHERE idempotency_key = :key
            """), {
                "key": idempotency_key,
                "error": error_message
            })
            conn.commit()
    
    def is_already_posted(
        self,
        content_id: str,
        platform: str,
        account_id: Optional[str] = None
    ) -> bool:
        """
        Check if this content has already been posted to this platform/account.
        
        Returns True if already posted (should NOT post again)
        """
        with self.engine.connect() as conn:
            # Check scheduled_posts table
            query = """
                SELECT COUNT(*) FROM scheduled_posts
                WHERE content_id = :content_id
                  AND platform = :platform
                  AND status = 'posted'
            """
            params = {"content_id": content_id, "platform": platform}
            
            if account_id:
                query += " AND account_id = :account_id"
                params["account_id"] = account_id
            
            count = conn.execute(text(query), params).scalar()
            
            if count > 0:
                logger.warning(f"⚠️ Content {content_id} already posted to {platform}")
                return True
            
            # Also check posted_content table
            query2 = """
                SELECT COUNT(*) FROM posted_content
                WHERE platform = :platform
                  AND status = 'published'
            """
            params2 = {"platform": platform}
            
            if account_id:
                query2 += " AND account_id = :account_id"
                params2["account_id"] = account_id
            
            # Try to match by content_id if the column exists
            try:
                query2 += " AND content_id = :content_id"
                params2["content_id"] = content_id
                count2 = conn.execute(text(query2), params2).scalar()
                
                if count2 > 0:
                    logger.warning(f"⚠️ Content {content_id} found in posted_content for {platform}")
                    return True
            except Exception:
                # content_id column might not exist in posted_content
                pass
            
            return False
    
    def prevent_concurrent_publish(
        self,
        scheduled_post_id: str
    ) -> bool:
        """
        Try to lock a scheduled post for publishing.
        
        Returns:
            True if lock acquired (safe to publish)
            False if already being processed (skip)
        """
        with self.engine.connect() as conn:
            # Try to update status from 'scheduled' to 'publishing'
            result = conn.execute(text("""
                UPDATE scheduled_posts
                SET 
                    status = 'publishing',
                    updated_at = NOW()
                WHERE id = :id
                  AND status = 'scheduled'
                RETURNING id
            """), {"id": scheduled_post_id})
            
            conn.commit()
            
            # If we updated a row, we got the lock
            return result.rowcount > 0
    
    def cleanup_stale_locks(self, max_age_minutes: int = 30):
        """
        Clean up posts stuck in 'publishing' status for too long.
        This handles cases where the publisher crashed mid-publish.
        """
        with self.engine.connect() as conn:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
            
            result = conn.execute(text("""
                UPDATE scheduled_posts
                SET 
                    status = 'scheduled',
                    updated_at = NOW()
                WHERE status = 'publishing'
                  AND updated_at < :cutoff
                RETURNING id
            """), {"cutoff": cutoff})
            
            conn.commit()
            
            count = result.rowcount
            if count > 0:
                logger.warning(f"🔓 Released {count} stale publish locks")
            
            return count
    
    def get_duplicate_schedules(self) -> List[Dict[str, Any]]:
        """
        Find duplicate scheduled posts (same content, platform, account, time).
        Useful for debugging and cleanup.
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    content_id, platform, account_id, scheduled_time,
                    COUNT(*) as duplicate_count,
                    array_agg(id) as post_ids
                FROM scheduled_posts
                WHERE status IN ('scheduled', 'posted')
                GROUP BY content_id, platform, account_id, scheduled_time
                HAVING COUNT(*) > 1
            """))
            
            duplicates = []
            for row in result.fetchall():
                duplicates.append({
                    "content_id": row[0],
                    "platform": row[1],
                    "account_id": row[2],
                    "scheduled_time": row[3].isoformat() if row[3] else None,
                    "duplicate_count": row[4],
                    "post_ids": row[5]
                })
            
            return duplicates


# Singleton instance
_guard_instance = None

def get_deduplication_guard() -> DeduplicationGuard:
    """Get or create the deduplication guard singleton"""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = DeduplicationGuard()
        _guard_instance.ensure_deduplication_schema()
    return _guard_instance
