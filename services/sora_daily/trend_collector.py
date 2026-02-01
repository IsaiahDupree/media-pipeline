"""
Trend Collector for Sora Daily Automation
Collects trending topics from comments, DMs, and social platforms.
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from uuid import uuid4
from loguru import logger
from sqlalchemy import create_engine, text


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres")


@dataclass
class TrendSource:
    """A trending topic source."""
    id: str = field(default_factory=lambda: str(uuid4()))
    source_type: str = "comment"  # comment, dm, twitter, tiktok
    topic: str = ""
    relevance_score: float = 0.5
    used_in_story: bool = False
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "topic": self.topic,
            "relevance_score": self.relevance_score,
            "used_in_story": self.used_in_story,
            "discovered_at": self.discovered_at.isoformat() if self.discovered_at else None
        }


class TrendCollector:
    """
    Collects and analyzes trending topics from various sources.
    
    Sources:
    - Comments on posts
    - DMs and inbox messages
    - Twitter trending topics
    - TikTok trending sounds/topics
    """
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        logger.info("✅ TrendCollector initialized")
    
    async def collect_from_comments(self, limit: int = 50) -> List[TrendSource]:
        """Collect trending topics from recent comments."""
        trends = []
        
        try:
            # Get recent inbox messages (comments)
            from services.inbox import get_inbox_service
            inbox = get_inbox_service()
            
            messages = inbox.get_messages(
                message_type="comment",
                limit=limit
            )
            
            # Extract topics from comments
            topics = await self._extract_topics([m.content for m in messages])
            
            for topic, score in topics:
                trend = TrendSource(
                    source_type="comment",
                    topic=topic,
                    relevance_score=score
                )
                trends.append(trend)
                self._save_trend(trend)
            
        except Exception as e:
            logger.error(f"Failed to collect from comments: {e}")
        
        return trends
    
    async def collect_from_dms(self, limit: int = 30) -> List[TrendSource]:
        """Collect trending topics from DMs."""
        trends = []
        
        try:
            from services.inbox import get_inbox_service
            inbox = get_inbox_service()
            
            messages = inbox.get_messages(
                message_type="dm",
                limit=limit
            )
            
            topics = await self._extract_topics([m.content for m in messages])
            
            for topic, score in topics:
                trend = TrendSource(
                    source_type="dm",
                    topic=topic,
                    relevance_score=score
                )
                trends.append(trend)
                self._save_trend(trend)
            
        except Exception as e:
            logger.error(f"Failed to collect from DMs: {e}")
        
        return trends
    
    async def collect_from_relationship_context(self) -> List[TrendSource]:
        """Collect topics from relationship CRM context cards."""
        trends = []
        
        try:
            from services.relationship_crm import get_relationship_crm
            crm = get_relationship_crm()
            
            # Get contacts needing care - their context may have relevant topics
            contacts = crm.get_needs_care(limit=20)
            
            topics = []
            for contact in contacts:
                if contact and contact.context:
                    if contact.context.struggles:
                        topics.append(contact.context.struggles)
                    if contact.context.building:
                        topics.append(contact.context.building)
            
            extracted = await self._extract_topics(topics)
            
            for topic, score in extracted:
                trend = TrendSource(
                    source_type="crm_context",
                    topic=topic,
                    relevance_score=score
                )
                trends.append(trend)
                self._save_trend(trend)
            
        except Exception as e:
            logger.error(f"Failed to collect from CRM: {e}")
        
        return trends
    
    async def _extract_topics(self, texts: List[str]) -> List[tuple]:
        """
        Extract topics from text using AI.
        
        Returns list of (topic, relevance_score) tuples.
        """
        if not texts:
            return []
        
        try:
            from openai import OpenAI
        except ImportError:
            return []
        
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            combined = "\n".join(texts[:50])  # Limit input
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Extract trending topics from these messages.
Return a JSON array of objects with format:
{"topics": [{"topic": "topic name", "score": 0.0-1.0}]}

Focus on:
- What people are asking about
- Common pain points
- Popular interests
- Timely events or trends

Return max 10 topics, ordered by relevance."""
                    },
                    {
                        "role": "user",
                        "content": combined
                    }
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return [(t["topic"], t["score"]) for t in result.get("topics", [])]
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return []
    
    def _save_trend(self, trend: TrendSource):
        """Save trend to database."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO sora_trend_sources (id, source_type, topic, relevance_score, used_in_story)
                    VALUES (:id, :source_type, :topic, :score, :used)
                    ON CONFLICT DO NOTHING
                """), {
                    "id": trend.id,
                    "source_type": trend.source_type,
                    "topic": trend.topic,
                    "score": trend.relevance_score,
                    "used": trend.used_in_story
                })
                conn.commit()
        except Exception as e:
            logger.debug(f"Trend save failed: {e}")
    
    def get_unused_trends(self, limit: int = 20) -> List[TrendSource]:
        """Get trends not yet used in stories."""
        try:
            with self.engine.connect() as conn:
                results = conn.execute(text("""
                    SELECT id, source_type, topic, relevance_score, used_in_story, discovered_at
                    FROM sora_trend_sources
                    WHERE used_in_story = FALSE
                    ORDER BY relevance_score DESC, discovered_at DESC
                    LIMIT :limit
                """), {"limit": limit}).fetchall()
                
                return [
                    TrendSource(
                        id=r[0],
                        source_type=r[1],
                        topic=r[2],
                        relevance_score=r[3],
                        used_in_story=r[4],
                        discovered_at=r[5]
                    )
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Failed to get unused trends: {e}")
            return []
    
    def mark_trend_used(self, trend_id: str):
        """Mark a trend as used in a story."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE sora_trend_sources
                    SET used_in_story = TRUE
                    WHERE id = :id
                """), {"id": trend_id})
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to mark trend used: {e}")
    
    async def collect_all(self) -> List[TrendSource]:
        """Collect trends from all sources."""
        all_trends = []
        
        comment_trends = await self.collect_from_comments()
        all_trends.extend(comment_trends)
        
        dm_trends = await self.collect_from_dms()
        all_trends.extend(dm_trends)
        
        crm_trends = await self.collect_from_relationship_context()
        all_trends.extend(crm_trends)
        
        # Sort by relevance
        all_trends.sort(key=lambda t: t.relevance_score, reverse=True)
        
        logger.info(f"📈 Collected {len(all_trends)} trends")
        return all_trends


# =============================================================================
# SINGLETON
# =============================================================================

_trend_collector_instance: Optional[TrendCollector] = None

def get_trend_collector() -> TrendCollector:
    """Get singleton instance of TrendCollector."""
    global _trend_collector_instance
    if _trend_collector_instance is None:
        _trend_collector_instance = TrendCollector()
    return _trend_collector_instance
