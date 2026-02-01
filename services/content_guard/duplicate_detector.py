"""
Duplicate Content Detector
==========================
Prevents double-posting by detecting similar content across accounts.

Features:
- Transcript similarity check using TF-IDF + cosine similarity
- Configurable similarity threshold
- Per-account duplicate tracking
- Cross-platform detection
- Quality gates before posting

Usage:
    detector = DuplicateDetector()
    
    # Check before posting
    result = await detector.check_content(
        account_id="807",
        transcript="Your video transcript...",
        platform="instagram"
    )
    
    if result.is_duplicate:
        print(f"BLOCKED: Similar to post {result.similar_post_id}")
        print(f"Similarity: {result.similarity_score:.2%}")
"""
import os
import re
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from sqlalchemy import create_engine, text
from loguru import logger

# Optional: Use sklearn for better similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using basic similarity")


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")

# Similarity thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.85  # 85% similar = duplicate
STRICT_SIMILARITY_THRESHOLD = 0.70   # For stricter checking
LOOSE_SIMILARITY_THRESHOLD = 0.92    # For looser checking


@dataclass
class DuplicateCheckResult:
    """Result of duplicate content check"""
    is_duplicate: bool
    similarity_score: float
    similar_post_id: Optional[str]
    similar_post_platform: Optional[str]
    similar_post_date: Optional[str]
    reason: str
    can_post: bool
    warnings: List[str] = field(default_factory=list)


@dataclass
class ContentFingerprint:
    """Fingerprint of content for duplicate detection"""
    content_id: str
    account_id: str
    platform: str
    transcript_hash: str
    transcript_normalized: str
    posted_at: str
    word_count: int


class DuplicateDetector:
    """
    Detects and prevents duplicate content posting.
    
    Uses multiple strategies:
    1. Exact hash matching (fastest)
    2. Normalized text comparison
    3. TF-IDF cosine similarity (most accurate)
    """
    
    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        lookback_days: int = 90
    ):
        self.engine = create_engine(DATABASE_URL)
        self.similarity_threshold = similarity_threshold
        self.lookback_days = lookback_days
        
        # In-memory cache for recent fingerprints
        self._fingerprint_cache: Dict[str, List[ContentFingerprint]] = defaultdict(list)
        self._cache_loaded = False
    
    async def check_content(
        self,
        account_id: str,
        transcript: str,
        platform: str,
        title: Optional[str] = None,
        strict: bool = False
    ) -> DuplicateCheckResult:
        """
        Check if content is a duplicate before posting.
        
        Args:
            account_id: The account to check against
            transcript: Video transcript or script
            platform: Target platform (instagram, tiktok, etc.)
            title: Optional title for additional checking
            strict: Use stricter similarity threshold
        
        Returns:
            DuplicateCheckResult with is_duplicate flag and details
        """
        if not transcript or len(transcript.strip()) < 10:
            return DuplicateCheckResult(
                is_duplicate=False,
                similarity_score=0.0,
                similar_post_id=None,
                similar_post_platform=None,
                similar_post_date=None,
                reason="Content too short to check",
                can_post=True,
                warnings=["Very short content - consider adding more"]
            )
        
        # Normalize the transcript
        normalized = self._normalize_text(transcript)
        content_hash = self._hash_text(normalized)
        
        # Get threshold
        threshold = STRICT_SIMILARITY_THRESHOLD if strict else self.similarity_threshold
        
        # Load existing fingerprints for this account
        existing = await self._get_account_fingerprints(account_id)
        
        if not existing:
            return DuplicateCheckResult(
                is_duplicate=False,
                similarity_score=0.0,
                similar_post_id=None,
                similar_post_platform=None,
                similar_post_date=None,
                reason="No previous posts found for this account",
                can_post=True,
                warnings=[]
            )
        
        # Step 1: Check exact hash match (fastest)
        for fp in existing:
            if fp.transcript_hash == content_hash:
                return DuplicateCheckResult(
                    is_duplicate=True,
                    similarity_score=1.0,
                    similar_post_id=fp.content_id,
                    similar_post_platform=fp.platform,
                    similar_post_date=fp.posted_at,
                    reason="Exact duplicate detected",
                    can_post=False,
                    warnings=[]
                )
        
        # Step 2: Check similarity
        best_match = await self._find_best_match(normalized, existing, threshold)
        
        if best_match:
            fp, score = best_match
            return DuplicateCheckResult(
                is_duplicate=True,
                similarity_score=score,
                similar_post_id=fp.content_id,
                similar_post_platform=fp.platform,
                similar_post_date=fp.posted_at,
                reason=f"Content is {score:.1%} similar to previous post",
                can_post=False,
                warnings=[]
            )
        
        # No duplicate found - check for warnings
        warnings = []
        
        # Check for near-duplicates (warn but allow)
        near_match = await self._find_best_match(normalized, existing, threshold - 0.15)
        if near_match:
            fp, score = near_match
            warnings.append(
                f"Similar content ({score:.1%}) posted on {fp.platform} on {fp.posted_at[:10]}"
            )
        
        return DuplicateCheckResult(
            is_duplicate=False,
            similarity_score=near_match[1] if near_match else 0.0,
            similar_post_id=None,
            similar_post_platform=None,
            similar_post_date=None,
            reason="Content is unique",
            can_post=True,
            warnings=warnings
        )
    
    async def register_posted_content(
        self,
        content_id: str,
        account_id: str,
        platform: str,
        transcript: str,
        posted_at: Optional[datetime] = None
    ) -> bool:
        """
        Register content after posting to track for future duplicate checks.
        
        Args:
            content_id: Unique ID for this content
            account_id: Account that posted
            platform: Platform posted to
            transcript: Content transcript
            posted_at: When it was posted
        
        Returns:
            True if registered successfully
        """
        normalized = self._normalize_text(transcript)
        content_hash = self._hash_text(normalized)
        posted_at = posted_at or datetime.now(timezone.utc)
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO content_fingerprints 
                    (content_id, account_id, platform, transcript_hash, transcript_normalized, posted_at, word_count)
                    VALUES (:content_id, :account_id, :platform, :hash, :normalized, :posted_at, :word_count)
                    ON CONFLICT (content_id, account_id, platform) 
                    DO UPDATE SET transcript_hash = :hash, transcript_normalized = :normalized
                """), {
                    "content_id": content_id,
                    "account_id": account_id,
                    "platform": platform,
                    "hash": content_hash,
                    "normalized": normalized[:5000],  # Limit storage
                    "posted_at": posted_at,
                    "word_count": len(normalized.split())
                })
                conn.commit()
            
            # Update cache
            fp = ContentFingerprint(
                content_id=content_id,
                account_id=account_id,
                platform=platform,
                transcript_hash=content_hash,
                transcript_normalized=normalized,
                posted_at=posted_at.isoformat(),
                word_count=len(normalized.split())
            )
            self._fingerprint_cache[account_id].append(fp)
            
            logger.info(f"Registered content fingerprint: {content_id} for account {account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register content: {e}")
            return False
    
    async def get_account_history(
        self,
        account_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get posting history for an account"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT content_id, platform, posted_at, word_count
                    FROM content_fingerprints
                    WHERE account_id = :account_id
                    ORDER BY posted_at DESC
                    LIMIT :limit
                """), {"account_id": account_id, "limit": limit})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.warning(f"Could not fetch account history: {e}")
            return []
    
    async def _get_account_fingerprints(self, account_id: str) -> List[ContentFingerprint]:
        """Get fingerprints for an account from DB or cache"""
        if account_id in self._fingerprint_cache:
            return self._fingerprint_cache[account_id]
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT content_id, account_id, platform, transcript_hash, 
                           transcript_normalized, posted_at, word_count
                    FROM content_fingerprints
                    WHERE account_id = :account_id
                    AND posted_at > :cutoff
                    ORDER BY posted_at DESC
                """), {"account_id": account_id, "cutoff": cutoff_date})
                
                fingerprints = []
                for row in result:
                    r = row._mapping
                    fingerprints.append(ContentFingerprint(
                        content_id=r["content_id"],
                        account_id=r["account_id"],
                        platform=r["platform"],
                        transcript_hash=r["transcript_hash"],
                        transcript_normalized=r["transcript_normalized"],
                        posted_at=str(r["posted_at"]),
                        word_count=r["word_count"]
                    ))
                
                self._fingerprint_cache[account_id] = fingerprints
                return fingerprints
                
        except Exception as e:
            logger.warning(f"Could not fetch fingerprints from DB: {e}")
            return []
    
    async def _find_best_match(
        self,
        normalized_text: str,
        existing: List[ContentFingerprint],
        threshold: float
    ) -> Optional[Tuple[ContentFingerprint, float]]:
        """Find the best matching fingerprint above threshold"""
        if not existing:
            return None
        
        if SKLEARN_AVAILABLE and len(existing) > 0:
            return self._sklearn_similarity(normalized_text, existing, threshold)
        else:
            return self._basic_similarity(normalized_text, existing, threshold)
    
    def _sklearn_similarity(
        self,
        text: str,
        existing: List[ContentFingerprint],
        threshold: float
    ) -> Optional[Tuple[ContentFingerprint, float]]:
        """Use TF-IDF + cosine similarity"""
        texts = [fp.transcript_normalized for fp in existing]
        texts.append(text)
        
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
            
            best_idx = similarities.argmax()
            best_score = similarities[best_idx]
            
            if best_score >= threshold:
                return (existing[best_idx], float(best_score))
            return None
            
        except Exception as e:
            logger.warning(f"TF-IDF similarity failed: {e}")
            return self._basic_similarity(text, existing, threshold)
    
    def _basic_similarity(
        self,
        text: str,
        existing: List[ContentFingerprint],
        threshold: float
    ) -> Optional[Tuple[ContentFingerprint, float]]:
        """Basic word overlap similarity"""
        text_words = set(text.lower().split())
        
        best_match = None
        best_score = 0.0
        
        for fp in existing:
            fp_words = set(fp.transcript_normalized.lower().split())
            
            if not text_words or not fp_words:
                continue
            
            intersection = len(text_words & fp_words)
            union = len(text_words | fp_words)
            
            if union > 0:
                jaccard = intersection / union
                if jaccard > best_score:
                    best_score = jaccard
                    best_match = fp
        
        if best_match and best_score >= threshold:
            return (best_match, best_score)
        return None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove emojis (basic)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _hash_text(self, text: str) -> str:
        """Create hash of normalized text"""
        return hashlib.sha256(text.encode()).hexdigest()[:32]


# Database migration
MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS content_fingerprints (
    id SERIAL PRIMARY KEY,
    content_id VARCHAR(255) NOT NULL,
    account_id VARCHAR(255) NOT NULL,
    platform VARCHAR(50) NOT NULL,
    transcript_hash VARCHAR(64) NOT NULL,
    transcript_normalized TEXT,
    posted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    word_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(content_id, account_id, platform)
);

CREATE INDEX IF NOT EXISTS idx_fingerprints_account ON content_fingerprints(account_id);
CREATE INDEX IF NOT EXISTS idx_fingerprints_hash ON content_fingerprints(transcript_hash);
CREATE INDEX IF NOT EXISTS idx_fingerprints_posted ON content_fingerprints(posted_at);
"""


async def run_migration():
    """Run database migration"""
    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text(MIGRATION_SQL))
            conn.commit()
        logger.success("✅ content_fingerprints table created")
    except Exception as e:
        logger.error(f"Migration failed: {e}")


# Test function
async def test_duplicate_detector():
    """Test the duplicate detector"""
    print("\n" + "="*60)
    print("🔍 DUPLICATE DETECTOR TEST")
    print("="*60)
    
    detector = DuplicateDetector()
    
    # Test 1: Register some content
    print("\n1. Registering test content...")
    await detector.register_posted_content(
        content_id="test_001",
        account_id="807",
        platform="instagram",
        transcript="Hey everyone! Today I'm going to show you 5 productivity hacks that changed my life. First, wake up early and do the hardest task first."
    )
    print("   ✅ Registered test_001")
    
    # Test 2: Check exact duplicate
    print("\n2. Checking exact duplicate...")
    result = await detector.check_content(
        account_id="807",
        transcript="Hey everyone! Today I'm going to show you 5 productivity hacks that changed my life. First, wake up early and do the hardest task first.",
        platform="instagram"
    )
    print(f"   Is duplicate: {result.is_duplicate}")
    print(f"   Similarity: {result.similarity_score:.1%}")
    print(f"   Can post: {result.can_post}")
    
    # Test 3: Check similar content
    print("\n3. Checking similar content...")
    result = await detector.check_content(
        account_id="807",
        transcript="Hey guys! Here are 5 productivity tips that transformed my work. Number one, get up early and tackle the biggest task immediately.",
        platform="tiktok"
    )
    print(f"   Is duplicate: {result.is_duplicate}")
    print(f"   Similarity: {result.similarity_score:.1%}")
    print(f"   Reason: {result.reason}")
    
    # Test 4: Check unique content
    print("\n4. Checking unique content...")
    result = await detector.check_content(
        account_id="807",
        transcript="This is a completely different video about cooking recipes and kitchen organization.",
        platform="instagram"
    )
    print(f"   Is duplicate: {result.is_duplicate}")
    print(f"   Can post: {result.can_post}")
    print(f"   Warnings: {result.warnings}")
    
    print("\n✅ Duplicate detector tests complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_migration())
    asyncio.run(test_duplicate_detector())
