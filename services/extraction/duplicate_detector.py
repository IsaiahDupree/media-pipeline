"""
Duplicate Video Detector
=========================

Finds videos with similar transcripts for potential cleanup/deletion.

Key Features:
- Transcript-based similarity detection using SequenceMatcher
- Caption protection: Prioritizes keeping captioned versions
- Duration-based tiebreaker: Keeps longer video when captions equal
- Configurable similarity threshold (default 85%)

Protection Rules:
1. If one video has captions and other doesn't → KEEP captioned version
2. If both have same caption status → KEEP longer video  
3. If same duration and caption status → Flag for manual review

Usage:
    detector = DuplicateDetector(db_session)
    duplicates = await detector.find_duplicates(similarity_threshold=0.85)
    
API Endpoints:
- GET /api/duplicates/find - Find video duplicates
- GET /api/duplicates/find-image-duplicates - Find image duplicates
- POST /api/duplicates/auto-detect-and-recommend - Auto-detect with caption protection
- DELETE /api/duplicates/execute-deletion - Execute marked deletions
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Video, VideoAnalysis

logger = logging.getLogger(__name__)


@dataclass
class DuplicatePair:
    """A pair of videos that appear to be duplicates"""
    video1_id: str
    video1_filename: str
    video1_duration: float
    video1_has_captions: bool
    video2_id: str
    video2_filename: str
    video2_duration: float
    video2_has_captions: bool
    similarity_score: float
    transcript_preview: str
    recommendation: str  # "keep_with_captions", "keep_longer", "review_manually"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video1": {
                "id": self.video1_id,
                "filename": self.video1_filename,
                "duration_sec": self.video1_duration,
                "has_captions": self.video1_has_captions,
            },
            "video2": {
                "id": self.video2_id,
                "filename": self.video2_filename,
                "duration_sec": self.video2_duration,
                "has_captions": self.video2_has_captions,
            },
            "similarity_score": round(self.similarity_score, 3),
            "transcript_preview": self.transcript_preview,
            "recommendation": self.recommendation,
        }


class DuplicateDetector:
    """Detects duplicate videos based on transcript similarity"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    def _normalize_transcript(self, transcript: str) -> str:
        """Normalize transcript for comparison"""
        if not transcript:
            return ""
        # Remove extra whitespace, lowercase, strip punctuation
        normalized = " ".join(transcript.lower().split())
        # Remove common punctuation
        for char in ".,!?;:'\"()-":
            normalized = normalized.replace(char, "")
        return normalized
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts (0-1)"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _get_recommendation(
        self, 
        video1_has_captions: bool, 
        video2_has_captions: bool,
        video1_duration: float,
        video2_duration: float
    ) -> Tuple[str, str]:
        """
        Determine which video to keep and why.
        Returns (recommendation, delete_candidate_id)
        """
        # If one has captions and one doesn't, prefer the one WITH captions
        if video1_has_captions and not video2_has_captions:
            return "keep_video1_has_captions", "video2"
        elif video2_has_captions and not video1_has_captions:
            return "keep_video2_has_captions", "video1"
        
        # Both have captions or both don't - prefer longer duration
        if abs(video1_duration - video2_duration) > 1:  # More than 1 second difference
            if video1_duration > video2_duration:
                return "keep_video1_longer", "video2"
            else:
                return "keep_video2_longer", "video1"
        
        # Similar duration, need manual review
        return "review_manually", "either"
    
    async def find_duplicates(
        self,
        similarity_threshold: float = 0.85,
        min_transcript_length: int = 50,
        limit: int = 100,
        compare_same_caption_status_only: bool = True,
    ) -> List[DuplicatePair]:
        """
        Find videos with similar transcripts.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) to consider as duplicate
            min_transcript_length: Minimum transcript length to consider
            limit: Maximum number of duplicate pairs to return
            compare_same_caption_status_only: If True, only compare videos with same caption status
        
        Returns:
            List of DuplicatePair objects
        """
        # Fetch all videos with transcripts
        query = text("""
            SELECT 
                v.id,
                v.file_name,
                v.duration_sec,
                va.transcript,
                COALESCE(va.visual_analysis->>'has_embedded_captions', 'false') as has_captions
            FROM videos v
            JOIN video_analysis va ON v.id = va.video_id
            WHERE va.transcript IS NOT NULL 
              AND LENGTH(va.transcript) > :min_length
            ORDER BY v.created_at DESC
            LIMIT 500
        """)
        
        result = await self.db.execute(query, {"min_length": min_transcript_length})
        videos = result.fetchall()
        
        logger.info(f"[DuplicateDetector] Checking {len(videos)} videos for duplicates (threshold: {similarity_threshold:.0%})")
        
        # Build list of video data
        video_data = []
        for row in videos:
            video_data.append({
                "id": str(row[0]),
                "filename": row[1] or "",
                "duration": row[2] or 0,
                "transcript": row[3] or "",
                "normalized_transcript": self._normalize_transcript(row[3]),
                "has_captions": row[4] == 'true' or row[4] == True,
            })
        
        # Compare all pairs
        duplicates = []
        seen_pairs = set()
        
        for i, v1 in enumerate(video_data):
            if len(duplicates) >= limit:
                break
                
            for j, v2 in enumerate(video_data[i+1:], i+1):
                if len(duplicates) >= limit:
                    break
                
                # Skip if we've already seen this pair
                pair_key = tuple(sorted([v1["id"], v2["id"]]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                # Skip if comparing different caption statuses (when flag is set)
                if compare_same_caption_status_only:
                    if v1["has_captions"] != v2["has_captions"]:
                        continue
                
                # Skip if transcripts are too different in length (likely not duplicates)
                len_ratio = min(len(v1["normalized_transcript"]), len(v2["normalized_transcript"])) / \
                           max(len(v1["normalized_transcript"]), len(v2["normalized_transcript"]), 1)
                if len_ratio < 0.5:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(
                    v1["normalized_transcript"], 
                    v2["normalized_transcript"]
                )
                
                if similarity >= similarity_threshold:
                    recommendation, _ = self._get_recommendation(
                        v1["has_captions"],
                        v2["has_captions"],
                        v1["duration"],
                        v2["duration"]
                    )
                    
                    # Create preview (first 100 chars of transcript)
                    preview = v1["transcript"][:100] + "..." if len(v1["transcript"]) > 100 else v1["transcript"]
                    
                    duplicates.append(DuplicatePair(
                        video1_id=v1["id"],
                        video1_filename=v1["filename"],
                        video1_duration=v1["duration"],
                        video1_has_captions=v1["has_captions"],
                        video2_id=v2["id"],
                        video2_filename=v2["filename"],
                        video2_duration=v2["duration"],
                        video2_has_captions=v2["has_captions"],
                        similarity_score=similarity,
                        transcript_preview=preview,
                        recommendation=recommendation,
                    ))
        
        # Sort by similarity (highest first)
        duplicates.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Log results
        if duplicates:
            caption_protected = sum(1 for d in duplicates if 'captions' in d.recommendation)
            logger.info(f"[DuplicateDetector] Found {len(duplicates)} duplicate pairs ({caption_protected} with caption protection)")
        else:
            logger.info(f"[DuplicateDetector] No duplicates found above {similarity_threshold:.0%} threshold")
        
        return duplicates[:limit]
    
    async def find_exact_duplicates(self, limit: int = 50) -> List[DuplicatePair]:
        """Find videos with exactly matching transcripts (100% similarity)"""
        return await self.find_duplicates(
            similarity_threshold=0.99,
            limit=limit,
            compare_same_caption_status_only=False  # Include all for exact matches
        )
    
    async def find_near_duplicates(self, limit: int = 50) -> List[DuplicatePair]:
        """Find videos with very similar transcripts (85-99% similarity)"""
        return await self.find_duplicates(
            similarity_threshold=0.85,
            limit=limit,
            compare_same_caption_status_only=True
        )
