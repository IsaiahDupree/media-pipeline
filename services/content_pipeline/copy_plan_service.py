"""
CopyPlan Service
Generates platform-optimized copy (titles, captions, descriptions) from video analysis.
Enforces character limits with 80% target margin.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, field, asdict
from loguru import logger

from openai import OpenAI
from sqlalchemy import create_engine, text

from .text_utils import (
    count_by_rule, compute_target, truncate_smart, 
    format_hashtags, validate_text_fits
)


Platform = Literal["youtube", "instagram", "tiktok", "facebook", "threads", "x", "linkedin", "pinterest", "bluesky", "other"]
Surface = Literal["video", "short", "reel", "feed", "post", "pin", "standard_post", "long_post", "other"]
TextField = Literal["title", "description", "caption"]
CountRule = Literal["graphemes", "utf16", "utf8_bytes"]


@dataclass
class TextConstraint:
    platform: Platform
    surface: Surface
    field: TextField
    max_chars: Optional[int] = None
    soft_cap_chars: Optional[int] = None
    target_margin_pct: float = 0.20
    count_rule: CountRule = "graphemes"
    max_hashtags: Optional[int] = None
    max_mentions: Optional[int] = None


@dataclass
class CopyVariant:
    text: str
    char_count: int
    target_chars: Optional[int] = None
    max_chars: Optional[int] = None
    fits: bool = True
    rationale: Optional[str] = None


@dataclass
class CopyPlanInput:
    hook: str
    topics: List[str]
    keywords: List[str] = field(default_factory=list)
    audience: List[str] = field(default_factory=list)
    cta: Optional[Dict[str, str]] = None
    tone: Optional[str] = None
    platform_intent: Optional[str] = None
    pain_points: List[str] = field(default_factory=list)
    emotional_drivers: List[str] = field(default_factory=list)
    content_type: Optional[str] = None


@dataclass
class PlatformCopyVariant:
    platform: Platform
    surface: Surface
    constraints: Dict[str, Any]
    title_variants: Optional[List[CopyVariant]] = None
    description: Optional[CopyVariant] = None
    caption: Optional[CopyVariant] = None
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    generation_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CopyPlanV1:
    schema: str = "copy_plan_v1"
    inputs: CopyPlanInput = None
    variants: List[PlatformCopyVariant] = field(default_factory=list)


class CopyPlanService:
    """
    Service for generating platform-optimized copy from video analysis.
    """
    
    # Platform-surface configurations
    PLATFORM_SURFACES = {
        "youtube": ["video", "short"],
        "instagram": ["reel", "feed"],
        "tiktok": ["video"],
        "facebook": ["post", "reel"],
        "threads": ["post"],
        "x": ["standard_post"],
        "linkedin": ["post"],
        "pinterest": ["pin"],
        "bluesky": ["post"],
    }
    
    def __init__(self, db_url: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres")
        self.engine = create_engine(self.db_url)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        
        # Cache constraints
        self._constraints_cache: Dict[str, TextConstraint] = {}
        self._load_constraints()
    
    def _load_constraints(self):
        """Load platform text constraints from database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT platform, surface, field, max_chars, soft_cap_chars, 
                           target_margin_pct, count_rule, max_hashtags, max_mentions
                    FROM platform_text_constraints
                """))
                for row in result:
                    key = f"{row[0]}:{row[1]}:{row[2]}"
                    self._constraints_cache[key] = TextConstraint(
                        platform=row[0],
                        surface=row[1],
                        field=row[2],
                        max_chars=row[3],
                        soft_cap_chars=row[4],
                        target_margin_pct=float(row[5]) if row[5] else 0.20,
                        count_rule=row[6] or "graphemes",
                        max_hashtags=row[7],
                        max_mentions=row[8]
                    )
            logger.info(f"Loaded {len(self._constraints_cache)} platform constraints")
        except Exception as e:
            logger.warning(f"Could not load constraints from DB: {e}")
    
    def get_constraint(self, platform: Platform, surface: Surface, field: TextField) -> Optional[TextConstraint]:
        """Get constraint for platform/surface/field combination"""
        key = f"{platform}:{surface}:{field}"
        return self._constraints_cache.get(key)
    
    def build_copy_variant(
        self, 
        raw_text: str, 
        constraint: Optional[TextConstraint]
    ) -> CopyVariant:
        """Build a copy variant with enforcement and character counting"""
        rule = constraint.count_rule if constraint else "graphemes"
        max_chars = constraint.max_chars if constraint else None
        target_chars = compute_target(
            max_chars, 
            constraint.target_margin_pct if constraint else 0.20,
            constraint.soft_cap_chars if constraint else None
        )
        
        # Truncate if needed
        final_text = truncate_smart(raw_text, target_chars, rule) if target_chars else raw_text
        char_count = count_by_rule(final_text, rule)
        fits = max_chars is None or char_count <= max_chars
        
        return CopyVariant(
            text=final_text,
            char_count=char_count,
            target_chars=target_chars,
            max_chars=max_chars,
            fits=fits
        )
    
    async def generate_copy_for_platform(
        self,
        inputs: CopyPlanInput,
        platform: Platform,
        surface: Surface
    ) -> PlatformCopyVariant:
        """Generate platform-specific copy using AI"""
        
        # Get constraints
        title_constraint = self.get_constraint(platform, surface, "title")
        caption_constraint = self.get_constraint(platform, surface, "caption")
        desc_constraint = self.get_constraint(platform, surface, "description")
        
        # Compute targets
        title_target = compute_target(
            title_constraint.max_chars if title_constraint else None,
            title_constraint.target_margin_pct if title_constraint else 0.20,
            title_constraint.soft_cap_chars if title_constraint else None
        )
        caption_target = compute_target(
            caption_constraint.max_chars if caption_constraint else None,
            caption_constraint.target_margin_pct if caption_constraint else 0.20,
            caption_constraint.soft_cap_chars if caption_constraint else None
        )
        desc_target = compute_target(
            desc_constraint.max_chars if desc_constraint else None,
            desc_constraint.target_margin_pct if desc_constraint else 0.20,
            desc_constraint.soft_cap_chars if desc_constraint else None
        )
        
        # Build prompt
        prompt = self._build_generation_prompt(
            inputs, platform, surface,
            title_constraint, caption_constraint, desc_constraint,
            title_target, caption_target, desc_target
        )
        
        # Call LLM
        raw_response = await self._call_llm(prompt)
        
        # Build variants with enforcement
        result = PlatformCopyVariant(
            platform=platform,
            surface=surface,
            constraints={
                "title": asdict(title_constraint) if title_constraint else None,
                "caption": asdict(caption_constraint) if caption_constraint else None,
                "description": asdict(desc_constraint) if desc_constraint else None,
                "computed_targets": {
                    "title_target": title_target,
                    "caption_target": caption_target,
                    "description_target": desc_target
                }
            },
            generation_meta={
                "model": "gpt-4o-mini",
                "prompt_version": "copy_plan_v1",
                "generated_at_iso": datetime.utcnow().isoformat()
            }
        )
        
        # Process titles (for YouTube, Pinterest)
        if raw_response.get("titles"):
            result.title_variants = [
                self.build_copy_variant(t, title_constraint)
                for t in raw_response["titles"][:3]
            ]
        
        # Process caption
        if raw_response.get("caption"):
            result.caption = self.build_copy_variant(
                raw_response["caption"], caption_constraint
            )
        
        # Process description
        if raw_response.get("description"):
            result.description = self.build_copy_variant(
                raw_response["description"], desc_constraint
            )
        
        # Process hashtags
        if raw_response.get("hashtags"):
            max_tags = caption_constraint.max_hashtags if caption_constraint else 30
            result.hashtags = raw_response["hashtags"][:max_tags]
        
        # Process mentions
        if raw_response.get("mentions"):
            max_mentions = caption_constraint.max_mentions if caption_constraint else 20
            result.mentions = raw_response["mentions"][:max_mentions]
        
        return result
    
    def _build_generation_prompt(
        self,
        inputs: CopyPlanInput,
        platform: Platform,
        surface: Surface,
        title_constraint: Optional[TextConstraint],
        caption_constraint: Optional[TextConstraint],
        desc_constraint: Optional[TextConstraint],
        title_target: Optional[int],
        caption_target: Optional[int],
        desc_target: Optional[int]
    ) -> str:
        """Build LLM prompt for copy generation"""
        
        prompt = f"""Generate platform-specific social media copy optimized for engagement.

PLATFORM: {platform}
SURFACE: {surface}

CHARACTER LIMITS (aim for target, never exceed max):
- Title max: {title_constraint.max_chars if title_constraint else 'N/A'}, target: {title_target or 'N/A'}
- Caption max: {caption_constraint.max_chars if caption_constraint else 'N/A'}, target: {caption_target or 'N/A'}
- Description max: {desc_constraint.max_chars if desc_constraint else 'N/A'}, target: {desc_target or 'N/A'}

CONTENT INPUTS:
- HOOK: {inputs.hook}
- TOPICS: {', '.join(inputs.topics)}
- KEYWORDS: {', '.join(inputs.keywords) if inputs.keywords else 'None'}
- TARGET AUDIENCE: {', '.join(inputs.audience) if inputs.audience else 'General'}
- TONE: {inputs.tone or 'Engaging, authentic'}
- CONTENT TYPE: {inputs.content_type or 'General'}
- PAIN POINTS: {', '.join(inputs.pain_points) if inputs.pain_points else 'None'}
- EMOTIONAL DRIVERS: {', '.join(inputs.emotional_drivers) if inputs.emotional_drivers else 'None'}
- CTA: {inputs.cta['type'] + ': ' + inputs.cta['text'] if inputs.cta else 'Follow for more'}

PLATFORM-SPECIFIC GUIDELINES:
"""
        
        # Add platform-specific guidance
        platform_tips = {
            "youtube": "- Titles should be searchable and click-worthy\n- Include relevant keywords in description\n- Front-load important info",
            "instagram": "- Captions can be longer but hook in first line\n- Strategic hashtag placement\n- Emojis work well",
            "tiktok": "- Short, punchy captions\n- Trending hashtags important\n- Keep it casual",
            "x": "- Brevity is key (280 chars for standard)\n- No hashtag overload\n- Drive engagement with questions",
            "linkedin": "- Professional tone\n- Value-driven content\n- Longer form accepted",
            "threads": "- Conversational, authentic\n- Limited characters (500)\n- Minimal hashtags",
            "pinterest": "- Descriptive, searchable titles\n- Include relevant keywords\n- Call-to-action helpful",
        }
        prompt += platform_tips.get(platform, "- Follow platform best practices")
        
        prompt += """

Return JSON with ONLY fields relevant to this platform:
{
  "titles": ["(if platform uses titles) 2-3 variants"],
  "caption": "(if platform uses captions)",
  "description": "(if platform uses descriptions)",
  "hashtags": ["relevant", "hashtags", "without #"],
  "mentions": [],
  "rationale": "Brief explanation of approach"
}

CRITICAL: Stay within character targets. Quality over quantity."""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI for copy generation"""
        if not self.client:
            logger.warning("No OpenAI client - returning mock response")
            return self._mock_response()
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert social media copywriter. Generate engaging, platform-optimized copy that fits within character limits."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._mock_response()
    
    def _mock_response(self) -> Dict[str, Any]:
        """Return mock response when LLM unavailable"""
        return {
            "titles": ["Engaging Title Here"],
            "caption": "Check this out! 🔥",
            "description": "Full description of the content.",
            "hashtags": ["content", "creator"],
            "mentions": []
        }
    
    async def generate_copy_plan(
        self,
        inputs: CopyPlanInput,
        platforms: List[Platform],
        asset_id: Optional[str] = None,
        audit_id: Optional[str] = None
    ) -> CopyPlanV1:
        """
        Generate complete copy plan for multiple platforms.
        
        Args:
            inputs: Copy generation inputs from video analysis
            platforms: List of platforms to generate for
            asset_id: Optional content asset ID
            audit_id: Optional deep audit ID
        
        Returns:
            Complete CopyPlanV1 with all platform variants
        """
        plan = CopyPlanV1(inputs=inputs, variants=[])
        
        for platform in platforms:
            surfaces = self.PLATFORM_SURFACES.get(platform, ["post"])
            for surface in surfaces:
                try:
                    variant = await self.generate_copy_for_platform(inputs, platform, surface)
                    plan.variants.append(variant)
                    logger.info(f"Generated copy for {platform}/{surface}")
                except Exception as e:
                    logger.error(f"Failed to generate copy for {platform}/{surface}: {e}")
        
        # Save to database
        if asset_id or audit_id:
            await self._save_copy_plan(plan, asset_id, audit_id)
        
        return plan
    
    async def _save_copy_plan(
        self,
        plan: CopyPlanV1,
        asset_id: Optional[str],
        audit_id: Optional[str]
    ):
        """Save copy plan to database"""
        try:
            with self.engine.connect() as conn:
                for variant in plan.variants:
                    conn.execute(text("""
                        INSERT INTO copy_plan (
                            asset_id, audit_id, platform, surface, data,
                            title, caption, description, hashtags, mentions,
                            model, prompt_version
                        ) VALUES (
                            :asset_id, :audit_id, :platform, :surface, :data,
                            :title, :caption, :description, :hashtags, :mentions,
                            :model, :prompt_version
                        )
                    """), {
                        "asset_id": asset_id,
                        "audit_id": audit_id,
                        "platform": variant.platform,
                        "surface": variant.surface,
                        "data": json.dumps(asdict(variant)),
                        "title": variant.title_variants[0].text if variant.title_variants else None,
                        "caption": variant.caption.text if variant.caption else None,
                        "description": variant.description.text if variant.description else None,
                        "hashtags": variant.hashtags,
                        "mentions": variant.mentions,
                        "model": variant.generation_meta.get("model"),
                        "prompt_version": variant.generation_meta.get("prompt_version")
                    })
                conn.commit()
                logger.info(f"Saved {len(plan.variants)} copy plan variants to database")
        except Exception as e:
            logger.error(f"Failed to save copy plan: {e}")
    
    @classmethod
    def from_video_analysis(cls, analysis: Dict[str, Any]) -> CopyPlanInput:
        """
        Create CopyPlanInput from video analysis results.
        Maps analysis fields to copy generation inputs.
        """
        return CopyPlanInput(
            hook=analysis.get("detected_hook") or (analysis.get("hooks", [""])[0] if analysis.get("hooks") else ""),
            topics=analysis.get("topics", []),
            keywords=analysis.get("topics", [])[:5],  # Use topics as keywords
            audience=[analysis.get("target_audience", {}).get("demographic", "")] if analysis.get("target_audience") else [],
            cta=analysis.get("call_to_action"),
            tone=analysis.get("tone"),
            platform_intent=None,
            pain_points=analysis.get("pain_points", []),
            emotional_drivers=analysis.get("emotional_drivers", []),
            content_type=analysis.get("content_type")
        )
