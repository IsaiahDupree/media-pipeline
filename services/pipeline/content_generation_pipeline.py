"""
Content Generation Pipeline (OPS-008)
=====================================
Generate AI content aligned to Awareness × FATE frameworks with full attribution.

Pipeline Flow:
1. Receive generation request (template, offer, ICP, awareness level)
2. Build prompt from template + inputs
3. Call OpenAI API (REAL calls, no mocks)
4. Generate N variants
5. Score via FATE
6. Classify awareness level
7. Return with full traceback

Entities:
- Template: Awareness level, FATE weights, format, CTA strength
- Offer: Promise, CTAs, landing URL
- ICP: Pains, outcomes, objections, language patterns
- PromptRun: Template + inputs + model → generated text
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from uuid import uuid4
from loguru import logger
from pydantic import BaseModel, Field
import openai

# Services
from services.fate_scorer import get_fate_scorer
from services.awareness_classifier import get_awareness_classifier


class GenerationRequest(BaseModel):
    """Request to generate content"""
    template_id: str = Field(..., description="Template ID to use")
    offer_id: str = Field(..., description="Offer being promoted")
    icp_id: str = Field(..., description="Target ICP")
    awareness_level: int = Field(..., ge=1, le=5, description="Target awareness level (1-5)")
    channel: str = Field(..., description="Channel: post|comment|dm|email")
    platform: str = Field(..., description="Platform: x|instagram|tiktok|youtube|threads")
    variants: int = Field(default=3, ge=1, le=5, description="Number of variants to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="OpenAI temperature")
    model: str = Field(default="gpt-4o", description="OpenAI model to use")
    inputs: Optional[Dict[str, Any]] = Field(default=None, description="Additional template inputs")


class PromptRun(BaseModel):
    """Record of a prompt execution"""
    prompt_run_id: str
    template_id: str
    offer_id: str
    icp_id: str
    awareness_level: int
    channel: str
    platform: str
    inputs: Dict[str, Any]
    llm_config: Dict[str, Any]  # Renamed from model_config to avoid Pydantic conflict
    prompt_text: str
    output_text: str
    fate_scores: Dict[str, float]
    classified_awareness: int
    created_at: str


class GenerationResult(BaseModel):
    """Result of content generation"""
    success: bool
    variants: List[PromptRun] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContentGenerationPipeline:
    """
    Content Generation Pipeline

    Generates AI content with full FATE scoring and awareness classification.
    Maintains complete attribution chain: content → prompt → template → offer → ICP.

    Usage:
        pipeline = ContentGenerationPipeline.get_instance()
        request = GenerationRequest(
            template_id="tpl_problem_aware_001",
            offer_id="offer_keyword_radar",
            icp_id="icp_indie_founders",
            awareness_level=2,
            channel="post",
            platform="x",
            variants=3
        )
        result = await pipeline.generate(request)
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> "ContentGenerationPipeline":
        """Get singleton instance of ContentGenerationPipeline"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the content generation pipeline"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("⚠️  OPENAI_API_KEY not set - generation will fail")

        openai.api_key = self.openai_api_key

        # Services
        self.fate_scorer = get_fate_scorer()
        self.awareness_classifier = get_awareness_classifier()

        logger.info("📝 Content Generation Pipeline initialized")

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate content variants with full attribution

        Args:
            request: Generation request with template, offer, ICP, etc.

        Returns:
            GenerationResult with variants and FATE scores
        """
        try:
            logger.info(
                f"🎨 Generating {request.variants} variants | "
                f"Template: {request.template_id} | "
                f"Offer: {request.offer_id} | "
                f"ICP: {request.icp_id} | "
                f"Awareness: {request.awareness_level}"
            )

            # TODO: Load actual template from database
            # For now, use a placeholder template system
            template_data = await self._get_template(request.template_id)
            offer_data = await self._get_offer(request.offer_id)
            icp_data = await self._get_icp(request.icp_id)

            # Build prompt
            prompt = self._build_prompt(
                template=template_data,
                offer=offer_data,
                icp=icp_data,
                awareness_level=request.awareness_level,
                channel=request.channel,
                platform=request.platform,
                inputs=request.inputs or {}
            )

            logger.debug(f"📝 Prompt built ({len(prompt)} chars)")

            # Generate variants
            variants = []
            for i in range(request.variants):
                try:
                    # CRITICAL: Real OpenAI API call (no mocks)
                    logger.info(f"🤖 Calling OpenAI API (variant {i+1}/{request.variants})...")

                    response = await self._call_openai(
                        prompt=prompt,
                        model=request.model,
                        temperature=request.temperature
                    )

                    generated_text = response.strip()

                    # Score with FATE
                    fate_scores = self.fate_scorer.score_all(generated_text)

                    # Classify awareness
                    classified_awareness = self.awareness_classifier.classify(generated_text)

                    # Create prompt run record
                    prompt_run = PromptRun(
                        prompt_run_id=f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
                        template_id=request.template_id,
                        offer_id=request.offer_id,
                        icp_id=request.icp_id,
                        awareness_level=request.awareness_level,
                        channel=request.channel,
                        platform=request.platform,
                        inputs=request.inputs or {},
                        llm_config={
                            "name": request.model,
                            "temperature": request.temperature
                        },
                        prompt_text=prompt,
                        output_text=generated_text,
                        fate_scores=fate_scores,
                        classified_awareness=classified_awareness,
                        created_at=datetime.now(timezone.utc).isoformat()
                    )

                    variants.append(prompt_run)

                    logger.success(
                        f"✓ Variant {i+1} generated | "
                        f"FATE: F={fate_scores['F']:.2f} A={fate_scores['A']:.2f} "
                        f"T={fate_scores['T']:.2f} E={fate_scores['E']:.2f} | "
                        f"Awareness: {classified_awareness}"
                    )

                except Exception as e:
                    logger.error(f"Failed to generate variant {i+1}: {e}")
                    # Continue with other variants

            if not variants:
                return GenerationResult(
                    success=False,
                    error="Failed to generate any variants"
                )

            logger.success(f"✅ Generated {len(variants)} variants successfully")

            return GenerationResult(
                success=True,
                variants=variants,
                metadata={
                    "template_id": request.template_id,
                    "offer_id": request.offer_id,
                    "icp_id": request.icp_id,
                    "awareness_level": request.awareness_level,
                    "requested_variants": request.variants,
                    "generated_variants": len(variants)
                }
            )

        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return GenerationResult(
                success=False,
                error=str(e)
            )

    async def _call_openai(self, prompt: str, model: str, temperature: float) -> str:
        """
        Call OpenAI API (REAL call, no mocks)

        Args:
            prompt: The prompt to send
            model: OpenAI model name
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        # Use new OpenAI client
        client = openai.OpenAI(api_key=self.openai_api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert copywriter who creates persuasive content using the FATE framework (Focus, Authority, Tribe, Emotion)."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )

        return response.choices[0].message.content

    def _build_prompt(
        self,
        template: Dict[str, Any],
        offer: Dict[str, Any],
        icp: Dict[str, Any],
        awareness_level: int,
        channel: str,
        platform: str,
        inputs: Dict[str, Any]
    ) -> str:
        """
        Build prompt from template + context

        Args:
            template: Template data
            offer: Offer data
            icp: ICP data
            awareness_level: Target awareness level (1-5)
            channel: Channel type
            platform: Platform name
            inputs: Additional inputs

        Returns:
            Complete prompt text
        """
        # Awareness level copy patterns
        awareness_patterns = {
            1: "Unaware - Surface symptom via story, make them curious",
            2: "Problem-Aware - Mirror their pain, clarify the cause",
            3: "Solution-Aware - Educate on approaches, compare options",
            4: "Product-Aware - Differentiate, handle objections",
            5: "Most-Aware - Offer + urgency + remove friction"
        }

        # Build prompt
        prompt = f"""Create a {channel} for {platform} targeting {awareness_patterns[awareness_level]}.

OFFER:
{offer.get('promise', 'N/A')}
Landing URL: {offer.get('landing_url', 'N/A')}
CTA: {offer.get('cta', 'Learn more')}

TARGET AUDIENCE (ICP):
{icp.get('description', 'General audience')}
Pains: {', '.join(icp.get('pains', []))}
Desired Outcomes: {', '.join(icp.get('outcomes', []))}

TEMPLATE:
{template.get('structure', 'Write compelling copy')}

CONSTRAINTS:
- Platform: {platform}
- Channel: {channel}
- Awareness Level: {awareness_level} ({awareness_patterns[awareness_level]})
- Use FATE framework:
  * Focus (F): Pattern interrupt, curiosity gap, stakes
  * Authority (A): Numbers, proof, mechanism
  * Tribe (T): Identity markers, us-vs-them
  * Emotion (E): Story beats, contrast, loss aversion, hope
- Keep it concise and compelling
- End with a clear CTA

Additional context:
{self._format_inputs(inputs)}

Generate the {channel} now:"""

        return prompt

    def _format_inputs(self, inputs: Dict[str, Any]) -> str:
        """Format additional inputs for prompt"""
        if not inputs:
            return "None"

        lines = []
        for key, value in inputs.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    async def _get_template(self, template_id: str) -> Dict[str, Any]:
        """
        Get template data

        TODO: Load from database
        For now, return placeholder
        """
        return {
            "template_id": template_id,
            "awareness_level": 2,
            "structure": "Hook → Problem → Mechanism → CTA",
            "fate_weights": {"F": 0.8, "A": 0.7, "T": 0.6, "E": 0.9}
        }

    async def _get_offer(self, offer_id: str) -> Dict[str, Any]:
        """
        Get offer data

        TODO: Load from database via ENTITY-002 API
        For now, return placeholder
        """
        return {
            "offer_id": offer_id,
            "promise": "Find profitable keywords for your SaaS in minutes, not days",
            "landing_url": "https://keywordradar.app",
            "cta": "Try KeywordRadar free",
            "who_for": "Indie founders building SaaS products",
            "who_not_for": "Enterprise teams with dedicated SEO staff"
        }

    async def _get_icp(self, icp_id: str) -> Dict[str, Any]:
        """
        Get ICP data

        TODO: Load from database via ENTITY-003 API
        For now, return placeholder
        """
        return {
            "icp_id": icp_id,
            "description": "Indie founders who launched a SaaS but struggle with SEO and keyword research",
            "pains": [
                "Wasting hours on keyword research tools",
                "Not sure which keywords to target",
                "Missing obvious opportunities competitors found"
            ],
            "outcomes": [
                "Find profitable keywords quickly",
                "Validate SEO strategy",
                "Outrank competitors"
            ],
            "objections": [
                "Too expensive",
                "Too complicated",
                "Already have Ahrefs/Semrush"
            ],
            "language_patterns": [
                "bootstrapper",
                "indie hacker",
                "maker",
                "solo founder"
            ]
        }


# Singleton
_pipeline_instance = None

def get_content_generation_pipeline() -> ContentGenerationPipeline:
    """Get singleton content generation pipeline"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = ContentGenerationPipeline()
    return _pipeline_instance
