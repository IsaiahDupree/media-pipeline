"""
Background Removal Service (CHAR-002)
=====================================
Automated background removal for transparent PNGs.

Uses rembg library with AI models to remove backgrounds from images,
making them suitable for video overlays.

Features:
- Remove background from character images
- Support multiple AI models (u2net, u2netp, u2net_human_seg, etc.)
- Batch processing
- Quality validation

Usage:
    from services.background_removal import BackgroundRemovalService

    remover = BackgroundRemovalService()

    # Remove background from a character
    await remover.remove_background(
        input_path="/path/to/character.png",
        output_path="/path/to/character_transparent.png"
    )

    # Process character asset
    await remover.process_character_asset(character_id=uuid)
"""

import asyncio
import os
from pathlib import Path
from typing import Optional, Literal
from uuid import UUID

from loguru import logger
from PIL import Image
from sqlalchemy import select

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    logger.warning("rembg library not installed. Background removal will not be available.")

from config import get_settings
from database.connection import async_session_maker
from database.models import CharacterAsset, CharacterVariant


ModelType = Literal["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta"]


class BackgroundRemovalService:
    """
    Background Removal Service

    Removes backgrounds from images to create transparent PNGs
    suitable for video overlays.
    """

    def __init__(self):
        """Initialize background removal service"""
        self.settings = get_settings()
        self.session_maker = async_session_maker

        if not REMBG_AVAILABLE:
            logger.error("❌ rembg library not available. Install with: pip install rembg[gpu]")

    async def remove_background(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        model: ModelType = "u2net",
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10
    ) -> str:
        """
        Remove background from an image

        Args:
            input_path: Path to input image
            output_path: Path for output (auto-generated if None)
            model: AI model to use for segmentation
            alpha_matting: Enable alpha matting for better edges
            alpha_matting_foreground_threshold: Foreground threshold
            alpha_matting_background_threshold: Background threshold
            alpha_matting_erode_size: Erosion size

        Returns:
            Path to output image with transparent background

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If rembg is not available or processing fails
        """
        if not REMBG_AVAILABLE:
            raise RuntimeError("rembg library not installed. Cannot remove background.")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_transparent.png")

        logger.info(f"🎨 Removing background: {input_path} → {output_path}")

        try:
            # Load input image
            input_image = Image.open(input_path)

            # Run background removal in executor to avoid blocking
            loop = asyncio.get_event_loop()
            output_image = await loop.run_in_executor(
                None,
                lambda: rembg_remove(
                    input_image,
                    model_name=model,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size
                )
            )

            # Save output
            output_image.save(output_path, format="PNG")

            # Verify transparency
            has_transparency = self._verify_transparency(output_path)
            if has_transparency:
                logger.success(f"✓ Background removed: {output_path}")
            else:
                logger.warning(f"⚠️  Background removal may have failed (no transparency detected)")

            return output_path

        except Exception as e:
            logger.error(f"Failed to remove background: {e}")
            raise RuntimeError(f"Background removal failed: {e}") from e

    async def process_character_asset(
        self,
        character_id: UUID,
        model: ModelType = "u2net",
        alpha_matting: bool = True,
        overwrite: bool = False
    ) -> str:
        """
        Process a character asset to remove background

        Updates the character's has_transparent_background flag
        and creates a new file with transparent background.

        Args:
            character_id: Character asset ID
            model: AI model to use
            alpha_matting: Enable alpha matting for better edges
            overwrite: Overwrite existing transparent version

        Returns:
            Path to transparent image

        Raises:
            ValueError: If character not found
            RuntimeError: If processing fails
        """
        logger.info(f"🎨 Processing character asset: {character_id}")

        # Load character
        async with self.session_maker() as session:
            character = await session.get(CharacterAsset, character_id)
            if not character:
                raise ValueError(f"Character not found: {character_id}")

            # Check if already processed
            if character.has_transparent_background and not overwrite:
                logger.info(f"Character already has transparent background")
                return character.file_path

            # Generate output path
            input_path = character.file_path
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_transparent.png")

            # Skip if output exists and not overwriting
            if os.path.exists(output_path) and not overwrite:
                logger.info(f"Transparent version already exists: {output_path}")
                # Update database
                character.has_transparent_background = True
                await session.commit()
                return output_path

        # Remove background
        transparent_path = await self.remove_background(
            input_path=input_path,
            output_path=output_path,
            model=model,
            alpha_matting=alpha_matting
        )

        # Update character in database
        async with self.session_maker() as session:
            character = await session.get(CharacterAsset, character_id)
            if character:
                # Update to use transparent version
                character.file_path = transparent_path
                character.has_transparent_background = True

                await session.commit()

                logger.success(f"✓ Character asset updated with transparent background")

        return transparent_path

    async def process_character_variant(
        self,
        variant_id: UUID,
        model: ModelType = "u2net",
        alpha_matting: bool = True,
        overwrite: bool = False
    ) -> str:
        """
        Process a character variant to remove background

        Args:
            variant_id: Variant ID
            model: AI model to use
            alpha_matting: Enable alpha matting
            overwrite: Overwrite existing

        Returns:
            Path to transparent image

        Raises:
            ValueError: If variant not found
            RuntimeError: If processing fails
        """
        logger.info(f"🎨 Processing character variant: {variant_id}")

        # Load variant
        async with self.session_maker() as session:
            variant = await session.get(CharacterVariant, variant_id)
            if not variant:
                raise ValueError(f"Variant not found: {variant_id}")

            if variant.has_transparent_background and not overwrite:
                logger.info(f"Variant already has transparent background")
                return variant.file_path

            input_path = variant.file_path
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_transparent.png")

            if os.path.exists(output_path) and not overwrite:
                logger.info(f"Transparent version already exists")
                variant.has_transparent_background = True
                await session.commit()
                return output_path

        # Remove background
        transparent_path = await self.remove_background(
            input_path=input_path,
            output_path=output_path,
            model=model,
            alpha_matting=alpha_matting
        )

        # Update variant
        async with self.session_maker() as session:
            variant = await session.get(CharacterVariant, variant_id)
            if variant:
                variant.file_path = transparent_path
                variant.has_transparent_background = True
                await session.commit()

        return transparent_path

    async def batch_process_character(
        self,
        character_id: UUID,
        model: ModelType = "u2net",
        alpha_matting: bool = True,
        process_variants: bool = True
    ) -> dict:
        """
        Process character and all its variants

        Args:
            character_id: Character ID
            model: AI model to use
            alpha_matting: Enable alpha matting
            process_variants: Also process all variants

        Returns:
            Dict with processing results
        """
        logger.info(f"🎨 Batch processing character: {character_id}")

        results = {
            "character_id": str(character_id),
            "base_processed": False,
            "variants_processed": 0,
            "variants_failed": 0,
            "errors": []
        }

        # Process base character
        try:
            await self.process_character_asset(
                character_id=character_id,
                model=model,
                alpha_matting=alpha_matting
            )
            results["base_processed"] = True
        except Exception as e:
            logger.error(f"Failed to process base character: {e}")
            results["errors"].append(f"Base character: {str(e)}")

        # Process variants if requested
        if process_variants:
            async with self.session_maker() as session:
                query = select(CharacterVariant).where(
                    CharacterVariant.character_id == character_id
                )
                result = await session.execute(query)
                variants = result.scalars().all()

                for variant in variants:
                    try:
                        await self.process_character_variant(
                            variant_id=variant.id,
                            model=model,
                            alpha_matting=alpha_matting
                        )
                        results["variants_processed"] += 1
                    except Exception as e:
                        logger.error(f"Failed to process variant {variant.id}: {e}")
                        results["variants_failed"] += 1
                        results["errors"].append(f"Variant {variant.expression}: {str(e)}")

        logger.success(
            f"✓ Batch processing complete: "
            f"base={results['base_processed']}, "
            f"variants={results['variants_processed']}/{results['variants_processed'] + results['variants_failed']}"
        )

        return results

    def _verify_transparency(self, image_path: str) -> bool:
        """
        Verify that an image has transparency

        Args:
            image_path: Path to image

        Returns:
            True if image has alpha channel with transparency
        """
        try:
            img = Image.open(image_path)

            # Check if image has alpha channel
            if img.mode not in ('RGBA', 'LA', 'PA'):
                return False

            # Check if alpha channel has transparency (values < 255)
            alpha = img.getchannel('A')
            min_alpha = min(alpha.getdata())

            return min_alpha < 255

        except Exception as e:
            logger.error(f"Failed to verify transparency: {e}")
            return False


# Singleton instance
_background_removal_service: Optional[BackgroundRemovalService] = None


def get_background_removal_service() -> BackgroundRemovalService:
    """Get singleton background removal service"""
    global _background_removal_service
    if _background_removal_service is None:
        _background_removal_service = BackgroundRemovalService()
    return _background_removal_service
