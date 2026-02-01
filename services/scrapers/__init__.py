"""
Social Media Scrapers with Swappable Providers
Centralized initialization and configuration
"""
import os
import logging
from typing import Optional

from .provider_base import Platform
from .provider_factory import ProviderFactory, ProviderConfig, get_factory
from .tiktok_providers import TikTokFeatureSummaryProvider, TikTokScraper7Provider
from .instagram_providers import InstagramStatisticsProvider, InstagramPremiumProvider

logger = logging.getLogger(__name__)


def initialize_providers(api_key: Optional[str] = None) -> ProviderFactory:
    """
    Initialize and register all social media providers
    
    Args:
        api_key: RapidAPI key (defaults to env variable)
        
    Returns:
        Configured ProviderFactory instance
    """
    factory = get_factory()
    rapidapi_key = api_key or os.getenv("RAPIDAPI_KEY")
    
    if not rapidapi_key:
        logger.warning("RAPIDAPI_KEY not set - providers will not be available")
        return factory
    
    # ===== TIKTOK PROVIDERS =====
    
    # Provider 1: TikTok Video Feature Summary (PRIMARY)
    factory.register_provider(ProviderConfig(
        provider_class=TikTokFeatureSummaryProvider,
        api_key=rapidapi_key,
        base_url="https://tiktok-video-feature-summary.p.rapidapi.com",
        name="TikTok Feature Summary",
        platform=Platform.TIKTOK,
        priority=1,  # Primary
        enabled=True
    ))
    
    # Provider 2: TikTok Scraper7 (FALLBACK)
    factory.register_provider(ProviderConfig(
        provider_class=TikTokScraper7Provider,
        api_key=rapidapi_key,
        base_url="https://tiktok-scraper7.p.rapidapi.com",
        name="TikTok Scraper7",
        platform=Platform.TIKTOK,
        priority=2,  # Fallback
        enabled=True
    ))
    
    # ===== INSTAGRAM PROVIDERS =====
    
    # Provider 1: Instagram Statistics API (PRIMARY)
    factory.register_provider(ProviderConfig(
        provider_class=InstagramStatisticsProvider,
        api_key=rapidapi_key,
        base_url="https://instagram-statistics-api.p.rapidapi.com",
        name="Instagram Statistics API",
        platform=Platform.INSTAGRAM,
        priority=1,  # Primary
        enabled=True
    ))
    
    # Provider 2: Instagram Premium API (FALLBACK)
    factory.register_provider(ProviderConfig(
        provider_class=InstagramPremiumProvider,
        api_key=rapidapi_key,
        base_url="https://instagram-premium-api-2023.p.rapidapi.com",
        name="Instagram Premium API",
        platform=Platform.INSTAGRAM,
        priority=2,  # Fallback
        enabled=True
    ))
    
    logger.info("✅ Initialized social media providers:")
    logger.info(f"  - TikTok: 2 providers registered")
    logger.info(f"  - Instagram: 2 providers registered")
    
    return factory


# Initialize on import
_factory = initialize_providers()


def get_tiktok_provider():
    """Get a TikTok provider instance (with automatic fallback)"""
    return _factory.get_provider(Platform.TIKTOK)


def get_instagram_provider():
    """Get an Instagram provider instance (with automatic fallback)"""
    return _factory.get_provider(Platform.INSTAGRAM)


async def get_tiktok_profile(username: str):
    """
    Quick helper: Get TikTok profile
    Uses automatic provider selection and fallback
    """
    return await _factory.execute_with_fallback(
        Platform.TIKTOK,
        "get_profile",
        username
    )


async def get_instagram_profile(username: str):
    """
    Quick helper: Get Instagram profile  
    Uses automatic provider selection and fallback
    """
    return await _factory.execute_with_fallback(
        Platform.INSTAGRAM,
        "get_profile",
        username
    )


async def get_tiktok_analytics(username: str, posts_limit: int = 50):
    """Quick helper: Get TikTok analytics"""
    return await _factory.execute_with_fallback(
        Platform.TIKTOK,
        "get_analytics",
        username,
        posts_limit
    )


async def get_instagram_analytics(username: str, posts_limit: int = 50):
    """Quick helper: Get Instagram analytics"""
    return await _factory.execute_with_fallback(
        Platform.INSTAGRAM,
        "get_analytics",
        username,
        posts_limit
    )


__all__ = [
    # Factory
    "get_factory",
    "initialize_providers",
    
    # Providers
    "get_tiktok_provider",
    "get_instagram_provider",
    
    # Quick helpers
    "get_tiktok_profile",
    "get_instagram_profile",
    "get_tiktok_analytics",
    "get_instagram_analytics",
    
    # Platform enum
    "Platform",
]
