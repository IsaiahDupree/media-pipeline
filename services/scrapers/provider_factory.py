"""
Provider Factory with automatic fallback and health monitoring
Manages multiple providers per platform with intelligent switching
"""
import logging
import asyncio
import time
from typing import List, Optional, Dict
from datetime import datetime, timedelta

from .provider_base import (
    ProviderInterface,
    ProviderConfig,
    ProviderMetrics,
    Platform,
    ProfileData,
    PostData,
    AnalyticsData
)

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating and managing social media API providers
    Supports automatic fallback, health monitoring, and provider switching
    """
    
    def __init__(self):
        self._providers: Dict[Platform, List[ProviderConfig]] = {}
        self._health_cache: Dict[str, tuple[bool, datetime]] = {}
        self._performance_metrics: Dict[str, List[ProviderMetrics]] = {}
        self._health_check_interval = timedelta(minutes=5)
    
    def register_provider(self, config: ProviderConfig):
        """
        Register a provider configuration
        
        Args:
            config: ProviderConfig object
        """
        platform = config.platform
        if platform not in self._providers:
            self._providers[platform] = []
        
        self._providers[platform].append(config)
        
        # Sort by priority (lower number = higher priority)
        self._providers[platform].sort(key=lambda x: x.priority)
        
        logger.info(f"Registered provider: {config.name} for {platform.value} (priority: {config.priority})")
    
    def get_providers(self, platform: Platform) -> List[ProviderConfig]:
        """Get all registered providers for a platform"""
        return self._providers.get(platform, [])
    
    async def get_provider(
        self,
        platform: Platform,
        prefer_priority: int = 1,
        check_health: bool = True
    ) -> Optional[ProviderInterface]:
        """
        Get a healthy provider instance for a platform
        
        Args:
            platform: Platform enum
            prefer_priority: Preferred priority level
            check_health: Whether to check provider health
            
        Returns:
            Provider instance or None if no healthy provider
        """
        providers = self.get_providers(platform)
        if not providers:
            logger.error(f"No providers registered for {platform.value}")
            return None
        
        # Try preferred priority first
        preferred = [p for p in providers if p.priority == prefer_priority and p.enabled]
        others = [p for p in providers if p.priority != prefer_priority and p.enabled]
        
        # Combine with preferred first
        ordered = preferred + others
        
        for config in ordered:
            if check_health:
                is_healthy = await self._check_provider_health(config)
                if not is_healthy:
                    logger.warning(f"Provider {config.name} is unhealthy, trying next...")
                    continue
            
            try:
                instance = config.create_instance()
                logger.info(f"Using provider: {config.name} for {platform.value}")
                return instance
            except Exception as e:
                logger.error(f"Failed to create provider {config.name}: {e}")
                continue
        
        logger.error(f"No healthy providers available for {platform.value}")
        return None
    
    async def _check_provider_health(self, config: ProviderConfig) -> bool:
        """
        Check if a provider is healthy (with caching)
        
        Args:
            config: ProviderConfig to check
            
        Returns:
            True if healthy, False otherwise
        """
        cache_key = f"{config.platform.value}:{config.name}"
        
        # Check cache
        if cache_key in self._health_cache:
            is_healthy, checked_at = self._health_cache[cache_key]
            if datetime.now() - checked_at < self._health_check_interval:
                return is_healthy
        
        # Perform health check
        try:
            instance = config.create_instance()
            is_healthy = await instance.health_check()
            self._health_cache[cache_key] = (is_healthy, datetime.now())
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed for {config.name}: {e}")
            self._health_cache[cache_key] = (False, datetime.now())
            return False
    
    async def execute_with_fallback(
        self,
        platform: Platform,
        operation: str,
        *args,
        **kwargs
    ):
        """
        Execute an operation with automatic fallback to next provider
        
        Args:
            platform: Platform enum
            operation: Method name to call (e.g., 'get_profile')
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            Result from the operation or None if all providers failed
        """
        providers = self.get_providers(platform)
        if not providers:
            logger.error(f"No providers registered for {platform.value}")
            return None
        
        errors = []
        
        for config in providers:
            if not config.enabled:
                continue
            
            try:
                start_time = time.time()
                instance = config.create_instance()
                
                # Get the method
                method = getattr(instance, operation, None)
                if not method:
                    logger.error(f"Provider {config.name} doesn't support operation: {operation}")
                    continue
                
                # Execute the method
                result = await method(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                
                # Track metrics
                self._track_metrics(config.name, latency_ms, True, None)
                
                if result is not None:
                    logger.info(
                        f"✅ {config.name} succeeded for {operation} "
                        f"(latency: {latency_ms:.0f}ms)"
                    )
                    return result
                else:
                    errors.append(f"{config.name}: Returned None")
                    
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else 0
                error_msg = str(e)
                errors.append(f"{config.name}: {error_msg}")
                self._track_metrics(config.name, latency_ms, False, error_msg)
                logger.warning(f"❌ {config.name} failed for {operation}: {error_msg}")
                continue
        
        # All providers failed
        logger.error(
            f"All providers failed for {platform.value}.{operation}:\n" +
            "\n".join(f"  - {err}" for err in errors)
        )
        return None
    
    def _track_metrics(
        self,
        provider_name: str,
        latency_ms: float,
        success: bool,
        error: Optional[str]
    ):
        """Track provider performance metrics"""
        if provider_name not in self._performance_metrics:
            self._performance_metrics[provider_name] = []
        
        metric = ProviderMetrics(
            name=provider_name,
            latency_ms=latency_ms,
            success=success,
            error=error,
            timestamp=datetime.now()
        )
        
        self._performance_metrics[provider_name].append(metric)
        
        # Keep only last 100 metrics per provider
        if len(self._performance_metrics[provider_name]) > 100:
            self._performance_metrics[provider_name] = self._performance_metrics[provider_name][-100:]
    
    def get_provider_stats(self, provider_name: str) -> Dict:
        """Get performance statistics for a provider"""
        metrics = self._performance_metrics.get(provider_name, [])
        if not metrics:
            return {
                "provider": provider_name,
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0
            }
        
        total = len(metrics)
        successful = sum(1 for m in metrics if m.success)
        success_rate = (successful / total) * 100
        avg_latency = sum(m.latency_ms for m in metrics) / total
        
        return {
            "provider": provider_name,
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": total - successful,
            "success_rate": round(success_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "last_request": metrics[-1].timestamp.isoformat()
        }
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all providers"""
        return {
            name: self.get_provider_stats(name)
            for name in self._performance_metrics.keys()
        }
    
    async def compare_providers(
        self,
        platform: Platform,
        test_username: str
    ) -> List[Dict]:
        """
        Compare all providers for a platform by running the same test
        
        Args:
            platform: Platform to test
            test_username: Username to test with
            
        Returns:
            List of comparison results
        """
        providers = self.get_providers(platform)
        results = []
        
        for config in providers:
            if not config.enabled:
                continue
            
            try:
                start_time = time.time()
                instance = config.create_instance()
                
                # Try to get profile
                profile = await instance.get_profile(test_username)
                latency_ms = (time.time() - start_time) * 1000
                
                result = {
                    "provider": config.name,
                    "priority": config.priority,
                    "success": profile is not None,
                    "latency_ms": round(latency_ms, 2),
                    "data_completeness": self._check_completeness(profile) if profile else 0,
                    "error": None
                }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "provider": config.name,
                    "priority": config.priority,
                    "success": False,
                    "latency_ms": 0,
                    "data_completeness": 0,
                    "error": str(e)
                })
        
        # Sort by priority
        results.sort(key=lambda x: x["priority"])
        return results
    
    def _check_completeness(self, profile: ProfileData) -> float:
        """Check how complete the profile data is (0-100%)"""
        fields = [
            profile.username,
            profile.full_name,
            profile.bio,
            profile.profile_pic_url,
            profile.followers_count > 0,
            profile.posts_count > 0
        ]
        
        complete = sum(1 for f in fields if f)
        return (complete / len(fields)) * 100


# Global factory instance
factory = ProviderFactory()


def get_factory() -> ProviderFactory:
    """Get the global provider factory instance"""
    return factory
