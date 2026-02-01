import asyncio
from datetime import datetime, timedelta
from typing import List
from connectors.base import ContentVariant
from services.ingestion.social_metrics import SocialMetricsIngestionService

class StatsCheckbackJob:
    """
    Periodically checks for stats on recently posted content.
    """
    def __init__(self):
        self.ingestion_service = SocialMetricsIngestionService()
        
    async def run_checkback(self, lookback_hours: int = 24):
        """
        Run checkback for content posted within the last N hours.
        """
        print(f"Starting stats checkback for content from last {lookback_hours} hours...")
        
        # In a real app, fetch variants from DB where created_at > now - lookback_hours
        # For now, we'll use the mock variants from the ingestion service
        # or define a new set of "recent" variants
        
        recent_variants = [
            ContentVariant(content_id="recent_1", platform="instagram"),
            ContentVariant(content_id="recent_1", platform="tiktok"),
            ContentVariant(content_id="recent_2", platform="youtube"),
        ]
        
        # Reuse the ingestion service logic to fetch and save metrics
        # We temporarily override the mock variants in the service for this run
        # (In a real app, the service would take a list of variants as input)
        
        # For demonstration, we'll just call the ingestion service's internal method
        # assuming we refactored it to accept variants. 
        # Since we didn't refactor it yet, let's just duplicate the loop here for clarity.
        
        for variant in recent_variants:
            for connector in self.ingestion_service.enabled_connectors:
                if variant.platform in connector.list_supported_platforms():
                    try:
                        print(f"Checking stats for {variant.platform} post {variant.content_id}...")
                        metrics = await connector.fetch_metrics_for_variant(variant)
                        await self.ingestion_service._save_metrics(metrics)
                    except Exception as e:
                        print(f"Error checking stats for {variant.platform}: {e}")

if __name__ == "__main__":
    job = StatsCheckbackJob()
    asyncio.run(job.run_checkback())
