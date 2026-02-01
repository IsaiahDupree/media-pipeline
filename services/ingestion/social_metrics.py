import asyncio
from typing import List
from datetime import datetime
from connectors.base import SourceAdapter, ContentVariant
from connectors.meta.connector import MetaConnector
from connectors.blotato.connector import BlotatoConnector
from connectors.youtube.connector import YouTubeConnector
from connectors.tiktok.connector import TikTokConnector
from connectors.linkedin.connector import LinkedInConnector

class SocialMetricsIngestionService:
    def __init__(self):
        self.connectors: List[SourceAdapter] = [
            MetaConnector(),
            BlotatoConnector(),
            YouTubeConnector(),
            TikTokConnector(),
            LinkedInConnector()
        ]
        self.enabled_connectors = [c for c in self.connectors if c.is_enabled()]

    async def run_ingestion(self):
        """
        Main entry point for the ingestion job.
        Iterates over all content variants and fetches metrics from enabled connectors.
        """
        print(f"Starting social metrics ingestion with {len(self.enabled_connectors)} enabled connectors.")
        
        # In a real implementation, we would fetch variants from the DB
        # For now, we'll mock a list of variants to demonstrate the flow
        mock_variants = self._get_mock_variants()

        for variant in mock_variants:
            for connector in self.enabled_connectors:
                if variant.platform in connector.list_supported_platforms():
                    try:
                        metrics = await connector.fetch_metrics_for_variant(variant)
                        await self._save_metrics(metrics)
                    except Exception as e:
                        print(f"Error fetching metrics for {variant.platform} via {connector.id}: {e}")

    def _get_mock_variants(self) -> List[ContentVariant]:
        return [
            ContentVariant(content_id="c1", platform="instagram"),
            ContentVariant(content_id="c1", platform="youtube"),
            ContentVariant(content_id="c2", platform="tiktok"),
        ]

    async def _save_metrics(self, metrics_list):
        """
        Save fetched metrics to the database (content_metrics table).
        """
        for m in metrics_list:
            print(f"Saving metrics for {m.platform}: {m.views} views, {m.likes} likes")
            # TODO: Insert into Supabase content_metrics table
            pass

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    service = SocialMetricsIngestionService()
    asyncio.run(service.run_ingestion())
