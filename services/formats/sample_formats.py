"""
Sample Format Definitions
Pre-built format templates for common video types
"""

DEV_VLOG_MEME_FORMAT = {
    "id": "dev_vlog_meme_v1",
    "name": "Dev Vlog + Memes",
    "description": "Developer-focused content with meme overlays and pattern interrupts",
    "version": "1.0.0",
    "status": "active",
    "composition": {
        "remotionCompositionId": "DevVlogMeme",
        "fps": 30,
        "width": 1080,
        "height": 1920,
        "defaultDurationSec": 55,
        "variantSets": [
            {"id": "shorts_9x16", "label": "Shorts/Reels/TikTok", "width": 1080, "height": 1920, "maxDurationSec": 60},
            {"id": "square_1x1", "label": "Square (Feed)", "width": 1080, "height": 1080, "maxDurationSec": 60}
        ]
    },
    "defaults": {
        "params": {
            "hookIntensity": 0.9,
            "patternInterruptSec": 4,
            "captionStyle": "bold_pop",
            "zoomPunch": True
        },
        "providers": {
            "tts": {"provider": "huggingface", "model": "IndexTTS2"},
            "music": {"provider": "library"},
            "visuals": {"provider": "local"},
            "matting": {"provider": "rvm"}
        },
        "qualityProfileId": "qp_shortform_v1"
    },
    "dataSources": [
        {"id": "trendCluster", "type": "supabase_query", "queryName": "topTrendCluster", "params": {"windowHours": 24}},
        {"id": "memes", "type": "local_library", "libraryId": "meme_bank", "filter": {"limit": 20}}
    ],
    "bindings": [
        {"target": "topic", "from": "trendCluster.name", "required": True},
        {"target": "script.hook", "from": "trendCluster.angles[0].hook"},
        {"target": "script.segments", "from": "trendCluster.angles[0].segments"},
        {"target": "visuals.memes", "from": "memes.items"}
    ],
    "gates": [
        {"id": "max_duration", "type": "duration", "level": "fail", "config": {"maxSec": 60}}
    ]
}

AI_EXPLAINER_FORMAT = {
    "id": "ai_explainer_v1",
    "name": "AI Explainer",
    "description": "Educational explainer videos with AI-generated visuals and clear narration",
    "version": "1.0.0",
    "status": "active",
    "composition": {
        "remotionCompositionId": "Explainer",
        "fps": 30,
        "width": 1080,
        "height": 1920,
        "defaultDurationSec": 45,
        "variantSets": [
            {"id": "shorts_9x16", "label": "Shorts/Reels/TikTok", "width": 1080, "height": 1920, "maxDurationSec": 60}
        ]
    },
    "defaults": {
        "params": {
            "hookIntensity": 0.8,
            "patternInterruptSec": 5,
            "captionStyle": "clean_subs",
            "zoomPunch": False
        },
        "providers": {
            "tts": {"provider": "elevenlabs"},
            "music": {"provider": "library"},
            "visuals": {"provider": "pexels"}
        },
        "qualityProfileId": "qp_shortform_v1"
    },
    "dataSources": [
        {"id": "topic", "type": "supabase_query", "queryName": "trendingTopics", "params": {"limit": 5}}
    ],
    "bindings": [
        {"target": "topic", "from": "topic.topics[0].name", "required": True},
        {"target": "script.segments", "from": "topic.topics[0].explainer_segments"}
    ]
}

TREND_BREAKDOWN_FORMAT = {
    "id": "trend_breakdown_v1",
    "name": "Trend Breakdown",
    "description": "Quick breakdown of trending topics with data overlays",
    "version": "1.0.0",
    "status": "active",
    "composition": {
        "remotionCompositionId": "TrendBreakdown",
        "fps": 30,
        "width": 1080,
        "height": 1920,
        "defaultDurationSec": 30,
        "variantSets": [
            {"id": "shorts_9x16", "label": "Shorts", "width": 1080, "height": 1920, "maxDurationSec": 45}
        ]
    },
    "defaults": {
        "params": {
            "hookIntensity": 0.95,
            "patternInterruptSec": 3,
            "captionStyle": "bold_pop",
            "zoomPunch": True
        },
        "providers": {
            "tts": {"provider": "huggingface"},
            "music": {"provider": "library"},
            "visuals": {"provider": "local"}
        },
        "qualityProfileId": "qp_shortform_v1"
    },
    "dataSources": [
        {"id": "trend", "type": "supabase_query", "queryName": "topTrendCluster", "params": {"windowHours": 12}}
    ],
    "bindings": [
        {"target": "topic", "from": "trend.name", "required": True},
        {"target": "script.hook", "from": "trend.hook"},
        {"target": "script.segments", "from": "trend.breakdown_points"}
    ]
}

PRODUCT_PROMO_FORMAT = {
    "id": "product_promo_v1",
    "name": "Product Promo",
    "description": "Product showcase with feature highlights and CTA",
    "version": "1.0.0",
    "status": "draft",
    "composition": {
        "remotionCompositionId": "ProductPromo",
        "fps": 30,
        "width": 1080,
        "height": 1920,
        "defaultDurationSec": 30,
        "variantSets": [
            {"id": "shorts_9x16", "label": "Shorts/Stories", "width": 1080, "height": 1920, "maxDurationSec": 30},
            {"id": "feed_4x5", "label": "Feed Post", "width": 1080, "height": 1350, "maxDurationSec": 60}
        ]
    },
    "defaults": {
        "params": {
            "hookIntensity": 0.85,
            "patternInterruptSec": 4,
            "captionStyle": "bold_pop",
            "zoomPunch": True,
            "ctaType": "swipe_up"
        },
        "providers": {
            "tts": {"provider": "elevenlabs"},
            "music": {"provider": "library"},
            "visuals": {"provider": "local"}
        },
        "qualityProfileId": "qp_shortform_v1"
    },
    "dataSources": [
        {"id": "product", "type": "local_library", "libraryId": "products"}
    ],
    "bindings": [
        {"target": "topic", "from": "product.items[0].name", "required": True},
        {"target": "script.segments", "from": "product.items[0].features"}
    ]
}

UGC_CORNER_FORMAT = {
    "id": "ugc_corner_v1",
    "name": "UGC Corner Overlay",
    "description": "User-generated content with speaker in corner overlay",
    "version": "1.0.0",
    "status": "active",
    "composition": {
        "remotionCompositionId": "UGCCorner",
        "fps": 30,
        "width": 1080,
        "height": 1920,
        "defaultDurationSec": 45,
        "variantSets": [
            {"id": "shorts_9x16", "label": "Shorts/Reels", "width": 1080, "height": 1920, "maxDurationSec": 60}
        ]
    },
    "defaults": {
        "params": {
            "hookIntensity": 0.9,
            "patternInterruptSec": 5,
            "captionStyle": "bold_pop",
            "zoomPunch": False,
            "ugcPlacement": "bottom_left",
            "ugcSize": 0.35
        },
        "providers": {
            "tts": {"provider": "huggingface"},
            "music": {"provider": "library"},
            "visuals": {"provider": "local"},
            "matting": {"provider": "rvm"}
        },
        "qualityProfileId": "qp_shortform_v1"
    },
    "dataSources": [
        {"id": "ugcClip", "type": "local_library", "libraryId": "ugc_clips"},
        {"id": "backgrounds", "type": "local_library", "libraryId": "backgrounds"}
    ],
    "bindings": [
        {"target": "topic", "from": "ugcClip.items[0].topic", "required": True},
        {"target": "visuals.ugc.originalUrl", "from": "ugcClip.items[0].url"},
        {"target": "visuals.broll", "from": "backgrounds.items"}
    ]
}

# B-Roll + Text Overlay Format
# Automatically finds videos with person not talking OR pure b-roll footage
# Perfect for adding text overlays, quotes, tips, etc.
BROLL_TEXT_FORMAT = {
    "id": "broll_text_v1",
    "name": "B-Roll + Text Overlay",
    "description": "Auto-finds b-roll clips (person not talking or no person) and adds text overlays. Perfect for tips, quotes, and educational content.",
    "version": "1.0.0",
    "status": "active",
    "composition": {
        "remotionCompositionId": "BrollText",
        "fps": 30,
        "width": 1080,
        "height": 1920,
        "defaultDurationSec": 15,
        "variantSets": [
            {"id": "shorts_9x16", "label": "Shorts/Reels/TikTok", "width": 1080, "height": 1920, "maxDurationSec": 60},
            {"id": "square_1x1", "label": "Square (Feed)", "width": 1080, "height": 1080, "maxDurationSec": 60},
            {"id": "story_9x16", "label": "Story", "width": 1080, "height": 1920, "maxDurationSec": 15}
        ]
    },
    "defaults": {
        "params": {
            "hookIntensity": 0.7,
            "captionStyle": "bold_pop",
            "textPosition": "center",  # center, top, bottom
            "textAnimation": "fade_scale",  # fade_scale, typewriter, slide_up
            "backgroundDim": 0.3,  # Dim the video slightly for text readability
            "autoFindClips": True,  # Key feature: auto-discovery
            "clipFilter": "broll_text"  # Uses format classifier
        },
        "providers": {
            "music": {"provider": "library", "mood": "upbeat"},
            "visuals": {"provider": "local"}
        },
        "qualityProfileId": "qp_shortform_v1"
    },
    "dataSources": [
        # Auto-discover b-roll candidates from library
        {
            "id": "brollCandidates",
            "type": "local_library",
            "libraryId": "media_db",
            "filter": {
                "format_type": "broll_text",  # Person visible but not talking
                "limit": 50,
                "has_captions": False
            }
        },
        {
            "id": "pureBrollCandidates",
            "type": "local_library",
            "libraryId": "media_db",
            "filter": {
                "format_type": "pure_broll",  # No person, no speech
                "limit": 50,
                "has_captions": False
            }
        }
    ],
    "bindings": [
        {"target": "visuals.broll", "from": "brollCandidates.items", "required": False},
        {"target": "visuals.pureBroll", "from": "pureBrollCandidates.items", "required": False},
        {"target": "script.segments", "from": "params.textSegments"}  # User provides text
    ],
    "gates": [
        {"id": "max_duration", "type": "duration", "level": "fail", "config": {"maxSec": 60}},
        {"id": "has_broll", "type": "visual", "level": "warn", "config": {"requireBroll": True}}
    ]
}

# Pure B-Roll Format (no person at all)
PURE_BROLL_FORMAT = {
    "id": "pure_broll_v1",
    "name": "Pure B-Roll + Text",
    "description": "Auto-finds footage without people for cinematic text overlays. Great for motivational content, quotes, and aesthetic posts.",
    "version": "1.0.0",
    "status": "active",
    "composition": {
        "remotionCompositionId": "PureBrollText",
        "fps": 30,
        "width": 1080,
        "height": 1920,
        "defaultDurationSec": 15,
        "variantSets": [
            {"id": "shorts_9x16", "label": "Shorts/Reels/TikTok", "width": 1080, "height": 1920, "maxDurationSec": 60},
            {"id": "square_1x1", "label": "Square (Feed)", "width": 1080, "height": 1080, "maxDurationSec": 60}
        ]
    },
    "defaults": {
        "params": {
            "captionStyle": "clean_subs",
            "textPosition": "center",
            "textAnimation": "fade_scale",
            "backgroundDim": 0.2,
            "autoFindClips": True,
            "clipFilter": "pure_broll"
        },
        "providers": {
            "music": {"provider": "library", "mood": "calm"},
            "visuals": {"provider": "local"}
        },
        "qualityProfileId": "qp_shortform_v1"
    },
    "dataSources": [
        {
            "id": "pureBroll",
            "type": "local_library",
            "libraryId": "media_db",
            "filter": {
                "format_type": "pure_broll",
                "limit": 50,
                "has_captions": False
            }
        }
    ],
    "bindings": [
        {"target": "visuals.broll", "from": "pureBroll.items", "required": True},
        {"target": "script.segments", "from": "params.textSegments"}
    ],
    "gates": [
        {"id": "max_duration", "type": "duration", "level": "fail", "config": {"maxSec": 60}}
    ]
}

# All sample formats
SAMPLE_FORMATS = [
    DEV_VLOG_MEME_FORMAT,
    AI_EXPLAINER_FORMAT,
    TREND_BREAKDOWN_FORMAT,
    PRODUCT_PROMO_FORMAT,
    UGC_CORNER_FORMAT,
    BROLL_TEXT_FORMAT,
    PURE_BROLL_FORMAT,
]


def get_sample_format(format_id: str):
    """Get a sample format by ID."""
    for fmt in SAMPLE_FORMATS:
        if fmt["id"] == format_id:
            return fmt
    return None


def list_sample_formats():
    """List all sample formats (for seeding)."""
    return SAMPLE_FORMATS
