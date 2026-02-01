"""
Event Topics
============
Standardized topic names for all pub/sub events.

Topic Naming Convention:
    {domain}.{entity}.{action}
    
Examples:
    - media.ingested
    - media.analysis.started
    - publish.completed
"""


class Topics:
    """
    Centralized topic registry for all events.
    
    Usage:
        from services.event_bus import Topics
        
        await bus.publish(Topics.MEDIA_INGESTED, {...})
        bus.subscribe(Topics.ANALYSIS_COMPLETED, handler)
    """
    
    # =========================================================================
    # MEDIA LIFECYCLE
    # =========================================================================
    MEDIA_INGESTED = "media.ingested"           # New video added to library
    MEDIA_UPDATED = "media.updated"             # Video metadata updated
    MEDIA_DELETED = "media.deleted"             # Video removed from library
    MEDIA_THUMBNAIL_READY = "media.thumbnail.ready"  # Thumbnail generated
    
    # =========================================================================
    # ANALYSIS PIPELINE
    # =========================================================================
    ANALYSIS_REQUESTED = "media.analysis.requested"   # User/system requests analysis
    ANALYSIS_STARTED = "media.analysis.started"       # Analysis worker picked up job
    ANALYSIS_PROGRESS = "media.analysis.progress"     # Progress update (% complete)
    ANALYSIS_STEP_COMPLETED = "media.analysis.step.completed"  # Individual step done
    ANALYSIS_COMPLETED = "media.analysis.completed"   # Full analysis finished
    ANALYSIS_FAILED = "media.analysis.failed"         # Analysis error
    
    # Analysis sub-steps
    TRANSCRIPT_STARTED = "media.analysis.transcript.started"
    TRANSCRIPT_COMPLETED = "media.analysis.transcript.completed"
    VISUAL_STARTED = "media.analysis.visual.started"
    VISUAL_COMPLETED = "media.analysis.visual.completed"
    AI_ANALYSIS_STARTED = "media.analysis.ai.started"
    AI_ANALYSIS_COMPLETED = "media.analysis.ai.completed"
    CAPTIONS_GENERATED = "media.analysis.captions.generated"
    
    # =========================================================================
    # PUBLISHING PIPELINE
    # =========================================================================
    PUBLISH_REQUESTED = "publish.requested"           # Publish job created
    PUBLISH_QUEUED = "publish.queued"                 # Added to publish queue
    PUBLISH_STARTED = "publish.started"               # Worker picked up job
    PUBLISH_UPLOADING = "publish.uploading"           # Uploading to cloud/blotato
    PUBLISH_UPLOAD_COMPLETED = "publish.upload.completed"  # Upload done
    PUBLISH_SUBMITTED = "publish.submitted"           # Sent to platform
    PUBLISH_POLLING = "publish.polling"               # Waiting for platform URL
    PUBLISH_COMPLETED = "publish.completed"           # URL obtained, success
    PUBLISH_FAILED = "publish.failed"                 # Publish error
    PUBLISH_RETRYING = "publish.retrying"             # Retry scheduled
    
    # =========================================================================
    # SCHEDULING
    # =========================================================================
    SCHEDULE_CREATED = "schedule.created"             # New scheduled post
    SCHEDULE_UPDATED = "schedule.updated"             # Schedule modified
    SCHEDULE_CANCELLED = "schedule.cancelled"         # Schedule cancelled
    SCHEDULE_DUE = "schedule.due"                     # Post is due for publishing
    SCHEDULER_TICK = "scheduler.tick"                 # Periodic scheduler heartbeat
    SCHEDULER_STARTED = "scheduler.started"           # Scheduler service started
    SCHEDULER_STOPPED = "scheduler.stopped"           # Scheduler service stopped
    
    # =========================================================================
    # METRICS & ANALYTICS
    # =========================================================================
    METRICS_FETCH_REQUESTED = "metrics.fetch.requested"   # Request to fetch metrics
    METRICS_FETCH_STARTED = "metrics.fetch.started"       # Fetching from platform
    METRICS_FETCH_COMPLETED = "metrics.fetch.completed"   # Metrics fetched
    METRICS_UPDATED = "metrics.updated"                   # New metrics available
    METRICS_AGGREGATED = "metrics.aggregated"             # Aggregation complete

    # =========================================================================
    # POST TRACKING (PTK-001 to PTK-006)
    # =========================================================================
    POST_PUBLISHED = "post.published"                     # Post successfully published
    CHECKBACK_SCHEDULED = "checkback.scheduled"           # Checkback period scheduled
    CHECKBACK_TRIGGERED = "checkback.triggered"           # Checkback period triggered
    CHECKBACK_COMPLETED = "checkback.completed"           # Checkback metrics collected
    
    # =========================================================================
    # AI GENERATION
    # =========================================================================
    AI_GENERATION_REQUESTED = "ai.generation.requested"   # AI video generation request
    AI_GENERATION_STARTED = "ai.generation.started"       # Generation in progress
    AI_GENERATION_PROGRESS = "ai.generation.progress"     # Progress update
    AI_GENERATION_COMPLETED = "ai.generation.completed"   # Generation done
    AI_GENERATION_FAILED = "ai.generation.failed"         # Generation error
    
    # =========================================================================
    # CLIP EXTRACTION (Long-form to Short-form)
    # =========================================================================
    CLIP_EXTRACTION_REQUESTED = "clip.extraction.requested"     # Request to extract clips
    CLIP_EXTRACTION_STARTED = "clip.extraction.started"         # Worker picked up job
    CLIP_EXTRACTION_PROGRESS = "clip.extraction.progress"       # Progress update
    CLIP_TRANSCRIPT_COMPLETED = "clip.extraction.transcript"    # Transcript ready
    CLIP_SEGMENTS_IDENTIFIED = "clip.extraction.segments"       # AI found segments
    CLIP_RENDERING_STARTED = "clip.extraction.rendering"        # Rendering clips
    CLIP_SINGLE_COMPLETED = "clip.extraction.clip_done"         # One clip finished
    CLIP_EXTRACTION_COMPLETED = "clip.extraction.completed"     # All clips done
    CLIP_EXTRACTION_FAILED = "clip.extraction.failed"           # Extraction error
    
    # =========================================================================
    # NOTIFICATIONS
    # =========================================================================
    NOTIFICATION_CREATED = "notification.created"         # New notification
    NOTIFICATION_SENT = "notification.sent"               # Notification delivered
    
    # =========================================================================
    # COMMENTS & ENGAGEMENT
    # =========================================================================
    COMMENT_RECEIVED = "comment.received"                 # New comment detected
    COMMENT_ANALYZED = "comment.analyzed"                 # Sentiment/intent analyzed
    COMMENT_REPLIED = "comment.replied"                   # Auto-reply sent
    COMMENT_FLAGGED = "comment.flagged"                   # Comment flagged for review
    FAN_IDENTIFIED = "fan.identified"                     # Top fan identified
    FAN_TIER_CHANGED = "fan.tier_changed"                 # Fan tier upgraded/downgraded
    TOP_FAN_ALERT = "fan.top_fan_alert"                   # Top fan activity alert
    
    # =========================================================================
    # WORKFLOWS
    # =========================================================================
    WORKFLOW_CREATED = "workflow.created"                 # New workflow created
    WORKFLOW_STARTED = "workflow.started"                 # Workflow execution started
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"   # Workflow step finished
    WORKFLOW_COMPLETED = "workflow.completed"             # Workflow finished
    WORKFLOW_FAILED = "workflow.failed"                   # Workflow error
    
    # =========================================================================
    # SOCIAL ACCOUNTS
    # =========================================================================
    ACCOUNT_CONNECTED = "account.connected"               # New account connected
    ACCOUNT_SYNCED = "account.synced"                     # Account data refreshed
    ACCOUNT_DISCONNECTED = "account.disconnected"         # Account removed
    ACCOUNT_METRICS_UPDATED = "account.metrics_updated"   # Account metrics refreshed
    
    # =========================================================================
    # EMAIL NOTIFICATIONS
    # =========================================================================
    EMAIL_QUEUED = "email.queued"                         # Email queued for sending
    EMAIL_SENT = "email.sent"                             # Email sent successfully
    EMAIL_FAILED = "email.failed"                         # Email send failed

    # =========================================================================
    # CONTENT SOURCING (PIPE-001)
    # =========================================================================
    CONTENT_DISCOVERED = "content.discovered"             # New content files discovered
    CONTENT_INGESTED = "content.ingested"                 # Content file ingested to DB
    CONTENT_ANALYSIS_COMPLETED = "content.analysis.completed"  # AI analysis completed
    
    # =========================================================================
    # VIDEO READY PIPELINE
    # =========================================================================
    VIDEO_READY = "video.ready"                           # Video is ready for processing
    VIDEO_READY_PROCESSED = "video.ready.processed"       # Video ready pipeline completed
    VIDEO_READY_PUBLISHED = "video.ready.published"       # Video published to platforms
    
    # =========================================================================
    # HYDRATION (State Management)
    # =========================================================================
    HYDRATION_SNAPSHOT_READY = "mp.hydration.evt.snapshot_ready"  # Fresh state available
    HYDRATION_FEATURES_READY = "mp.hydration.evt.features_ready"  # Derived features computed
    
    # =========================================================================
    # NARRATIVE BUILDER (Mainline Brain)
    # =========================================================================
    NARRATIVE_PLAN_REQUESTED = "mp.narrative.cmd.plan"            # Request new plan
    NARRATIVE_PLAN_GENERATED = "mp.narrative.evt.plan_generated"  # Plan ready
    NARRATIVE_GOAL_UPDATED = "mp.narrative.evt.goal_updated"      # Goal progress changed
    NARRATIVE_SIGNALS_UPDATED = "narrative.signals.updated"      # Signals refreshed
    
    # =========================================================================
    # GOALS
    # =========================================================================
    GOAL_CREATED = "goal.created"                                 # New goal created
    GOAL_UPDATED = "goal.updated"                                 # Goal progress updated
    GOAL_COMPLETED = "goal.completed"                             # Goal achieved
    
    # =========================================================================
    # EXPERIMENTS (Research Brain)
    # =========================================================================
    EXPERIMENT_PLAN_RUN = "mp.experiments.cmd.plan_run"           # Start experiment
    EXPERIMENT_RUN_STARTED = "mp.experiments.evt.run_started"     # Run began
    EXPERIMENT_VARIANT_CREATED = "mp.experiments.evt.variant_created"  # Variant scheduled
    EXPERIMENT_RUN_COMPLETED = "mp.experiments.evt.run_completed" # Run finished
    EXPERIMENT_METRICS_READY = "mp.experiments.evt.metrics_ready" # Metrics collected
    
    # =========================================================================
    # KNOWLEDGE BASE (Rules & Learnings)
    # =========================================================================
    RULE_CREATED = "mp.rules.evt.rule_created"                    # New rule from experiment
    RULE_UPDATED = "mp.rules.evt.rule_updated"                    # Rule confidence updated
    RULE_DEPRECATED = "mp.rules.evt.rule_deprecated"              # Rule no longer valid
    TEMPLATE_CREATED = "mp.rules.evt.template_created"            # New template
    TEMPLATE_FORKED = "mp.template.forked"                         # Template auto-forked (AUTO-003)
    TEMPLATE_LEADERBOARD_UPDATED = "mp.template.leaderboard.updated"  # Template rankings updated
    TEMPLATE_RETIRED = "mp.template.retired"                       # Template retired (AUTO-004)
    PLAYBOOK_ACTIVATED = "mp.rules.evt.playbook_activated"        # Playbook in use
    
    # =========================================================================
    # SCHEDULER (Execution) - DEPRECATED: Use schedule.* topics instead
    # These are kept for backwards compatibility but new code should use:
    #   - SCHEDULE_CREATED instead of SCHEDULER_CREATE_ITEMS
    #   - SCHEDULE_UPDATED instead of SCHEDULER_UPDATE_ITEM
    #   - SCHEDULE_CANCELLED instead of SCHEDULER_CANCEL_ITEM
    #   - SCHEDULE_DUE instead of SCHEDULER_ITEM_DUE
    # =========================================================================
    SCHEDULER_CREATE_ITEMS = "mp.scheduler.cmd.create_items"      # DEPRECATED: Use SCHEDULE_CREATED
    SCHEDULER_UPDATE_ITEM = "mp.scheduler.cmd.update_item"        # DEPRECATED: Use SCHEDULE_UPDATED
    SCHEDULER_CANCEL_ITEM = "mp.scheduler.cmd.cancel_item"        # DEPRECATED: Use SCHEDULE_CANCELLED
    SCHEDULER_ITEM_SCHEDULED = "mp.scheduler.evt.item_scheduled"  # DEPRECATED: Use SCHEDULE_CREATED
    SCHEDULER_ITEM_DUE = "mp.scheduler.evt.item_due"              # DEPRECATED: Use SCHEDULE_DUE
    SCHEDULER_ITEM_CANCELED = "mp.scheduler.evt.item_canceled"    # DEPRECATED: Use SCHEDULE_CANCELLED
    
    # =========================================================================
    # TRENDS (Opportunity Signals)
    # =========================================================================
    TREND_SYNC_PROVIDER = "mp.trends.cmd.sync_provider"           # Trigger provider pull
    TREND_RAW_INGESTED = "mp.trends.evt.raw_ingested"             # Raw payload stored
    TREND_NORMALIZED_READY = "mp.trends.evt.normalized_ready"     # Normalized items ready
    TREND_CLUSTER_READY = "mp.trends.evt.cluster_ready"           # Cluster computed
    TREND_OPPORTUNITY_READY = "mp.trends.evt.opportunity_ready"   # Scored opportunity
    TREND_ALERT = "mp.trends.evt.alert"                           # Big spike / time-sensitive
    TREND_BRIEF_READY = "mp.briefs.evt.trend_brief_ready"         # Brief generated
    TREND_ASSET_MATCHED = "mp.trends.evt.asset_matched"           # Assets matched to trend
    
    # =========================================================================
    # UI / REALTIME
    # =========================================================================
    UI_TOAST = "mp.ui.evt.toast"                                  # Show notification
    UI_INVALIDATE_CACHE = "mp.ui.evt.invalidate"                  # Cache invalidation
    UI_ACTIVITY_FEED = "mp.ui.evt.activity"                       # Activity feed item
    
    # =========================================================================
    # COMPETITOR RESEARCH
    # =========================================================================
    COMPETITOR_ADDED = "competitor.added"                 # New competitor account tracked
    COMPETITOR_REMOVED = "competitor.removed"             # Competitor account removed
    COMPETITOR_SYNC_STARTED = "competitor.sync.started"   # Sync job started
    COMPETITOR_SYNC_COMPLETED = "competitor.sync.completed"  # Sync job finished
    COMPETITOR_ANALYZED = "competitor.analyzed"           # AI analysis complete
    COMPETITOR_CONTENT_DOWNLOADED = "competitor.content.downloaded"  # Videos downloaded
    
    # =========================================================================
    # DEVICE IMPORT (iOS/Android)
    # =========================================================================
    IMPORT_SCAN_STARTED = "import.scan.started"           # Scanning folder for files
    IMPORT_SCAN_COMPLETED = "import.scan.completed"       # Scan finished, files found
    IMPORT_JOB_STARTED = "import.job.started"             # Import job began
    IMPORT_JOB_PROGRESS = "import.job.progress"           # Progress update
    IMPORT_FILE_INGESTED = "import.file.ingested"         # Single file imported
    IMPORT_FILE_SKIPPED = "import.file.skipped"           # Duplicate skipped
    IMPORT_JOB_COMPLETED = "import.job.completed"         # Import job finished
    IMPORT_JOB_FAILED = "import.job.failed"               # Import job error
    IMPORT_JOB_PAUSED = "import.job.paused"               # Import paused
    IMPORT_JOB_RESUMED = "import.job.resumed"             # Import resumed
    IMPORT_JOB_CANCELLED = "import.job.cancelled"         # Import cancelled
    
    # =========================================================================
    # TTS (Text-to-Speech) SERVICE
    # =========================================================================
    TTS_REQUESTED = "tts.requested"                       # New TTS job
    TTS_STARTED = "tts.started"                          # Job picked up
    TTS_PROGRESS = "tts.progress"                        # Progress update
    TTS_COMPLETED = "tts.completed"                      # Audio generated
    TTS_FAILED = "tts.failed"                            # Generation error
    TTS_MODEL_LOADED = "tts.model.loaded"                # Model ready
    TTS_MODEL_UNLOADED = "tts.model.unloaded"            # Model freed
    
    # =========================================================================
    # REMOTION (Video Editing) SERVICE
    # =========================================================================
    REMOTION_REQUESTED = "remotion.requested"            # New render job
    REMOTION_STARTED = "remotion.started"               # Job picked up
    REMOTION_COMPOSING = "remotion.composing"           # Building composition
    REMOTION_RENDERING = "remotion.rendering"            # Rendering video
    REMOTION_PROGRESS = "remotion.progress"              # Progress update
    REMOTION_COMPLETED = "remotion.completed"          # Video rendered
    REMOTION_FAILED = "remotion.failed"                  # Render error
    
    # =========================================================================
    # VIDEO RENDERING (Format-Agnostic) SERVICE
    # =========================================================================
    VIDEO_RENDER_REQUESTED = "video.render.requested"           # New format-based render job
    VIDEO_RENDER_STARTED = "video.render.started"               # Job picked up by worker
    VIDEO_RENDER_SCENE_GRAPH_BUILT = "video.render.scene_graph.built"  # Scene graph created
    VIDEO_RENDER_SCENE_STARTED = "video.render.scene.started"   # Individual scene rendering started
    VIDEO_RENDER_SCENE_COMPLETED = "video.render.scene.completed"  # Individual scene completed
    VIDEO_RENDER_PROGRESS = "video.render.progress"              # Progress update (%)
    VIDEO_RENDER_COMPOSING = "video.render.composing"            # Composing final video
    VIDEO_RENDER_COMPLETED = "video.render.completed"            # Video rendered successfully
    VIDEO_RENDER_FAILED = "video.render.failed"                  # Render error
    
    # Format-specific events
    VIDEO_FORMAT_SELECTED = "video.format.selected"             # Format selected for content
    VIDEO_FORMAT_VALIDATED = "video.format.validated"           # Format validation passed
    
    # =========================================================================
    # VIDEO MATTING SERVICE
    # =========================================================================
    MATTING_REQUESTED = "matting.requested"              # New matting job
    MATTING_STARTED = "matting.started"                  # Job picked up
    MATTING_SEGMENTING = "matting.segmenting"           # Segmenting objects
    MATTING_EXTRACTING = "matting.extracting"           # Extracting foreground
    MATTING_COMPOSITING = "matting.compositing"          # Compositing into target
    MATTING_PROGRESS = "matting.progress"                # Progress update
    MATTING_COMPLETED = "matting.completed"             # Video processed
    MATTING_FAILED = "matting.failed"                    # Processing error
    
    # =========================================================================
    # CONTENT BRIEF ENHANCED
    # =========================================================================
    CONTENT_BRIEF_GENERATED = "content.brief.generated"  # New brief generated
    CONTENT_BRIEF_SCORED = "content.brief.scored"        # Brief scored
    CONTENT_BRIEF_APPROVED = "content.brief.approved"   # Brief approved for production
    CONTENT_BRIEF_SCRIPT_GENERATED = "content.brief.script.generated"  # Script.json generated
    
    # =========================================================================
    # PIPELINE ORCHESTRATION
    # =========================================================================
    PIPELINE_REQUESTED = "pipeline.requested"            # New pipeline job
    PIPELINE_STARTED = "pipeline.started"                # Pipeline started
    PIPELINE_STAGE_STARTED = "pipeline.stage.started"     # Stage started
    PIPELINE_STAGE_COMPLETED = "pipeline.stage.completed" # Stage completed
    PIPELINE_PROGRESS = "pipeline.progress"               # Progress update
    PIPELINE_COMPLETED = "pipeline.completed"            # Pipeline finished
    PIPELINE_FAILED = "pipeline.failed"                   # Pipeline error

    # =========================================================================
    # MASTER ORCHESTRATOR (ARCH-001)
    # =========================================================================
    ORCHESTRATOR_PIPELINE_STARTED = "orchestrator.pipeline.started"       # Full pipeline execution started
    ORCHESTRATOR_PIPELINE_COMPLETED = "orchestrator.pipeline.completed"   # Full pipeline execution completed
    ORCHESTRATOR_PIPELINE_FAILED = "orchestrator.pipeline.failed"         # Full pipeline execution failed
    ORCHESTRATOR_STEP_STARTED = "orchestrator.step.started"               # Pipeline step started
    ORCHESTRATOR_STEP_COMPLETED = "orchestrator.step.completed"           # Pipeline step completed
    ORCHESTRATOR_STEP_FAILED = "orchestrator.step.failed"                 # Pipeline step failed
    
    # =========================================================================
    # SORA (AI Video Generation) SERVICE
    # =========================================================================
    SORA_USAGE_CHECK_REQUESTED = "sora.usage.check.requested"    # Request usage check
    SORA_USAGE_CHECKED = "sora.usage.checked"                    # Usage data retrieved
    SORA_USAGE_LOW = "sora.usage.low"                            # Video gens running low (<5)
    
    SORA_VIDEO_REQUESTED = "sora.video.requested"                # Request video generation
    SORA_VIDEO_STARTED = "sora.video.started"                    # Video generation started
    SORA_VIDEO_PROGRESS = "sora.video.progress"                  # Generation progress
    SORA_VIDEO_COMPLETED = "sora.video.completed"                # Video ready
    SORA_VIDEO_FAILED = "sora.video.failed"                      # Generation failed
    SORA_VIDEO_DOWNLOADED = "sora.video.downloaded"              # Video downloaded locally
    
    SORA_BATCH_REQUESTED = "sora.batch.requested"                # Batch video request (ARCH-002)
    SORA_BATCH_STARTED = "sora.batch.started"                    # Batch processing started (ARCH-002)
    SORA_BATCH_PROGRESS = "sora.batch.progress"                  # Batch progress (ARCH-002)
    SORA_BATCH_COMPLETED = "sora.batch.completed"                # All videos complete (ARCH-002)
    SORA_BATCH_FAILED = "sora.batch.failed"                      # Batch generation failed (ARCH-002)
    
    SORA_POLL_STARTED = "sora.poll.started"                      # Polling for completion started
    SORA_POLL_TICK = "sora.poll.tick"                            # Polling heartbeat
    SORA_POLL_STOPPED = "sora.poll.stopped"                      # Polling stopped
    
    # =========================================================================
    # MUSIC SERVICE
    # =========================================================================
    MUSIC_REQUESTED = "music.requested"                  # New music request
    MUSIC_STARTED = "music.started"                      # Music job started
    MUSIC_SEARCHING = "music.searching"                  # Searching for music
    MUSIC_DOWNLOADING = "music.downloading"              # Downloading music
    MUSIC_PROGRESS = "music.progress"                    # Progress update
    MUSIC_COMPLETED = "music.completed"                  # Music ready
    MUSIC_FAILED = "music.failed"                        # Music error
    
    # =========================================================================
    # VISUALS SERVICE
    # =========================================================================
    VISUALS_REQUESTED = "visuals.requested"              # New visuals request
    VISUALS_STARTED = "visuals.started"                  # Visuals job started
    VISUALS_FETCHING = "visuals.fetching"                # Fetching visuals
    VISUALS_PROCESSING = "visuals.processing"            # Processing visuals
    VISUALS_PROGRESS = "visuals.progress"                # Progress update
    VISUALS_COMPLETED = "visuals.completed"              # Visuals ready
    VISUALS_FAILED = "visuals.failed"                    # Visuals error
    
    # =========================================================================
    # AUTO-ENGAGEMENT (Safari Browser Automation)
    # =========================================================================
    ENGAGEMENT_REQUESTED = "engagement.requested"              # Request to engage with platform
    ENGAGEMENT_STARTED = "engagement.started"                  # Worker picked up job
    ENGAGEMENT_POST_FOUND = "engagement.post_found"            # Found a post to engage with
    ENGAGEMENT_COMMENT_GENERATED = "engagement.comment_generated"  # AI generated a comment
    ENGAGEMENT_COMMENT_POSTED = "engagement.comment_posted"    # Comment successfully posted
    ENGAGEMENT_COMMENT_SKIPPED = "engagement.comment_skipped"  # Post skipped (duplicate/limit)
    ENGAGEMENT_COMPLETED = "engagement.completed"              # Engagement session completed
    ENGAGEMENT_FAILED = "engagement.failed"                    # Engagement failed
    ENGAGEMENT_DAILY_LIMIT_REACHED = "engagement.daily_limit_reached"  # Daily limit hit
    ENGAGEMENT_PAUSED = "engagement.paused"                    # Engagement paused
    ENGAGEMENT_RESUMED = "engagement.resumed"                  # Engagement resumed
    
    # =========================================================================
    # SYSTEM
    # =========================================================================
    SYSTEM_STARTUP = "system.startup"                     # Backend started
    SYSTEM_SHUTDOWN = "system.shutdown"                   # Backend stopping
    SYSTEM_HEALTH_CHECK = "system.health.check"           # Health check event
    WORKER_STARTED = "worker.started"                     # Worker came online
    WORKER_STOPPED = "worker.stopped"                     # Worker went offline
    SERVICE_STARTED = "service.started"                   # Generic service started
    SERVICE_STOPPED = "service.stopped"                   # Generic service stopped

    # =========================================================================
    # SLEEP/WAKE MODE (CPU Efficiency)
    # =========================================================================
    SLEEP_SERVICE_STARTED = "sleep.service.started"       # Sleep mode service started
    SLEEP_SERVICE_STOPPED = "sleep.service.stopped"       # Sleep mode service stopped
    SLEEP_ENTERED = "sleep.entered"                       # System entered sleep mode
    SLEEP_WAKE = "sleep.wake"                             # System woke from sleep
    SLEEP_WAKE_SCHEDULED = "sleep.wake.scheduled"         # Wake event scheduled
    SLEEP_WAKE_CANCELLED = "sleep.wake.cancelled"         # Wake event cancelled

    # =========================================================================
    # CONTENT OPS ENTITIES (Brand → Offer → ICP)
    # =========================================================================
    BRAND_CREATED = "brand.created"                       # New brand created
    BRAND_UPDATED = "brand.updated"                       # Brand updated
    BRAND_DELETED = "brand.deleted"                       # Brand deleted
    OFFER_CREATED = "offer.created"                       # New offer created
    OFFER_UPDATED = "offer.updated"                       # Offer updated
    OFFER_DELETED = "offer.deleted"                       # Offer deleted
    ICP_CREATED = "icp.created"                          # New ICP created
    ICP_UPDATED = "icp.updated"                          # ICP updated
    ICP_DELETED = "icp.deleted"                          # ICP deleted

    # =========================================================================
    # DM PERMISSIONS (OPS-017, OPS-018)
    # =========================================================================
    DM_CONSENT_REQUESTED = "dm.consent.requested"         # Consent requested from contact
    DM_CONSENT_GRANTED = "dm.consent.granted"             # Consent granted (can send links)
    DM_CONSENT_DENIED = "dm.consent.denied"               # Consent denied (no links)
    DM_CONTACT_STOPPED = "dm.contact.stopped"             # Contact said "stop" (no messages)

    # =========================================================================
    # TOUCHPOINTS (Unified Model for All Channels)
    # =========================================================================
    TOUCHPOINT_CREATED = "touchpoint.created"             # New touchpoint created
    TOUCHPOINT_UPDATED = "touchpoint.updated"             # Touchpoint metrics updated
    TOUCHPOINT_DELETED = "touchpoint.deleted"             # Touchpoint deleted

    # =========================================================================
    # MEDIA FACTORY (MF-001)
    # =========================================================================
    MEDIA_FACTORY_JOB_CREATED = "media_factory.job.created"           # New production job created
    MEDIA_FACTORY_JOB_STARTED = "media_factory.job.started"           # Job execution started
    MEDIA_FACTORY_JOB_STAGE_STARTED = "media_factory.job.stage.started"       # Pipeline stage started
    MEDIA_FACTORY_JOB_STAGE_COMPLETED = "media_factory.job.stage.completed"   # Pipeline stage completed
    MEDIA_FACTORY_JOB_STAGE_FAILED = "media_factory.job.stage.failed"         # Pipeline stage failed
    MEDIA_FACTORY_JOB_COMPLETED = "media_factory.job.completed"       # Job completed successfully
    MEDIA_FACTORY_JOB_FAILED = "media_factory.job.failed"             # Job failed
    MEDIA_FACTORY_JOB_CANCELLED = "media_factory.job.cancelled"       # Job cancelled
    MEDIA_FACTORY_STAGE_STARTED = "media_factory.stage.started"       # Stage started (alias for compatibility)
    MEDIA_FACTORY_STAGE_COMPLETED = "media_factory.stage.completed"   # Stage completed (alias for compatibility)

    @classmethod
    def all_topics(cls) -> list:
        """Return list of all defined topics."""
        return [
            value for name, value in vars(cls).items()
            if isinstance(value, str) and not name.startswith('_')
        ]
    
    @classmethod
    def get_domain(cls, topic: str) -> str:
        """Extract domain from topic (first segment)."""
        return topic.split('.')[0] if '.' in topic else topic
    
    @classmethod
    def matches_pattern(cls, pattern: str, topic: str) -> bool:
        """
        Check if topic matches pattern with wildcard support.
        
        Patterns:
            - "media.*" matches "media.ingested", "media.analysis.completed"
            - "*.completed" matches "publish.completed", "analysis.completed"
            - "*" matches everything
        """
        if pattern == "*":
            return True
        
        if "*" not in pattern:
            return pattern == topic
        
        # Convert glob pattern to regex-like matching
        pattern_parts = pattern.split('.')
        topic_parts = topic.split('.')
        
        # Handle "prefix.*" patterns
        if pattern.endswith('.*'):
            prefix = pattern[:-2]
            return topic.startswith(prefix + '.')
        
        # Handle "*.suffix" patterns
        if pattern.startswith('*.'):
            suffix = pattern[2:]
            return topic.endswith('.' + suffix) or topic == suffix
        
        return False
