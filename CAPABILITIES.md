# Media Pipeline Service - Capabilities

## Overview
Video/media processing service using ffmpeg, ffprobe, and AI-powered analysis.

**Port**: 6004  
**Repository**: https://github.com/IsaiahDupree/media-pipeline  
**Total Code**: 71,826 lines (moved from MediaPoster)

## Real Implementations

| Endpoint | Implementation | Source |
|----------|---------------|--------|
| `/api/analyze` | ✅ **ffprobe** | Real video analysis |
| `/api/thumbnail/generate` | ✅ **ffmpeg** | Real frame extraction |
| `/api/format/detect` | ✅ **FormatDetector** | `services/detection/format_detector.py` |
| `/api/clip/extract` | ✅ **ffmpeg** | Real clip extraction |
| `/api/deduplicate/check` | ✅ **DuplicateDetector** | `services/detection/duplicate_detector.py` |
| `/api/transcribe` | ✅ **Whisper** | OpenAI Whisper API |

## Services Copied from MediaPoster

### Analysis Services (2,034 lines)
| File | Lines | Purpose |
|------|-------|---------|
| `frame_analyzer.py` | 79 | Basic frame analysis |
| `frame_analyzer_enhanced.py` | 474 | Advanced frame analysis |
| `frame_sampler.py` | 265 | Smart frame sampling |
| `video_analyzer.py` | 519 | Complete video analysis |
| `batch_video_analyzer.py` | 408 | Batch processing |
| `video_analysis.py` | 289 | Video analysis utilities |

### Thumbnail Services (1,548 lines)
| File | Lines | Purpose |
|------|-------|---------|
| `thumbnail_generator.py` | 589 | Platform-specific thumbnails |
| `thumbnail_service.py` | 559 | Thumbnail orchestration |
| `ai_thumbnail_selector.py` | 400 | AI-powered selection |

### Detection Services (708 lines)
| File | Lines | Purpose |
|------|-------|---------|
| `format_detector.py` | 566 | 15 content format types |
| `broll_detector.py` | ~100 | B-roll detection |
| `duplicate_detector.py` | ~100 | Content fingerprinting |

### Transcription Services (1,124 lines)
| File | Lines | Purpose |
|------|-------|---------|
| `transcription.py` | 236 | Whisper transcription |
| `whisper_transcriber.py` | 195 | Whisper API wrapper |
| `transcription_adapter.py` | 693 | Multi-provider adapter |

### SFX Library (3,732 lines)
| File | Purpose |
|------|---------|
| `audio_mixer.py` | Audio mixing |
| `beat_extractor.py` | Beat detection |
| `cue_sheet.py` | Cue sheet generation |
| `llm_integration.py` | AI sound selection |
| `macros.py` | Audio macros |

### Video Generation (14,514 lines)
| File | Purpose |
|------|---------|
| `full_pipeline.py` | Complete video pipeline |
| `orchestrator.py` | Pipeline orchestration |
| `shot_plan.py` | Shot planning |
| `remotion_video_pipeline.py` | Remotion integration |
| `voice_engine.py` | Voice generation |
| `gemini_video_pipeline.py` | Gemini video |

### Repurpose Services
| File | Purpose |
|------|---------|
| `clip_extractor.py` | Clip extraction |
| `pipeline.py` | Repurpose pipeline |
| `video_analyzer.py` | Video analysis |

## API Endpoints

### Health Check
```bash
curl http://localhost:6004/health
```

### Video Analysis (Real ffprobe)
```bash
curl -X POST http://localhost:6004/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'
```
**Response:**
```json
{
  "status": "success",
  "analysis": {
    "duration": 45.23,
    "width": 1920,
    "height": 1080,
    "codec": "h264",
    "fps": 30.0,
    "bitrate": 5000000
  }
}
```

### Thumbnail Generation (Real ffmpeg)
```bash
curl -X POST http://localhost:6004/api/thumbnail/generate \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4", "count": 5, "output_dir": "/tmp/thumbs"}'
```

### Format Detection (Real Implementation)
```bash
curl -X POST http://localhost:6004/api/format/detect \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/video.mp4", "transcript": "Hello everyone, today I want to talk about..."}'
```
**Response:**
```json
{
  "format": {
    "primary_format": "talking_head",
    "confidence": 0.85,
    "has_speech": true,
    "is_vertical": false,
    "production_quality": "medium"
  },
  "implementation": "real"
}
```

### Deduplication Check
```bash
curl -X POST http://localhost:6004/api/deduplicate/check \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/video.mp4"}'
```

## Architecture

```
media-pipeline/
├── app.py                    # Flask application
├── services/
│   ├── detection/
│   │   └── format_detector.py    # 15 content format types
│   └── extraction/
│       └── duplicate_detector.py # Content fingerprinting
├── config/
│   └── settings.py           # Environment configuration
└── shared/
    └── service_client.py     # Inter-service HTTP client
```

## Format Types Detected

| Format | Description |
|--------|-------------|
| `talking_head` | Person speaking to camera |
| `interview` | Two+ people in conversation |
| `broll_scenic` | Landscape/environment footage |
| `broll_action` | Movement/action footage |
| `screen_recording` | Software demo, gameplay |
| `tutorial_hands` | Hands-on tutorial |
| `documentary` | Narrated voiceover |
| `montage` | Quick cuts with music |
| `meme_content` | Meme-style edits |

## Dependencies
```
flask>=3.0.0
httpx>=0.27.0
python-dotenv>=1.0.0
opencv-python>=4.9.0
Pillow>=10.0.0
numpy>=1.26.0
loguru>=0.7.0
```

## System Requirements
- **ffmpeg** - For video processing
- **ffprobe** - For video analysis

## Environment Variables
```bash
PORT=6004
CONTENT_INTELLIGENCE_URL=http://localhost:6006
```
