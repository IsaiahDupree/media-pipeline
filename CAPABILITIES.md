# Media Pipeline Capabilities

## Service Info
- **Port:** 6004
- **Health:** `GET /health`

## Endpoints

### Video Analysis
```http
POST /api/analyze
{ "video_path": "/path/to/video.mp4" }
```

### Thumbnail Generation
```http
POST /api/thumbnail/generate
{ "video_path": "/path/to/video.mp4", "count": 5 }
```

### Format Detection
```http
POST /api/format/detect
{ "file_path": "/path/to/media.mp4" }
```

### Clip Extraction
```http
POST /api/clip/extract
{ "video_path": "/path/to/video.mp4", "start_time": 0, "end_time": 30 }
```

### Deduplication Check
```http
POST /api/deduplicate/check
{ "file_path": "/path/to/video.mp4" }
```

## Capabilities Summary

| Capability | Status | Description |
|------------|--------|-------------|
| Video Analysis | ✅ Ready | Analyze video metadata, duration, format |
| Thumbnail Generation | ✅ Ready | Extract frames, generate thumbnails |
| Format Detection | ✅ Ready | Detect media format and codec |
| Clip Extraction | ✅ Ready | Extract segments from video |
| Deduplication | ✅ Ready | Content fingerprinting and matching |
| Background Removal | 🔄 Planned | Matting and background removal |
