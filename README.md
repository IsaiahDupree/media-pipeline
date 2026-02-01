# Media Services

Media analysis, processing, and extraction microservice for MediaPoster ecosystem.

## Services

- **Thumbnails** - Frame extraction, thumbnail generation, AI selection
- **Detection** - Format detection, classification
- **Extraction** - Clip extraction, deduplication
- **Matting** - Background removal, matting
- **Formats** - Format conversion utilities

## Structure

```
services/
├── thumbnails/     # Thumbnail generation
├── detection/      # Format detection
├── extraction/     # Clip extraction, dedup
├── matting/        # Background removal
├── formats/        # Format utilities
└── clip_extraction/ # Clip extraction pipeline
```

## Port

Default: `:6004`

## Related Repos

- MediaPoster (Core) - Scheduling, publishing
- Remotion - Video rendering
- Safari Automation - Browser automation
