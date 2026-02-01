"""
Media Pipeline Service - Media analysis, thumbnails, extraction, deduplication.
Port: 6004
"""
import os
import subprocess
import json
from flask import Flask, jsonify, request
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# Add services to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import real service implementations
try:
    from services.detection.format_detector import FormatDetector
    FORMAT_DETECTOR = FormatDetector()
    FORMAT_DETECTOR_AVAILABLE = True
except Exception as e:
    print(f"FormatDetector not available: {e}")
    FORMAT_DETECTOR_AVAILABLE = False
    FORMAT_DETECTOR = None

try:
    from services.extraction.duplicate_detector import DuplicateDetector
    DUPLICATE_DETECTOR = DuplicateDetector()
    DUPLICATE_DETECTOR_AVAILABLE = True
except Exception as e:
    print(f"DuplicateDetector not available: {e}")
    DUPLICATE_DETECTOR_AVAILABLE = False
    DUPLICATE_DETECTOR = None

SERVICE_NAME = "media-pipeline"
SERVICE_VERSION = "1.0.0"
SERVICE_PORT = int(os.getenv("PORT", 6004))


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """Analyze a video file using ffprobe."""
    data = request.get_json()
    video_path = data.get("video_path")
    
    if not video_path:
        return jsonify({"error": "video_path required"}), 400
    
    # Use ffprobe for real analysis
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", video_path
            ],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            probe_data = json.loads(result.stdout)
            format_info = probe_data.get("format", {})
            streams = probe_data.get("streams", [])
            
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
            
            return jsonify({
                "status": "success",
                "video_path": video_path,
                "analysis": {
                    "duration": float(format_info.get("duration", 0)),
                    "format": format_info.get("format_name", "unknown"),
                    "size_bytes": int(format_info.get("size", 0)),
                    "bitrate": int(format_info.get("bit_rate", 0)),
                    "video": {
                        "codec": video_stream.get("codec_name", "unknown"),
                        "width": video_stream.get("width", 0),
                        "height": video_stream.get("height", 0),
                        "fps": video_stream.get("r_frame_rate", "0/1")
                    },
                    "audio": {
                        "codec": audio_stream.get("codec_name", "unknown"),
                        "sample_rate": audio_stream.get("sample_rate", "0"),
                        "channels": audio_stream.get("channels", 0)
                    }
                }
            })
        else:
            return jsonify({
                "status": "error",
                "error": "ffprobe failed",
                "video_path": video_path
            }), 400
    except FileNotFoundError:
        return jsonify({"status": "error", "error": "ffprobe not installed"}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "error": "Analysis timed out"}), 504
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/thumbnail/generate", methods=["POST"])
def generate_thumbnails():
    """Generate thumbnails from video using ffmpeg."""
    data = request.get_json()
    video_path = data.get("video_path")
    count = data.get("count", 5)
    output_dir = data.get("output_dir", "/tmp/media-pipeline/thumbnails")
    
    if not video_path:
        return jsonify({"error": "video_path required"}), 400
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Get video duration first
        probe_result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=10
        )
        duration = float(probe_result.stdout.strip()) if probe_result.returncode == 0 else 60
        
        thumbnails = []
        interval = duration / (count + 1)
        
        for i in range(count):
            timestamp = interval * (i + 1)
            output_file = f"{output_dir}/thumb_{i:03d}.jpg"
            
            result = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
                 "-vframes", "1", "-q:v", "2", output_file],
                capture_output=True, timeout=30
            )
            
            if result.returncode == 0 and Path(output_file).exists():
                thumbnails.append({
                    "path": output_file,
                    "timestamp": timestamp,
                    "index": i
                })
        
        return jsonify({
            "status": "success",
            "video_path": video_path,
            "thumbnails": thumbnails,
            "count": len(thumbnails)
        })
    except FileNotFoundError:
        return jsonify({"status": "error", "error": "ffmpeg not installed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/format/detect", methods=["POST"])
def detect_format():
    """Detect media format using real FormatDetector."""
    data = request.get_json()
    file_path = data.get("file_path")
    transcript = data.get("transcript", "")
    visual_analysis = data.get("visual_analysis", {})
    
    if not file_path:
        return jsonify({"error": "file_path required"}), 400
    
    if FORMAT_DETECTOR_AVAILABLE and FORMAT_DETECTOR:
        try:
            # Use real format detector
            result = FORMAT_DETECTOR.detect_format(
                transcript=transcript,
                visual_analysis=visual_analysis,
                duration_sec=data.get("duration")
            )
            return jsonify({
                "status": "success",
                "file_path": file_path,
                "format": {
                    "primary_format": result.primary_format.value,
                    "confidence": round(result.confidence, 2),
                    "has_speech": result.has_speech,
                    "has_music": result.has_music,
                    "has_people": result.has_people,
                    "is_vertical": result.is_vertical,
                    "production_quality": result.production_quality.value,
                    "best_platforms": result.best_platforms,
                    "reasons": result.reasons[:5]
                },
                "implementation": "real"
            })
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500
    else:
        # Fallback - use ffprobe for basic detection
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", file_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                probe = json.loads(result.stdout)
                fmt = probe.get("format", {})
                streams = probe.get("streams", [])
                video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
                
                width = video_stream.get("width", 0)
                height = video_stream.get("height", 0)
                is_vertical = height > width if width and height else False
                
                return jsonify({
                    "status": "success",
                    "file_path": file_path,
                    "format": {
                        "primary_format": "unknown",
                        "container": fmt.get("format_name", "unknown"),
                        "codec": video_stream.get("codec_name", "unknown"),
                        "is_vertical": is_vertical,
                        "width": width,
                        "height": height
                    },
                    "implementation": "ffprobe"
                })
        except Exception:
            pass
        
        return jsonify({
            "status": "success",
            "file_path": file_path,
            "format": {"type": "video", "codec": "unknown", "container": "mp4"},
            "implementation": "placeholder"
        })


@app.route("/api/clip/extract", methods=["POST"])
def extract_clip():
    """Extract a clip from video using ffmpeg."""
    data = request.get_json()
    video_path = data.get("video_path")
    start_time = data.get("start_time", 0)
    end_time = data.get("end_time")
    output_path = data.get("output_path")
    
    if not video_path:
        return jsonify({"error": "video_path required"}), 400
    
    if not os.path.exists(video_path):
        return jsonify({"error": "video file not found"}), 404
    
    if end_time is None:
        return jsonify({"error": "end_time required"}), 400
    
    # Generate output path if not provided
    if not output_path:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_name}_clip_{start_time}_{end_time}.mp4")
    
    try:
        # Calculate duration
        duration = float(end_time) - float(start_time)
        
        # Use ffmpeg to extract clip
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "error": f"ffmpeg failed: {result.stderr[:500]}"
            }), 500
        
        # Get clip info
        clip_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        
        return jsonify({
            "status": "success",
            "video_path": video_path,
            "clip_path": output_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "size_bytes": clip_size,
            "implementation": "ffmpeg"
        })
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "error": "ffmpeg timeout"}), 500
    except FileNotFoundError:
        return jsonify({"status": "error", "error": "ffmpeg not installed"}), 500
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/deduplicate/check", methods=["POST"])
def check_duplicate():
    """Check if content is a duplicate using real DuplicateDetector."""
    data = request.get_json()
    file_path = data.get("file_path")
    content_hash = data.get("content_hash", "")
    
    if not file_path:
        return jsonify({"error": "file_path required"}), 400
    
    if DUPLICATE_DETECTOR_AVAILABLE and DUPLICATE_DETECTOR:
        try:
            # Use real duplicate detector
            is_dup, similarity, match_id = DUPLICATE_DETECTOR.check_duplicate(
                file_path=file_path,
                content_hash=content_hash
            )
            return jsonify({
                "status": "success",
                "file_path": file_path,
                "is_duplicate": is_dup,
                "similarity_score": round(similarity, 2) if similarity else 0.0,
                "matched_content_id": match_id,
                "implementation": "real"
            })
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500
    else:
        # Fallback - compute file hash for basic dedup
        try:
            import hashlib
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read(8192)).hexdigest()
            return jsonify({
                "status": "success",
                "file_path": file_path,
                "is_duplicate": False,
                "similarity_score": 0.0,
                "file_hash": file_hash,
                "implementation": "hash"
            })
        except Exception:
            pass
        
        return jsonify({
            "status": "success",
            "file_path": file_path,
            "is_duplicate": False,
            "similarity_score": 0.0,
            "implementation": "placeholder"
        })


@app.route("/api/transcribe", methods=["POST"])
def transcribe_video():
    """Transcribe video/audio using OpenAI Whisper API."""
    data = request.get_json()
    video_path = data.get("video_path")
    language = data.get("language")
    
    if not video_path:
        return jsonify({"error": "video_path required"}), 400
    
    if not os.path.exists(video_path):
        return jsonify({"error": "video file not found"}), 404
    
    # Try to use real transcription service
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        if api_key:
            client = OpenAI(api_key=api_key)
            
            with open(video_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"]
                )
            
            return jsonify({
                "status": "success",
                "video_path": video_path,
                "text": response.text,
                "language": response.language,
                "duration": response.duration,
                "words": [{"word": w.word, "start": w.start, "end": w.end} for w in (response.words or [])],
                "segments": [{"text": s.text, "start": s.start, "end": s.end} for s in (response.segments or [])],
                "implementation": "whisper"
            })
        else:
            return jsonify({
                "status": "success",
                "video_path": video_path,
                "text": "",
                "words": [],
                "segments": [],
                "implementation": "placeholder",
                "note": "Set OPENAI_API_KEY for real transcription"
            })
    except ImportError:
        return jsonify({
            "status": "error",
            "error": "openai package not installed"
        }), 500
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/audio/analyze", methods=["POST"])
def analyze_audio():
    """Analyze audio in video file using ffprobe."""
    data = request.get_json()
    video_path = data.get("video_path")
    
    if not video_path:
        return jsonify({"error": "video_path required"}), 400
    
    if not os.path.exists(video_path):
        return jsonify({"error": "video file not found"}), 404
    
    try:
        # Use ffprobe for audio analysis
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "error": "ffprobe failed",
                "details": result.stderr
            }), 500
        
        probe_data = json.loads(result.stdout)
        audio_streams = probe_data.get("streams", [])
        
        if not audio_streams:
            return jsonify({
                "status": "success",
                "video_path": video_path,
                "has_audio": False,
                "audio_streams": [],
                "implementation": "ffprobe"
            })
        
        # Get audio info from first stream
        audio = audio_streams[0]
        
        # Try to get loudness with ffmpeg
        loudness_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-af", "volumedetect",
            "-f", "null",
            "-"
        ]
        loudness_result = subprocess.run(loudness_cmd, capture_output=True, text=True, timeout=60)
        
        # Parse loudness info
        mean_volume = None
        max_volume = None
        for line in loudness_result.stderr.split('\n'):
            if 'mean_volume' in line:
                try:
                    mean_volume = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                except:
                    pass
            if 'max_volume' in line:
                try:
                    max_volume = float(line.split('max_volume:')[1].split('dB')[0].strip())
                except:
                    pass
        
        return jsonify({
            "status": "success",
            "video_path": video_path,
            "has_audio": True,
            "audio_info": {
                "codec": audio.get("codec_name"),
                "channels": audio.get("channels"),
                "sample_rate": audio.get("sample_rate"),
                "bit_rate": audio.get("bit_rate"),
                "duration": audio.get("duration")
            },
            "loudness": {
                "mean_volume_db": mean_volume,
                "max_volume_db": max_volume
            },
            "audio_streams": len(audio_streams),
            "implementation": "ffprobe"
        })
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "error": "Analysis timed out"}), 500
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# Import video orchestrator
try:
    from services.video_orchestrator.director import DirectorService, DirectorConfig
    DIRECTOR_SERVICE = DirectorService(DirectorConfig())
    DIRECTOR_AVAILABLE = True
except Exception as e:
    print(f"DirectorService not available: {e}")
    DIRECTOR_AVAILABLE = False
    DIRECTOR_SERVICE = None

# Import video renderer
try:
    from services.video_renderer.renderer import VideoRenderer
    VIDEO_RENDERER_AVAILABLE = True
except Exception as e:
    print(f"VideoRenderer not available: {e}")
    VIDEO_RENDERER_AVAILABLE = False


@app.route("/api/orchestrate/plan", methods=["POST"])
def create_clip_plan():
    """Create a clip plan from a script using DirectorService."""
    if not DIRECTOR_AVAILABLE:
        return jsonify({"error": "DirectorService not available"}), 503
    
    data = request.get_json()
    script_text = data.get("script")
    title = data.get("title", "Untitled")
    
    if not script_text:
        return jsonify({"error": "script required"}), 400
    
    try:
        # Create clip plan from script
        from services.video_orchestrator.models import PlanConstraints, PacingConstraints
        
        constraints = PlanConstraints(
            target_duration=data.get("target_duration", 60),
            pacing=PacingConstraints(
                words_per_minute=data.get("wpm", 150),
                min_clip_seconds=data.get("min_clip", 4),
                max_clip_seconds=data.get("max_clip", 12)
            )
        )
        
        clip_plan = DIRECTOR_SERVICE.create_clip_plan(
            script_text=script_text,
            title=title,
            constraints=constraints
        )
        
        return jsonify({
            "status": "success",
            "clip_plan": {
                "id": str(clip_plan.id) if hasattr(clip_plan, 'id') else "generated",
                "title": title,
                "clip_count": len(clip_plan.clips) if hasattr(clip_plan, 'clips') else 0,
                "total_duration": sum(c.duration for c in clip_plan.clips) if hasattr(clip_plan, 'clips') else 0
            },
            "implementation": "DirectorService"
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/tts/generate", methods=["POST"])
def generate_tts():
    """Generate text-to-speech audio."""
    data = request.get_json()
    text = data.get("text")
    voice = data.get("voice", "default")
    output_path = data.get("output_path")
    
    if not text:
        return jsonify({"error": "text required"}), 400
    
    try:
        from services.tts.worker import TTSWorker
        worker = TTSWorker()
        
        result = worker.generate(
            text=text,
            voice=voice,
            output_path=output_path
        )
        
        return jsonify({
            "status": "success",
            "audio_path": result.get("path") if isinstance(result, dict) else str(result),
            "duration": result.get("duration") if isinstance(result, dict) else None,
            "implementation": "TTSWorker"
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/music/search", methods=["POST"])
def search_music():
    """Search for music tracks."""
    data = request.get_json()
    query = data.get("query")
    mood = data.get("mood")
    genre = data.get("genre")
    
    try:
        from services.music.worker import MusicWorker
        worker = MusicWorker()
        
        results = worker.search(
            query=query,
            mood=mood,
            genre=genre,
            limit=data.get("limit", 10)
        )
        
        return jsonify({
            "status": "success",
            "tracks": results if isinstance(results, list) else [],
            "count": len(results) if isinstance(results, list) else 0,
            "implementation": "MusicWorker"
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/sfx/search", methods=["POST"])
def search_sfx():
    """Search for sound effects."""
    data = request.get_json()
    query = data.get("query")
    category = data.get("category")
    
    try:
        from services.sfx_library.sfx_service import SFXService
        service = SFXService()
        
        results = service.search(
            query=query,
            category=category,
            limit=data.get("limit", 20)
        )
        
        return jsonify({
            "status": "success",
            "effects": results if isinstance(results, list) else [],
            "count": len(results) if isinstance(results, list) else 0,
            "implementation": "SFXService"
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/render/video", methods=["POST"])
def render_video():
    """Render a video from a clip plan or timeline."""
    data = request.get_json()
    timeline = data.get("timeline")
    output_path = data.get("output_path")
    format_type = data.get("format", "mp4")
    
    if not timeline:
        return jsonify({"error": "timeline required"}), 400
    
    try:
        from services.video_renderer.renderer import VideoRenderer
        renderer = VideoRenderer()
        
        result = renderer.render(
            timeline=timeline,
            output_path=output_path,
            format=format_type
        )
        
        return jsonify({
            "status": "success",
            "output_path": result.get("path") if isinstance(result, dict) else str(result),
            "duration": result.get("duration") if isinstance(result, dict) else None,
            "implementation": "VideoRenderer"
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    print(f"🚀 {SERVICE_NAME} starting on port {SERVICE_PORT}")
    app.run(host="0.0.0.0", port=SERVICE_PORT, debug=True)
