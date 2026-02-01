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

# Import real service implementations
try:
    from services.detection.format_detector import FormatDetector
    FORMAT_DETECTOR_AVAILABLE = True
except ImportError:
    FORMAT_DETECTOR_AVAILABLE = False

try:
    from services.extraction.duplicate_detector import DuplicateDetector
    DUPLICATE_DETECTOR_AVAILABLE = True
except ImportError:
    DUPLICATE_DETECTOR_AVAILABLE = False

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
    """Detect media format."""
    data = request.get_json()
    file_path = data.get("file_path")
    
    if not file_path:
        return jsonify({"error": "file_path required"}), 400
    
    # TODO: Implement format detection
    return jsonify({
        "status": "success",
        "file_path": file_path,
        "format": {
            "type": "video",
            "codec": "unknown",
            "container": "mp4"
        }
    })


@app.route("/api/clip/extract", methods=["POST"])
def extract_clip():
    """Extract a clip from video."""
    data = request.get_json()
    video_path = data.get("video_path")
    start_time = data.get("start_time", 0)
    end_time = data.get("end_time")
    
    if not video_path:
        return jsonify({"error": "video_path required"}), 400
    
    # TODO: Implement clip extraction
    return jsonify({
        "status": "success",
        "video_path": video_path,
        "clip_path": None,
        "start_time": start_time,
        "end_time": end_time
    })


@app.route("/api/deduplicate/check", methods=["POST"])
def check_duplicate():
    """Check if content is a duplicate."""
    data = request.get_json()
    file_path = data.get("file_path")
    
    if not file_path:
        return jsonify({"error": "file_path required"}), 400
    
    # TODO: Implement deduplication check
    return jsonify({
        "status": "success",
        "file_path": file_path,
        "is_duplicate": False,
        "similarity_score": 0.0
    })


if __name__ == "__main__":
    print(f"🚀 {SERVICE_NAME} starting on port {SERVICE_PORT}")
    app.run(host="0.0.0.0", port=SERVICE_PORT, debug=True)
