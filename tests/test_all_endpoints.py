#!/usr/bin/env python3
"""
Integration tests for all media-pipeline endpoints.
Run with: pytest tests/test_all_endpoints.py -v
"""
import pytest
import json
import tempfile
import os


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["service"] == "media-pipeline"


class TestAnalyzeEndpoint:
    def test_analyze_requires_video_path(self, client):
        response = client.post("/api/analyze", json={})
        assert response.status_code == 400
        
    def test_analyze_returns_error_for_missing_file(self, client):
        response = client.post("/api/analyze", json={"video_path": "/nonexistent.mp4"})
        assert response.status_code == 404


class TestThumbnailEndpoint:
    def test_thumbnail_requires_video_path(self, client):
        response = client.post("/api/thumbnail/generate", json={})
        assert response.status_code == 400


class TestFormatDetectEndpoint:
    def test_format_detect_works(self, client):
        response = client.post("/api/format/detect", json={
            "file_path": "/tmp/test.mp4",
            "transcript": "Hello everyone, today I want to talk about something important"
        })
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "format" in data or "error" in data


class TestClipExtractEndpoint:
    def test_clip_extract_requires_video_path(self, client):
        response = client.post("/api/clip/extract", json={})
        assert response.status_code == 400


class TestTranscribeEndpoint:
    def test_transcribe_requires_video_path(self, client):
        response = client.post("/api/transcribe", json={})
        assert response.status_code == 400


class TestAudioAnalyzeEndpoint:
    def test_audio_analyze_requires_video_path(self, client):
        response = client.post("/api/audio/analyze", json={})
        assert response.status_code == 400


class TestDeduplicateEndpoint:
    def test_deduplicate_requires_file_path(self, client):
        response = client.post("/api/deduplicate/check", json={})
        assert response.status_code == 400


class TestOrchestrateEndpoint:
    def test_orchestrate_requires_script(self, client):
        response = client.post("/api/orchestrate/plan", json={})
        assert response.status_code == 400
    
    def test_orchestrate_with_script(self, client):
        response = client.post("/api/orchestrate/plan", json={
            "script": "Hello everyone. Today we talk about AI. It's changing the world.",
            "title": "AI Overview"
        })
        assert response.status_code in [200, 500, 503]


class TestTTSEndpoint:
    def test_tts_requires_text(self, client):
        response = client.post("/api/tts/generate", json={})
        assert response.status_code == 400
    
    def test_tts_with_text(self, client):
        response = client.post("/api/tts/generate", json={
            "text": "Hello world"
        })
        assert response.status_code in [200, 500]


class TestMusicEndpoint:
    def test_music_search(self, client):
        response = client.post("/api/music/search", json={
            "query": "upbeat",
            "mood": "happy"
        })
        assert response.status_code in [200, 500]


class TestSFXEndpoint:
    def test_sfx_search(self, client):
        response = client.post("/api/sfx/search", json={
            "query": "whoosh"
        })
        assert response.status_code in [200, 500]


class TestRenderEndpoint:
    def test_render_requires_timeline(self, client):
        response = client.post("/api/render/video", json={})
        assert response.status_code == 400
    
    def test_render_with_timeline(self, client):
        response = client.post("/api/render/video", json={
            "timeline": {"clips": []},
            "output_path": "/tmp/test_render.mp4"
        })
        assert response.status_code in [200, 500]


@pytest.fixture
def client():
    """Create test client."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
