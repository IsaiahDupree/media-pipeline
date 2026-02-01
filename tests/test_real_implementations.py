"""
Tests for real service implementations in media-pipeline.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFormatDetector:
    """Test the real FormatDetector implementation."""
    
    def test_import(self):
        """Test that FormatDetector can be imported."""
        from services.detection.format_detector import FormatDetector
        detector = FormatDetector()
        assert detector is not None
    
    def test_detect_talking_head(self):
        """Test talking head format detection."""
        from services.detection.format_detector import FormatDetector
        detector = FormatDetector()
        
        # Long transcript indicates talking head
        transcript = """
        Hello everyone, welcome back to my channel. Today I want to talk to you about 
        something really important. I've been thinking about this for a while and I 
        wanted to share my thoughts with you directly. So let me explain what I mean 
        by this concept and why it matters for all of you watching.
        """
        
        result = detector.detect_format(
            transcript=transcript,
            visual_analysis={"visual_summary": "person speaking to camera"},
            duration_sec=120
        )
        
        assert result.primary_format is not None
        assert 0 <= result.confidence <= 1
        assert result.has_speech == True
    
    def test_detect_broll(self):
        """Test b-roll format detection."""
        from services.detection.format_detector import FormatDetector
        detector = FormatDetector()
        
        # No transcript indicates b-roll
        result = detector.detect_format(
            transcript="",
            visual_analysis={"visual_summary": "landscape, mountains, sunset"},
            duration_sec=30
        )
        
        assert result.primary_format is not None
        assert result.has_speech == False
    
    def test_detect_screen_recording(self):
        """Test screen recording detection."""
        from services.detection.format_detector import FormatDetector
        detector = FormatDetector()
        
        result = detector.detect_format(
            transcript="So here I'm going to click on this button",
            visual_analysis={"visual_summary": "computer screen, cursor, software interface"},
            duration_sec=180
        )
        
        assert result.primary_format is not None
        assert 0 <= result.confidence <= 1
    
    def test_content_format_enum(self):
        """Test ContentFormat enum values."""
        from services.detection.format_detector import ContentFormat
        
        expected_formats = [
            "talking_head", "interview", "broll_scenic", "broll_action",
            "broll_people", "animated", "screen_recording", "slideshow",
            "music_video", "montage", "documentary", "reaction",
            "tutorial_hands", "live_event", "meme_content", "unknown"
        ]
        
        for fmt in ContentFormat:
            assert fmt.value in expected_formats


class TestDuplicateDetector:
    """Test the DuplicateDetector implementation."""
    
    def test_import(self):
        """Test that DuplicateDetector can be imported."""
        try:
            from services.extraction.duplicate_detector import DuplicateDetector
            detector = DuplicateDetector()
            assert detector is not None
        except ImportError as e:
            pytest.skip(f"DuplicateDetector not available: {e}")


class TestFlaskApp:
    """Test the Flask application endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from app import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert data['service'] == 'media-pipeline'
    
    def test_format_detect_missing_path(self, client):
        """Test format detection with missing file_path."""
        response = client.post('/api/format/detect', json={})
        assert response.status_code == 400
    
    def test_format_detect_with_transcript(self, client):
        """Test format detection with transcript."""
        response = client.post('/api/format/detect', json={
            "file_path": "/tmp/test.mp4",
            "transcript": "Hello everyone, today I want to talk about content creation."
        })
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
