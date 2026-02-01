"""
Tests for media-pipeline service health and endpoints.
"""
import pytest
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test health check returns healthy status."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'media-pipeline'
    assert 'version' in data
    assert 'timestamp' in data


def test_analyze_requires_video_path(client):
    """Test analyze endpoint requires video_path."""
    response = client.post('/api/analyze', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_analyze_with_video_path(client):
    """Test analyze endpoint with valid video_path."""
    response = client.post('/api/analyze', json={
        'video_path': '/path/to/video.mp4'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert 'analysis' in data


def test_thumbnail_generate(client):
    """Test thumbnail generation endpoint."""
    response = client.post('/api/thumbnail/generate', json={
        'video_path': '/path/to/video.mp4',
        'count': 3
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert data['count'] == 3


def test_format_detect(client):
    """Test format detection endpoint."""
    response = client.post('/api/format/detect', json={
        'file_path': '/path/to/video.mp4'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert 'format' in data


def test_clip_extract(client):
    """Test clip extraction endpoint."""
    response = client.post('/api/clip/extract', json={
        'video_path': '/path/to/video.mp4',
        'start_time': 0,
        'end_time': 30
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'


def test_deduplicate_check(client):
    """Test deduplication check endpoint."""
    response = client.post('/api/deduplicate/check', json={
        'file_path': '/path/to/video.mp4'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert 'is_duplicate' in data
