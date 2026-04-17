"""
Database migrations for media pipeline.
Defines schema for media metadata and processing jobs.
"""


MIGRATIONS = {
    "001_create_media_table": """
    CREATE TABLE IF NOT EXISTS media (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        file_path VARCHAR NOT NULL,
        file_size INTEGER DEFAULT 0,
        format VARCHAR DEFAULT 'unknown',
        duration FLOAT DEFAULT 0,
        width INTEGER DEFAULT 0,
        height INTEGER DEFAULT 1080,
        codec VARCHAR DEFAULT 'unknown',
        bitrate INTEGER DEFAULT 0,
        fps FLOAT DEFAULT 30,
        status VARCHAR DEFAULT 'pending',
        metadata JSONB DEFAULT '{}'::jsonb,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        created_by VARCHAR,
        INDEX idx_status (status),
        INDEX idx_created_at (created_at)
    );
    """,

    "002_create_jobs_table": """
    CREATE TABLE IF NOT EXISTS processing_jobs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        media_id UUID REFERENCES media(id) ON DELETE CASCADE,
        operation VARCHAR NOT NULL,
        status VARCHAR DEFAULT 'pending',
        progress FLOAT DEFAULT 0,
        error TEXT,
        params JSONB DEFAULT '{}'::jsonb,
        result JSONB DEFAULT '{}'::jsonb,
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        INDEX idx_media_id (media_id),
        INDEX idx_status (status),
        INDEX idx_operation (operation)
    );
    """,

    "003_create_storage_bucket": """
    -- Note: This would be executed via Supabase API, not SQL
    -- Bucket name: media-pipeline
    -- Public: true
    -- File size limit: 5GB
    -- Allowed MIME types: video/*, image/*
    """,

    "004_create_indexes": """
    CREATE INDEX idx_media_format_status ON media(format, status);
    CREATE INDEX idx_jobs_media_operation ON processing_jobs(media_id, operation);
    CREATE INDEX idx_jobs_created_at ON processing_jobs(created_at DESC);
    """
}


def create_migrations() -> dict:
    """Get all migrations."""
    return MIGRATIONS


def get_migration(name: str) -> str:
    """Get a specific migration SQL."""
    return MIGRATIONS.get(name, "")


def get_all_migrations_sql() -> str:
    """Get all migrations as a single SQL string."""
    migrations = []
    for name in sorted(MIGRATIONS.keys()):
        sql = MIGRATIONS[name].strip()
        if sql and not sql.startswith("--"):
            migrations.append(sql)
    return ";\n".join(migrations) + ";"
