"""
Format Compiler
Resolves data sources, applies bindings, and compiles render props
"""
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .schema import (
    FormatDefinition,
    RenderProps,
    CompileResult,
    VideoConfig,
    Script,
    ScriptSegment,
    AudioConfig,
    DuckingConfig,
    VisualsConfig,
    StyleConfig,
    RenderMeta,
    Binding,
    DataSource,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DOT PATH UTILITIES
# =============================================================================

def get_path(obj: Any, path: str) -> Any:
    """Get a value from a nested object using dot notation with array support."""
    if not path:
        return obj
    
    parts = path.split(".")
    current = obj
    
    for part in parts:
        if current is None:
            return None
        
        # Handle array indexing: angles[0]
        match = re.match(r'^(.+)\[(\d+)\]$', part)
        if match:
            key, idx = match.groups()
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return None
            
            if isinstance(current, list) and int(idx) < len(current):
                current = current[int(idx)]
            else:
                return None
        else:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
    
    return current


def set_path(obj: Dict, path: str, value: Any) -> Dict:
    """Set a value in a nested dict using dot notation."""
    parts = path.split(".")
    current = obj
    
    for i, part in enumerate(parts[:-1]):
        if part not in current or not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    
    current[parts[-1]] = value
    return obj


def render_template(template: str, ctx: Dict) -> str:
    """Simple {{path}} template rendering."""
    def replacer(match):
        path = match.group(1).strip()
        value = get_path(ctx, path)
        return str(value) if value is not None else ""
    
    return re.sub(r'\{\{\s*([^\}]+)\s*\}\}', replacer, template)


# =============================================================================
# DATA SOURCE RESOLVERS
# =============================================================================

async def resolve_supabase_query(
    source: Dict,
    params: Dict,
    supabase_client: Any
) -> Any:
    """Resolve a Supabase query data source."""
    query_name = source.get("query_name") or source.get("queryName")
    source_params = source.get("params", {})
    merged_params = {**source_params, **params}
    
    # Query registry - add your named queries here
    query_registry = {
        "topTrendCluster": _query_top_trend_cluster,
        "recentAnalyzedVideos": _query_recent_analyzed_videos,
        "trendingTopics": _query_trending_topics,
    }
    
    query_fn = query_registry.get(query_name)
    if not query_fn:
        logger.warning(f"Unknown Supabase query: {query_name}")
        return {"error": f"Unknown query: {query_name}"}
    
    return await query_fn(supabase_client, merged_params)


async def _query_top_trend_cluster(client: Any, params: Dict) -> Any:
    """Get the top trend cluster from recent analysis."""
    # Placeholder - implement based on your trend_clusters table
    return {
        "name": params.get("topic", "Trending Topic"),
        "score": 85,
        "angles": [
            {
                "angle": "Main angle",
                "script": "This is the generated script for the trend.",
                "hook": "Did you know this surprising fact?"
            }
        ]
    }


async def _query_recent_analyzed_videos(client: Any, params: Dict) -> Any:
    """Get recently analyzed videos."""
    limit = params.get("limit", 10)
    # Placeholder - query video_analysis table
    return {"items": [], "count": 0}


async def _query_trending_topics(client: Any, params: Dict) -> Any:
    """Get trending topics from analysis."""
    # Placeholder
    return {"topics": [], "count": 0}


async def resolve_http_api(source: Dict, params: Dict) -> Any:
    """Resolve an HTTP API data source."""
    import httpx
    
    url = source.get("url", "")
    method = source.get("method", "GET")
    headers = source.get("headers", {})
    body_template = source.get("body_template") or source.get("bodyTemplate")
    
    # Hydrate URL and body with params
    url = render_template(url, {"params": params})
    
    body = None
    if body_template and method == "POST":
        body = hydrate_template_object(body_template, {"params": params})
    
    async with httpx.AsyncClient() as client:
        if method == "GET":
            response = await client.get(url, headers=headers)
        else:
            response = await client.post(url, headers=headers, json=body)
        
        response.raise_for_status()
        return response.json()


def resolve_local_library(source: Dict, libraries: Dict) -> Any:
    """Resolve a local library data source."""
    library_id = source.get("library_id") or source.get("libraryId")
    filter_config = source.get("filter", {})
    
    library = libraries.get(library_id)
    if not library:
        return {"items": [], "warning": f"Library not found: {library_id}"}
    
    items = library.get("items", [])
    limit = filter_config.get("limit", len(items))
    
    return {"items": items[:limit], "meta": {"limit": limit}}


def hydrate_template_object(obj: Any, ctx: Dict) -> Any:
    """Recursively hydrate template strings in an object."""
    if obj is None:
        return obj
    if isinstance(obj, str):
        return render_template(obj, ctx)
    if isinstance(obj, list):
        return [hydrate_template_object(item, ctx) for item in obj]
    if isinstance(obj, dict):
        return {k: hydrate_template_object(v, ctx) for k, v in obj.items()}
    return obj


async def resolve_data_sources(
    sources: List[Dict],
    params: Dict,
    supabase_client: Any = None,
    libraries: Dict = None
) -> Dict[str, Any]:
    """Resolve all data sources into a single resolved_inputs dict."""
    resolved = {}
    libraries = libraries or {}
    
    for source in sources:
        source_id = source.get("id")
        source_type = source.get("type")
        
        try:
            if source_type == "supabase_query":
                resolved[source_id] = await resolve_supabase_query(
                    source, params, supabase_client
                )
            elif source_type == "http_api":
                resolved[source_id] = await resolve_http_api(source, params)
            elif source_type == "local_library":
                resolved[source_id] = resolve_local_library(source, libraries)
            elif source_type == "rss":
                resolved[source_id] = {"url": source.get("url"), "note": "RSS not implemented"}
            else:
                logger.warning(f"Unknown data source type: {source_type}")
                resolved[source_id] = {"error": f"Unknown type: {source_type}"}
        except Exception as e:
            logger.error(f"Error resolving data source {source_id}: {e}")
            resolved[source_id] = {"error": str(e)}
    
    return resolved


# =============================================================================
# BINDING ENGINE
# =============================================================================

def apply_transform(transform: Dict, value: Any, ctx: Dict) -> Any:
    """Apply a transform to a value."""
    transform_type = transform.get("type")
    
    if transform_type == "pick":
        return get_path(value, transform.get("path", ""))
    
    elif transform_type == "map":
        if not isinstance(value, list):
            return []
        map_template = transform.get("map_template") or transform.get("mapTemplate", {})
        return [
            hydrate_template_object(map_template, {**ctx, "item": item})
            for item in value
        ]
    
    elif transform_type == "template":
        template = transform.get("template", "")
        return render_template(template, {**ctx, "value": value})
    
    elif transform_type == "coerce":
        to_type = transform.get("to")
        if to_type == "string":
            return str(value)
        elif to_type == "number":
            return float(value) if value else 0
        elif to_type == "boolean":
            return bool(value)
        elif to_type == "json":
            import json
            return json.loads(value) if isinstance(value, str) else value
    
    elif transform_type == "default":
        return value if value is not None else transform.get("value")
    
    return value


def apply_bindings(
    bindings: List[Dict],
    resolved_inputs: Dict,
    base: Dict,
    runtime_ctx: Dict = None
) -> Dict:
    """Apply bindings to map resolved inputs to render props."""
    ctx = {"inputs": resolved_inputs, **(runtime_ctx or {})}
    
    for binding in bindings:
        target = binding.get("target")
        from_path = binding.get("from") or binding.get("from_path")
        transform = binding.get("transform")
        required = binding.get("required", False)
        
        raw_value = get_path(resolved_inputs, from_path)
        
        if raw_value is None:
            if required:
                raise ValueError(f"Missing required binding: {from_path} -> {target}")
            continue
        
        value = apply_transform(transform, raw_value, ctx) if transform else raw_value
        set_path(base, target, value)
    
    return base


# =============================================================================
# RENDER PROPS CREATION
# =============================================================================

def create_base_render_props(format_def: Dict, run_id: str, variant_id: str = None) -> Dict:
    """Create a base render props object with defaults."""
    defaults = format_def.get("defaults", {})
    params = defaults.get("params", {})
    
    return {
        "topic": "Untitled",
        "script": {"segments": []},
        "audio": {
            "ducking": {
                "enabled": True,
                "music_gain_db": -10,
                "attack_ms": 80,
                "release_ms": 180
            }
        },
        "visuals": {},
        "style": {
            "caption_style": params.get("captionStyle", "bold_pop"),
            "hook_intensity": params.get("hookIntensity", 0.85),
            "pattern_interrupt_sec": params.get("patternInterruptSec", 4),
            "zoom_punch": params.get("zoomPunch", True)
        },
        "meta": {
            "run_id": run_id,
            "format_id": format_def.get("id"),
            "variant_id": variant_id
        }
    }


def apply_default_params(base: Dict, defaults: Dict, overrides: Dict = None) -> Dict:
    """Apply default params and variant overrides to base render props."""
    merged = {**defaults, **(overrides or {})}
    
    if "captionStyle" in merged:
        base["style"]["caption_style"] = merged["captionStyle"]
    if "hookIntensity" in merged:
        base["style"]["hook_intensity"] = merged["hookIntensity"]
    if "patternInterruptSec" in merged:
        base["style"]["pattern_interrupt_sec"] = merged["patternInterruptSec"]
    if "zoomPunch" in merged:
        base["style"]["zoom_punch"] = merged["zoomPunch"]
    
    return base


def infer_duration_from_script(render_props: Dict, fallback_sec: float) -> float:
    """Infer video duration from script segment timestamps."""
    segments = render_props.get("script", {}).get("segments", [])
    if not segments:
        return fallback_sec
    
    # Find max end time
    ends = [
        s.get("t_end_sec") or s.get("tEndSec")
        for s in segments
        if s.get("t_end_sec") or s.get("tEndSec")
    ]
    
    if not ends:
        return fallback_sec
    
    max_end = max(ends)
    # Add small tail for CTA
    return min(max(max_end + 0.5, 1), fallback_sec)


# =============================================================================
# MAIN COMPILER
# =============================================================================

async def compile_run(
    format_def: Dict,
    run_id: str,
    params: Dict,
    supabase_client: Any = None,
    libraries: Dict = None
) -> CompileResult:
    """
    Compile a format run into render props.
    
    This is the main entry point that:
    1. Resolves variant
    2. Resolves data sources
    3. Creates base props
    4. Applies defaults and bindings
    5. Calculates video config
    """
    composition = format_def.get("composition", {})
    defaults = format_def.get("defaults", {})
    
    # Resolve variant
    variant_id = params.get("variantId") or params.get("variant_id")
    variant_sets = composition.get("variant_sets") or composition.get("variantSets", [])
    variant = next((v for v in variant_sets if v.get("id") == variant_id), None)
    
    fps = composition.get("fps", 30)
    width = variant.get("width") if variant else composition.get("width", 1080)
    height = variant.get("height") if variant else composition.get("height", 1920)
    
    max_duration_sec = (
        variant.get("max_duration_sec") or variant.get("maxDurationSec")
        if variant else composition.get("default_duration_sec") or composition.get("defaultDurationSec", 55)
    )
    duration_sec = min(
        params.get("max_duration_sec") or params.get("maxDurationSec") or max_duration_sec,
        max_duration_sec
    )
    
    # Resolve data sources
    data_sources = format_def.get("data_sources") or format_def.get("dataSources", [])
    resolved_inputs = await resolve_data_sources(
        data_sources, params, supabase_client, libraries
    )
    
    # Create base render props
    base = create_base_render_props(format_def, run_id, variant_id)
    
    # Apply defaults and variant overrides
    default_params = defaults.get("params", {})
    variant_overrides = variant.get("overrides") if variant else {}
    apply_default_params(base, default_params, variant_overrides)
    
    # Apply bindings
    bindings = format_def.get("bindings", [])
    render_props = apply_bindings(
        bindings,
        resolved_inputs,
        base,
        {"params": params, "format": format_def, "variant": variant}
    )
    
    # Infer duration from script if available
    inferred_duration = infer_duration_from_script(render_props, duration_sec)
    duration_in_frames = max(1, int(inferred_duration * fps))
    
    return CompileResult(
        resolved_inputs=resolved_inputs,
        render_props=RenderProps(**render_props),
        video_config=VideoConfig(
            fps=fps,
            width=width,
            height=height,
            duration_in_frames=duration_in_frames
        )
    )
