"""
Audio Mixer

FFmpeg-based audio mixing for combining voiceover, music, and SFX
into a single audio bus for Motion Canvas rendering.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Optional
from loguru import logger

from .types import SfxManifest
from .cue_sheet import CueSheet, resolve_cue_paths


async def run_ffmpeg_async(args: list[str]) -> tuple[int, str, str]:
    """
    Run FFmpeg asynchronously.
    
    Args:
        args: FFmpeg arguments
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = ["ffmpeg", "-y"] + args
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    stdout, stderr = await process.communicate()
    
    return (
        process.returncode or 0,
        stdout.decode() if stdout else "",
        stderr.decode() if stderr else "",
    )


def run_ffmpeg_sync(args: list[str]) -> tuple[int, str, str]:
    """
    Run FFmpeg synchronously.
    
    Args:
        args: FFmpeg arguments
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = ["ffmpeg", "-y"] + args
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    return result.returncode, result.stdout, result.stderr


async def mix_audio_bus(
    cue_sheet: CueSheet,
    manifest: SfxManifest,
    sfx_root_dir: str,
    output_path: str,
    output_format: str = "wav",
    normalize: bool = False,
) -> str:
    """
    Mix base audio with SFX cues into a single audio bus.
    
    Args:
        cue_sheet: Cue sheet with base audio and SFX cues
        manifest: SFX manifest for file resolution
        sfx_root_dir: Root directory for SFX files
        output_path: Path to output audio file
        output_format: Output format (wav, mp3, etc.)
        normalize: Whether to normalize output
        
    Returns:
        Path to mixed audio file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Resolve cue paths
    resolved_cues = resolve_cue_paths(cue_sheet, manifest, sfx_root_dir)
    
    if not resolved_cues:
        # No SFX, just copy base audio
        args = [
            "-i", cue_sheet.base_audio,
            "-c:a", "pcm_s16le" if output_format == "wav" else "libmp3lame",
            output_path,
        ]
        code, stdout, stderr = await run_ffmpeg_async(args)
        if code != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr[:200]}")
        return output_path
    
    # Build FFmpeg filter complex
    inputs = ["-i", cue_sheet.base_audio]
    filters = []
    mix_inputs = ["[0:a]"]
    
    for idx, cue in enumerate(resolved_cues):
        inputs.extend(["-i", cue["path"]])
        
        delay_ms = max(0, int(cue["startSec"] * 1000))
        gain_db = cue["gainDb"]
        label = f"s{idx}"
        input_idx = idx + 1
        
        # adelay + volume adjustment
        filters.append(
            f"[{input_idx}:a]adelay={delay_ms}|{delay_ms},volume={gain_db}dB[{label}]"
        )
        mix_inputs.append(f"[{label}]")
    
    # Mix all inputs
    norm_flag = ":normalize=1" if normalize else ":normalize=0"
    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}{norm_flag}[aout]"
    )
    
    filter_complex = ";".join(filters)
    
    # Determine codec
    if output_format == "wav":
        codec_args = ["-c:a", "pcm_s16le"]
    elif output_format == "mp3":
        codec_args = ["-c:a", "libmp3lame", "-q:a", "2"]
    else:
        codec_args = ["-c:a", "aac", "-b:a", "192k"]
    
    args = inputs + [
        "-filter_complex", filter_complex,
        "-map", "[aout]",
    ] + codec_args + [output_path]
    
    logger.info(f"Mixing audio bus with {len(resolved_cues)} SFX cues")
    code, stdout, stderr = await run_ffmpeg_async(args)
    
    if code != 0:
        logger.error(f"FFmpeg mix failed: {stderr}")
        raise RuntimeError(f"Audio mix failed: {stderr[:200]}")
    
    logger.info(f"Mixed audio bus saved to: {output_path}")
    return output_path


def mix_audio_bus_sync(
    cue_sheet: CueSheet,
    manifest: SfxManifest,
    sfx_root_dir: str,
    output_path: str,
    output_format: str = "wav",
    normalize: bool = False,
) -> str:
    """Synchronous version of mix_audio_bus."""
    return asyncio.run(mix_audio_bus(
        cue_sheet=cue_sheet,
        manifest=manifest,
        sfx_root_dir=sfx_root_dir,
        output_path=output_path,
        output_format=output_format,
        normalize=normalize,
    ))


async def mix_tracks(
    tracks: list[dict],
    output_path: str,
    output_format: str = "wav",
) -> str:
    """
    Mix multiple audio tracks with individual settings.
    
    Args:
        tracks: List of dicts with 'path', 'startSec', 'volume' (0-1)
        output_path: Output file path
        output_format: Output format
        
    Returns:
        Path to mixed audio
    """
    if not tracks:
        raise ValueError("No tracks to mix")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    inputs = []
    filters = []
    mix_inputs = []
    
    for idx, track in enumerate(tracks):
        inputs.extend(["-i", track["path"]])
        
        delay_ms = max(0, int(track.get("startSec", 0) * 1000))
        volume = track.get("volume", 1.0)
        label = f"t{idx}"
        
        filters.append(
            f"[{idx}:a]adelay={delay_ms}|{delay_ms},volume={volume}[{label}]"
        )
        mix_inputs.append(f"[{label}]")
    
    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0[aout]"
    )
    
    filter_complex = ";".join(filters)
    
    if output_format == "wav":
        codec_args = ["-c:a", "pcm_s16le"]
    else:
        codec_args = ["-c:a", "libmp3lame", "-q:a", "2"]
    
    args = inputs + [
        "-filter_complex", filter_complex,
        "-map", "[aout]",
    ] + codec_args + [output_path]
    
    code, stdout, stderr = await run_ffmpeg_async(args)
    
    if code != 0:
        raise RuntimeError(f"Track mix failed: {stderr[:200]}")
    
    return output_path


async def get_audio_duration(path: str) -> float:
    """
    Get audio file duration in seconds.
    
    Args:
        path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    stdout, stderr = await process.communicate()
    
    try:
        return float(stdout.decode().strip())
    except ValueError:
        raise RuntimeError(f"Could not get duration: {path}")


async def normalize_audio(
    input_path: str,
    output_path: str,
    target_loudness: float = -14.0,
) -> str:
    """
    Normalize audio to target loudness (LUFS).
    
    Args:
        input_path: Input audio path
        output_path: Output audio path
        target_loudness: Target loudness in LUFS
        
    Returns:
        Path to normalized audio
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    args = [
        "-i", input_path,
        "-af", f"loudnorm=I={target_loudness}:TP=-1.5:LRA=11",
        "-ar", "48000",
        output_path,
    ]
    
    code, stdout, stderr = await run_ffmpeg_async(args)
    
    if code != 0:
        raise RuntimeError(f"Normalize failed: {stderr[:200]}")
    
    return output_path


async def trim_audio(
    input_path: str,
    output_path: str,
    start_sec: float = 0,
    duration_sec: Optional[float] = None,
) -> str:
    """
    Trim audio file.
    
    Args:
        input_path: Input path
        output_path: Output path
        start_sec: Start time in seconds
        duration_sec: Duration (None for rest of file)
        
    Returns:
        Path to trimmed audio
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    args = [
        "-i", input_path,
        "-ss", str(start_sec),
    ]
    
    if duration_sec is not None:
        args.extend(["-t", str(duration_sec)])
    
    args.extend(["-c", "copy", output_path])
    
    code, stdout, stderr = await run_ffmpeg_async(args)
    
    if code != 0:
        raise RuntimeError(f"Trim failed: {stderr[:200]}")
    
    return output_path
