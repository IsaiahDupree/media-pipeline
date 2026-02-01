"""
Text Fitting Utilities for Content Pipeline
Handles character counting by different rules and smart truncation
"""
from typing import Optional, Literal
import unicodedata


CountRule = Literal["graphemes", "utf16", "utf8_bytes"]


def count_utf8_bytes(s: str) -> int:
    """Count UTF-8 byte length of string"""
    return len(s.encode('utf-8'))


def count_utf16_runes(s: str) -> int:
    """
    Count UTF-16 code units (what TikTok uses).
    In Python, this is the len() of the string since Python strings 
    are internally similar to UTF-16 for BMP characters.
    For surrogate pairs (emoji, etc.), we need to count properly.
    """
    return len(s.encode('utf-16-le')) // 2


def count_graphemes(s: str) -> int:
    """
    Count user-perceived characters (grapheme clusters).
    This handles emoji, combining characters, etc. properly.
    """
    # Use unicodedata to segment by grapheme clusters
    # This is a simplified approach - for production, consider using
    # the 'grapheme' package or regex with \X pattern
    count = 0
    i = 0
    while i < len(s):
        # Skip combining characters (they don't form new graphemes)
        count += 1
        i += 1
        # Skip any following combining marks
        while i < len(s) and unicodedata.category(s[i]).startswith('M'):
            i += 1
    return count


def count_by_rule(s: str, rule: CountRule) -> int:
    """Count characters according to the specified counting rule"""
    if rule == "utf8_bytes":
        return count_utf8_bytes(s)
    elif rule == "utf16":
        return count_utf16_runes(s)
    else:  # graphemes
        return count_graphemes(s)


def compute_target(
    max_chars: Optional[int] = None, 
    margin_pct: float = 0.20, 
    soft_cap: Optional[int] = None
) -> Optional[int]:
    """
    Compute target character count (80% under max by default).
    
    Args:
        max_chars: Hard maximum character limit
        margin_pct: Margin to leave under max (0.20 = 20%)
        soft_cap: Soft cap (recommended limit)
    
    Returns:
        Target character count or None if no limits
    """
    if not max_chars and soft_cap:
        return soft_cap
    if not max_chars:
        return None

    target = int(max_chars * (1 - margin_pct))
    return min(target, soft_cap) if soft_cap else target


def truncate_smart(
    text: str, 
    target: int, 
    rule: CountRule,
    preserve_hashtags: bool = True
) -> str:
    """
    Smart truncation that respects word boundaries and optionally preserves hashtags.
    
    Args:
        text: Text to truncate
        target: Target character count
        rule: Counting rule to use
        preserve_hashtags: If True, try to keep hashtags at end
    
    Returns:
        Truncated text fitting within target
    """
    if count_by_rule(text, rule) <= target:
        return text

    # Extract hashtags if preserving them
    hashtags = ""
    main_text = text
    if preserve_hashtags and "#" in text:
        # Find where hashtags start (usually after double newline or at end)
        parts = text.rsplit("\n\n", 1)
        if len(parts) == 2 and parts[1].strip().startswith("#"):
            main_text = parts[0]
            hashtags = "\n\n" + parts[1]
            
            # Check if hashtags alone exceed target
            hashtag_len = count_by_rule(hashtags, rule)
            if hashtag_len >= target:
                hashtags = ""  # Can't preserve hashtags
                main_text = text

    # Iteratively shrink main text
    out = main_text
    
    # Quick shrink to approximate size
    while count_by_rule(out + hashtags, rule) > target and len(out) > 0:
        # Remove roughly 8% each iteration for speed
        out = out[:int(len(out) * 0.92)]

    # Fine-tune: remove character by character
    while count_by_rule(out + hashtags, rule) > target and len(out) > 0:
        out = out[:-1]

    # Snap to last word boundary to avoid ugly cut-offs
    out = out.rstrip()
    if out and not out[-1] in '.!?':
        # Find last space or punctuation
        last_space = max(out.rfind(' '), out.rfind('\n'))
        if last_space > len(out) * 0.7:  # Only if we don't lose too much
            out = out[:last_space].rstrip()
    
    # Add ellipsis if we truncated significantly
    if len(out) < len(main_text) * 0.95:
        if count_by_rule(out + "..." + hashtags, rule) <= target:
            out = out.rstrip('.,!? ') + "..."

    result = out + hashtags
    
    # Fallback if we couldn't fit anything
    if not result.strip():
        return text[:max(1, target // 2)]
    
    return result.strip()


def split_into_segments(text: str, max_segment_chars: int, rule: CountRule) -> list[str]:
    """
    Split text into segments that fit within character limit.
    Useful for platforms with strict limits (like X/Twitter threads).
    """
    if count_by_rule(text, rule) <= max_segment_chars:
        return [text]
    
    segments = []
    sentences = text.replace('\n', ' \n ').split('. ')
    
    current_segment = ""
    for sentence in sentences:
        test_segment = current_segment + (". " if current_segment else "") + sentence
        
        if count_by_rule(test_segment, rule) <= max_segment_chars:
            current_segment = test_segment
        else:
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = sentence
    
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments


def extract_hashtags(text: str) -> tuple[str, list[str]]:
    """
    Extract hashtags from text.
    
    Returns:
        Tuple of (text_without_hashtags, list_of_hashtags)
    """
    import re
    
    # Find all hashtags
    hashtag_pattern = r'#\w+'
    hashtags = re.findall(hashtag_pattern, text)
    
    # Remove hashtags from text
    clean_text = re.sub(hashtag_pattern, '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    return clean_text, hashtags


def format_hashtags(hashtags: list[str], max_count: Optional[int] = None) -> str:
    """
    Format hashtags for inclusion in post.
    
    Args:
        hashtags: List of hashtags (with or without #)
        max_count: Maximum number of hashtags to include
    
    Returns:
        Formatted hashtag string
    """
    # Ensure hashtags start with #
    formatted = [h if h.startswith('#') else f'#{h}' for h in hashtags]
    
    # Limit count
    if max_count:
        formatted = formatted[:max_count]
    
    return ' '.join(formatted)


def validate_text_fits(
    text: str,
    max_chars: Optional[int],
    rule: CountRule
) -> tuple[bool, int]:
    """
    Check if text fits within character limit.
    
    Returns:
        Tuple of (fits, character_count)
    """
    char_count = count_by_rule(text, rule)
    fits = max_chars is None or char_count <= max_chars
    return fits, char_count
