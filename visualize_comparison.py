"""
Single Sample Visualization Script

Compares byte-wise losses between two models and generates image visualizations
with color-coded text showing the difference (delta) from a baseline model.

Usage:
    python visualize_comparison.py --model_a path/to/model_a_results.json \
                                   --model_b path/to/model_b_results.json \
                                   --output_dir ./visualizations \
                                   [--sample_index 0]

Color scheme (based on deviation from average delta):
    Green = Better than average (delta < avg_delta)
    White = Equal to average (delta == avg_delta)
    Red = Worse than average (delta > avg_delta)
"""

import argparse
import json
import os
import re
from typing import List, Tuple, Optional, Set
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from helpers import TokenizerBytesConverter


# Global tokenizers (lazy loaded)
_qwen_tokenizer = None
_rwkv_tokenizer = None


def get_qwen_tokenizer():
    """Lazy load Qwen tokenizer."""
    global _qwen_tokenizer
    if _qwen_tokenizer is None:
        _qwen_tokenizer = TokenizerBytesConverter("Qwen/Qwen3-0.6B-Base")
    return _qwen_tokenizer


def get_rwkv_tokenizer():
    """Lazy load RWKV tokenizer."""
    global _rwkv_tokenizer
    if _rwkv_tokenizer is None:
        from rwkv.utils import PIPELINE
        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
        _rwkv_tokenizer = TRIE_TOKENIZER("rwkv_vocab_v20230424.txt")
    return _rwkv_tokenizer


def get_tokenizer_boundaries(text: str, tokenizer, is_rwkv: bool = False) -> Set[int]:
    """
    Get token boundaries (byte positions) for a given text using the specified tokenizer.

    Args:
        text: The input text
        tokenizer: Either TokenizerBytesConverter (for Qwen) or RWKV tokenizer
        is_rwkv: Whether this is an RWKV tokenizer

    Returns:
        Set of byte positions where token boundaries occur
    """
    boundaries = set()
    boundaries.add(0)  # Start is always a boundary

    if is_rwkv:
        # RWKV tokenizer
        tokenized = tokenizer.encode(text)
        if hasattr(tokenized, "ids"):
            token_ids = tokenized.ids
        else:
            token_ids = tokenized

        byte_pos = 0
        for token_id in token_ids:
            token_bytes = tokenizer.decodeBytes([token_id])
            byte_pos += len(token_bytes)
            boundaries.add(byte_pos)
    else:
        # Qwen tokenizer (TokenizerBytesConverter)
        token_bytes_list = tokenizer.encode_to_bytes(text)
        byte_pos = 0
        for token_bytes in token_bytes_list:
            byte_pos += len(token_bytes)
            boundaries.add(byte_pos)

    return boundaries


def get_token_info_for_text(text: str) -> dict:
    """
    Get detailed token information for each byte position.

    Returns:
        dict with:
            - 'common_boundaries': sorted list of common boundaries
            - 'qwen_tokens': list of (start, end, token_str) for Qwen
            - 'rwkv_tokens': list of (start, end, token_str) for RWKV
            - 'byte_to_qwen': mapping from byte_start to token index
            - 'byte_to_rwkv': mapping from byte_start to token index
    """
    qwen_tokenizer = get_qwen_tokenizer()
    rwkv_tokenizer = get_rwkv_tokenizer()

    text_bytes = text.encode("utf-8")

    # Get Qwen tokens with positions
    qwen_tokens = []
    byte_to_qwen = {}
    qwen_bytes_list = qwen_tokenizer.encode_to_bytes(text)
    byte_pos = 0
    for idx, token_bytes in enumerate(qwen_bytes_list):
        start = byte_pos
        end = byte_pos + len(token_bytes)
        try:
            token_str = bytes(token_bytes).decode("utf-8")
        except UnicodeDecodeError:
            token_str = repr(bytes(token_bytes))
        qwen_tokens.append((start, end, token_str))
        byte_to_qwen[start] = idx
        byte_pos = end

    # Get RWKV tokens with positions
    rwkv_tokens = []
    byte_to_rwkv = {}
    tokenized = rwkv_tokenizer.encode(text)
    if hasattr(tokenized, "ids"):
        token_ids = tokenized.ids
    else:
        token_ids = tokenized

    byte_pos = 0
    for idx, token_id in enumerate(token_ids):
        token_bytes = rwkv_tokenizer.decodeBytes([token_id])
        start = byte_pos
        end = byte_pos + len(token_bytes)
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = repr(token_bytes)
        rwkv_tokens.append((start, end, token_str))
        byte_to_rwkv[start] = idx
        byte_pos = end

    # Get common boundaries
    qwen_boundaries = set([0] + [t[1] for t in qwen_tokens])
    rwkv_boundaries = set([0] + [t[1] for t in rwkv_tokens])
    common_boundaries = sorted(qwen_boundaries & rwkv_boundaries)

    return {
        'common_boundaries': common_boundaries,
        'qwen_tokens': qwen_tokens,
        'rwkv_tokens': rwkv_tokens,
        'byte_to_qwen': byte_to_qwen,
        'byte_to_rwkv': byte_to_rwkv,
    }


def get_common_boundaries(text: str) -> List[int]:
    """
    Get common token boundaries from both Qwen and RWKV tokenizers.

    Args:
        text: The input text

    Returns:
        Sorted list of byte positions where both tokenizers have boundaries
    """
    qwen_tokenizer = get_qwen_tokenizer()
    rwkv_tokenizer = get_rwkv_tokenizer()

    qwen_boundaries = get_tokenizer_boundaries(text, qwen_tokenizer, is_rwkv=False)
    rwkv_boundaries = get_tokenizer_boundaries(text, rwkv_tokenizer, is_rwkv=True)

    # Find common boundaries (intersection)
    common_boundaries = qwen_boundaries & rwkv_boundaries

    return sorted(common_boundaries)


def load_results(file_path: str) -> dict:
    """Load evaluation results from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_model_name(model_path: str) -> str:
    """Extract clean model name from path, removing directory and extension."""
    # Get the last part of the path
    name = os.path.basename(model_path)
    # Remove common extensions
    for ext in [".pth", ".bin", ".safetensors", ".ckpt", ".pt"]:
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    return name


def delta_to_color(delta: float, avg_delta: float, max_deviation: float) -> Tuple[int, int, int]:
    """
    Map a delta value to an RGB color based on deviation from average.

    Args:
        delta: The difference in loss (model_a - model_b)
        avg_delta: The average delta across all bytes
        max_deviation: The maximum absolute deviation from average

    Returns:
        Tuple of (R, G, B) values

    Color scheme:
        - delta == avg_delta -> White (neutral)
        - delta < avg_delta -> Green (better than average)
        - delta > avg_delta -> Red (worse than average)
    """
    if max_deviation == 0:
        return (255, 255, 255)

    # Calculate deviation from average
    deviation = delta - avg_delta

    # Normalize deviation to [-1, 1] range
    normalized = max(-1, min(1, deviation / max_deviation))

    if normalized < 0:
        # Green: better than average
        intensity = -normalized
        r = int(255 * (1 - intensity * 0.7))
        g = 255
        b = int(255 * (1 - intensity * 0.7))
    else:
        # Red: worse than average
        intensity = normalized
        r = 255
        g = int(255 * (1 - intensity * 0.7))
        b = int(255 * (1 - intensity * 0.7))

    return (r, g, b)


def get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    """Get a monospace font, with fallbacks for different systems."""
    font_candidates = [
        "consola.ttf",  # Windows Consolas
        "cour.ttf",  # Windows Courier New
        "DejaVuSansMono.ttf",  # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",  # macOS
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue

    # Fallback to default font
    return ImageFont.load_default()


def calculate_text_layout(
    text: str,
    byte_losses_a: List[float],
    byte_losses_b: List[float],
    font: ImageFont.FreeTypeFont,
    max_width: int = 1200,
    line_height: int = 20,
    char_width: int = 9,
) -> Tuple[List[List[dict]], float, float, int, int]:
    """
    Calculate the layout of text with colors.

    Returns:
        Tuple of (lines, avg_delta, max_deviation, image_width, image_height)
        Each line is a list of dicts with 'char', 'color', 'x', 'y'
    """
    text_bytes = text.encode("utf-8")

    assert len(text_bytes) == len(byte_losses_a), \
        f"Text bytes ({len(text_bytes)}) != model A losses ({len(byte_losses_a)})"
    assert len(text_bytes) == len(byte_losses_b), \
        f"Text bytes ({len(text_bytes)}) != model B losses ({len(byte_losses_b)})"

    # Calculate deltas
    deltas = [a - b for a, b in zip(byte_losses_a, byte_losses_b)]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0

    # Calculate max deviation from average (for color normalization)
    # Use 95th percentile instead of max to avoid extreme outliers dominating the color range
    deviations = [d - avg_delta for d in deltas]
    abs_deviations = [abs(dev) for dev in deviations]
    max_deviation = float(np.percentile(abs_deviations, 100)) if abs_deviations else 0
    max_deviation = max(max_deviation, 1e-6)  # Avoid division by zero

    # Get common boundaries from both tokenizers
    token_info = get_token_info_for_text(text)
    common_boundaries = token_info['common_boundaries']

    # Build a mapping from byte position to token color
    byte_to_color = {}
    for i in range(len(common_boundaries) - 1):
        start_byte = common_boundaries[i]
        end_byte = common_boundaries[i + 1]

        # Calculate average delta for this token
        token_deltas = deltas[start_byte:end_byte]
        avg_token_delta = sum(token_deltas) / len(token_deltas) if token_deltas else 0
        color = delta_to_color(avg_token_delta, avg_delta, max_deviation)

        # Assign this color to all bytes in the token
        for b in range(start_byte, end_byte):
            byte_to_color[b] = color

    # Build character info with colors (using token-based colors)
    chars_with_colors = []
    byte_index = 0

    for char in text:
        char_bytes = char.encode("utf-8")
        num_bytes = len(char_bytes)

        # Use the color from the first byte of this character
        color = byte_to_color.get(byte_index, (255, 255, 255))
        chars_with_colors.append({"char": char, "color": color})

        byte_index += num_bytes

    # Layout into lines
    lines = []
    current_line = []
    x = 0
    y = 0
    max_x = 0
    chars_per_line = (max_width - 40) // char_width  # Leave margin

    for item in chars_with_colors:
        char = item["char"]
        color = item["color"]

        if char == "\n" or len(current_line) >= chars_per_line:
            if current_line:
                lines.append(current_line)
            current_line = []
            x = 0
            y += line_height
            if char == "\n":
                continue

        # Handle tabs
        if char == "\t":
            char = "    "  # Replace tab with 4 spaces

        for c in (char if char == "    " else [char]):
            current_line.append({
                "char": c if c != "    " else " ",
                "color": color,
                "x": x,
                "y": y,
            })
            x += char_width
            max_x = max(max_x, x)

    if current_line:
        lines.append(current_line)

    image_width = max(max_x + 40, 400)
    image_height = y + line_height + 20

    return lines, avg_delta, max_deviation, image_width, image_height


def generate_image(
    sample_index: int,
    text: str,
    byte_losses_a: List[float],
    byte_losses_b: List[float],
    model_a_name: str,
    model_b_name: str,
    output_path: str,
    font_size: int = 14,
    max_width: int = 1200,
):
    """Generate a PNG image for visualization."""

    font = get_font(font_size)
    header_font = get_font(font_size + 4)
    small_font = get_font(font_size - 2)

    # Calculate character width (approximate for monospace)
    char_width = font_size * 0.6
    line_height = font_size + 6

    # Calculate layout
    lines, avg_delta, max_delta, content_width, content_height = calculate_text_layout(
        text, byte_losses_a, byte_losses_b, font,
        max_width=max_width,
        line_height=int(line_height),
        char_width=int(char_width),
    )

    # Calculate average losses
    avg_loss_a = sum(byte_losses_a) / len(byte_losses_a) if byte_losses_a else 0
    avg_loss_b = sum(byte_losses_b) / len(byte_losses_b) if byte_losses_b else 0

    # Header height
    header_height = 120

    # Create image
    img_width = max(content_width, 800)
    img_height = header_height + content_height + 40
    img = Image.new("RGB", (img_width, img_height), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Draw header background
    draw.rectangle([(0, 0), (img_width, header_height)], fill=(51, 51, 51))

    # Draw title
    title = f"Sample {sample_index} - Byte-wise Loss Comparison"
    draw.text((20, 15), title, fill=(255, 255, 255), font=header_font)

    # Draw metadata
    y_offset = 45
    line_spacing = 18

    # Model names
    draw.text((20, y_offset), f"Model A: {model_a_name[:60]}", fill=(200, 200, 200), font=small_font)
    draw.text((20, y_offset + line_spacing), f"Baseline (B): {model_b_name[:60]}", fill=(200, 200, 200), font=small_font)

    # Stats
    stats_x = 450
    draw.text((stats_x, y_offset), f"Avg Loss A: {avg_loss_a:.4f}", fill=(200, 200, 200), font=small_font)
    draw.text((stats_x, y_offset + line_spacing), f"Avg Loss B: {avg_loss_b:.4f}", fill=(200, 200, 200), font=small_font)

    delta_color = (100, 255, 100) if avg_delta < 0 else (255, 100, 100)
    draw.text((stats_x + 200, y_offset), f"Avg Delta: {avg_delta:+.4f}", fill=delta_color, font=small_font)

    # Draw legend
    legend_y = y_offset + line_spacing * 2 + 5
    legend_x = 20

    # Green box
    draw.rectangle([(legend_x, legend_y), (legend_x + 20, legend_y + 12)], fill=(77, 255, 77))
    draw.text((legend_x + 25, legend_y - 2), "Better than avg", fill=(200, 200, 200), font=small_font)

    # White box
    draw.rectangle([(legend_x + 130, legend_y), (legend_x + 150, legend_y + 12)], fill=(255, 255, 255))
    draw.text((legend_x + 155, legend_y - 2), "= Avg delta", fill=(200, 200, 200), font=small_font)

    # Red box
    draw.rectangle([(legend_x + 260, legend_y), (legend_x + 280, legend_y + 12)], fill=(255, 77, 77))
    draw.text((legend_x + 285, legend_y - 2), "Worse than avg", fill=(200, 200, 200), font=small_font)

    # Draw content background
    content_y_start = header_height + 10
    draw.rectangle(
        [(10, content_y_start), (img_width - 10, img_height - 10)],
        fill=(255, 255, 255),
        outline=(200, 200, 200),
    )

    # Draw colored text
    text_x_offset = 20
    text_y_offset = content_y_start + 10

    for line in lines:
        for char_info in line:
            x = text_x_offset + char_info["x"]
            y = text_y_offset + char_info["y"]
            color = char_info["color"]
            char = char_info["char"]

            # Draw background rectangle
            draw.rectangle(
                [(x, y), (x + int(char_width), y + int(line_height) - 2)],
                fill=color,
            )

            # Draw character
            draw.text((x + 1, y), char, fill=(0, 0, 0), font=font)

    # Save image
    img.save(output_path, "PNG")


def generate_html(
    sample_index: int,
    text: str,
    byte_losses_a: List[float],
    byte_losses_b: List[float],
    model_a_name: str,
    model_b_name: str,
    output_path: str,
):
    """Generate an HTML file for visualization with word linking on hover."""

    # Calculate deltas
    deltas = [a - b for a, b in zip(byte_losses_a, byte_losses_b)]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0

    # Calculate max deviation (95th percentile)
    deviations = [d - avg_delta for d in deltas]
    abs_deviations = [abs(dev) for dev in deviations]
    max_deviation = float(np.percentile(abs_deviations, 100)) if abs_deviations else 0
    max_deviation = max(max_deviation, 1e-6)

    # Calculate average losses
    avg_loss_a = sum(byte_losses_a) / len(byte_losses_a) if byte_losses_a else 0
    avg_loss_b = sum(byte_losses_b) / len(byte_losses_b) if byte_losses_b else 0

    # Get token info from both tokenizers
    text_bytes = text.encode("utf-8")
    token_info = get_token_info_for_text(text)
    common_boundaries = token_info['common_boundaries']
    qwen_tokens = token_info['qwen_tokens']
    rwkv_tokens = token_info['rwkv_tokens']

    # Build byte position to token mapping for both tokenizers
    def get_tokens_for_range(byte_start, byte_end, token_list):
        """Find which tokens overlap with the given byte range."""
        result = []
        for idx, (t_start, t_end, t_str) in enumerate(token_list):
            if t_start < byte_end and t_end > byte_start:
                result.append((idx, t_str))
        return result

    # Build tokens based on common boundaries
    tokens = []
    for i in range(len(common_boundaries) - 1):
        start_byte = common_boundaries[i]
        end_byte = common_boundaries[i + 1]
        token_bytes = text_bytes[start_byte:end_byte]
        try:
            token_text = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # If we can't decode, skip this segment
            continue

        # Get corresponding tokens from both tokenizers
        qwen_toks = get_tokens_for_range(start_byte, end_byte, qwen_tokens)
        rwkv_toks = get_tokens_for_range(start_byte, end_byte, rwkv_tokens)

        # Determine if this token is a "word" (alphanumeric) or non-word
        # A word contains at least one alphanumeric character
        if re.search(r'\w', token_text, re.UNICODE):
            tokens.append({
                'type': 'word',
                'text': token_text,
                'byte_start': start_byte,
                'byte_end': end_byte,
                'word_lower': token_text.lower(),
                'qwen_tokens': qwen_toks,
                'rwkv_tokens': rwkv_toks,
            })
        else:
            tokens.append({
                'type': 'non-word',
                'text': token_text,
                'byte_start': start_byte,
                'byte_end': end_byte,
                'qwen_tokens': qwen_toks,
                'rwkv_tokens': rwkv_toks,
            })

    # Track word occurrences for linking
    word_occurrences = {}  # word_lower -> list of token indices
    word_id_counter = 0

    for i, token in enumerate(tokens):
        if token['type'] == 'word':
            word_lower = token['word_lower']
            if word_lower not in word_occurrences:
                word_occurrences[word_lower] = []
            word_occurrences[word_lower].append(i)
            token['word_id'] = word_id_counter
            word_id_counter += 1

    # Build HTML content
    html_content = []

    def escape_for_attr(s):
        """Escape string for use in HTML attribute."""
        return s.replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')

    for token in tokens:
        token_text = token['text']
        byte_start = token['byte_start']
        byte_end = token['byte_end']

        # Build token info strings for tooltip
        qwen_info = ', '.join([f'[{idx}] {repr(s)}' for idx, s in token['qwen_tokens']])
        rwkv_info = ', '.join([f'[{idx}] {repr(s)}' for idx, s in token['rwkv_tokens']])

        # Get raw bytes and per-byte losses for this token
        raw_bytes = list(text_bytes[byte_start:byte_end])
        losses_a = byte_losses_a[byte_start:byte_end]
        losses_b = byte_losses_b[byte_start:byte_end]

        # Format byte-wise data for tooltip
        bytes_str = ' '.join([f'{b:02x}' for b in raw_bytes])
        losses_a_str = ' '.join([f'{l:.2f}' for l in losses_a])
        losses_b_str = ' '.join([f'{l:.2f}' for l in losses_b])

        # Calculate average delta for the entire token (not per character)
        token_deltas = deltas[byte_start:byte_end]
        avg_token_delta = sum(token_deltas) / len(token_deltas) if token_deltas else 0

        # Get single color for the entire token
        color = delta_to_color(avg_token_delta, avg_delta, max_deviation)
        r, g, b = color

        # Build HTML for each character with the SAME color
        token_html_parts = []
        for char in token_text:
            # Escape HTML special characters
            if char == '<':
                escaped_char = '&lt;'
            elif char == '>':
                escaped_char = '&gt;'
            elif char == '&':
                escaped_char = '&amp;'
            elif char == '\n':
                escaped_char = '<br>'
            elif char == ' ':
                escaped_char = '&nbsp;'
            elif char == '\t':
                escaped_char = '&nbsp;&nbsp;&nbsp;&nbsp;'
            else:
                escaped_char = char

            token_html_parts.append(escaped_char)

        # Wrap in token-span with tokenizer info for hover
        token_span_content = ''.join(token_html_parts)
        data_attrs = (
            f'data-qwen="{escape_for_attr(qwen_info)}" '
            f'data-rwkv="{escape_for_attr(rwkv_info)}" '
            f'data-bytes="{escape_for_attr(bytes_str)}" '
            f'data-loss-a="{escape_for_attr(losses_a_str)}" '
            f'data-loss-b="{escape_for_attr(losses_b_str)}" '
            f'data-delta="{avg_token_delta:.6f}"'
        )
        style_attr = f'style="background-color: rgb({r},{g},{b})"'

        # Wrap words in a word-span for hover interaction
        if token['type'] == 'word':
            word_lower = token['word_lower']
            occurrences = word_occurrences[word_lower]
            # Only add linking if word appears more than once
            if len(occurrences) > 1:
                word_id = token['word_id']
                html_content.append(
                    f'<span class="token word" {data_attrs} {style_attr} data-word="{word_lower}" data-word-id="{word_id}">'
                    + token_span_content
                    + '</span>'
                )
            else:
                html_content.append(f'<span class="token" {data_attrs} {style_attr}>{token_span_content}</span>')
        else:
            html_content.append(f'<span class="token" {data_attrs} {style_attr}>{token_span_content}</span>')

    # Determine delta color for header
    delta_color = "#64ff64" if avg_delta < 0 else "#ff6464"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Sample {sample_index} - Byte-wise Loss Comparison</title>
    <style>
        body {{
            font-family: Consolas, 'Courier New', monospace;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .header h1 {{
            margin: 0 0 15px 0;
            font-size: 18px;
        }}
        .meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            font-size: 12px;
            color: #c8c8c8;
        }}
        .legend {{
            display: flex;
            gap: 15px;
            margin-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-box {{
            width: 20px;
            height: 12px;
            border: 1px solid #666;
        }}
        .content {{
            background-color: white;
            margin: 10px;
            padding: 15px;
            border: 1px solid #ccc;
            font-size: 14px;
            line-height: 1.8;
            word-wrap: break-word;
            position: relative;
        }}
        .content span {{
            padding: 1px 0;
        }}
        .word {{
            cursor: pointer;
            position: relative;
        }}
        .word:hover {{
            outline: 2px solid #007bff;
            outline-offset: 1px;
        }}
        .word.highlighted {{
            outline: 2px solid #ff6b6b;
            outline-offset: 1px;
        }}
        #svg-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1000;
        }}
        .link-line {{
            stroke: #007bff;
            stroke-width: 2;
            fill: none;
            opacity: 0.7;
        }}
        .link-dot {{
            fill: #007bff;
            opacity: 0.8;
        }}
        .token {{
            position: relative;
            cursor: help;
        }}
        .token:hover {{
            outline: 1px dashed #666;
        }}
        #tooltip {{
            position: fixed;
            background-color: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px 14px;
            border-radius: 6px;
            font-size: 12px;
            max-width: 400px;
            z-index: 2000;
            pointer-events: none;
            display: none;
            line-height: 1.6;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        #tooltip .label {{
            color: #aaa;
            font-weight: bold;
        }}
        #tooltip .bytes {{
            color: #a5f3fc;
            font-family: monospace;
        }}
        #tooltip .loss-a {{
            color: #86efac;
            font-family: monospace;
        }}
        #tooltip .loss-b {{
            color: #fca5a5;
            font-family: monospace;
        }}
        #tooltip .qwen {{
            color: #7dd3fc;
        }}
        #tooltip .rwkv {{
            color: #fcd34d;
        }}
    </style>
</head>
<body>
    <svg id="svg-overlay"></svg>
    <div id="tooltip"></div>
    <div class="header">
        <h1>Sample {sample_index} - Byte-wise Loss Comparison</h1>
        <div class="meta">
            <div>Model A: {model_a_name[:60]}</div>
            <div>Baseline (B): {model_b_name[:60]}</div>
            <div>Avg Loss A: {avg_loss_a:.4f}</div>
            <div>Avg Loss B: {avg_loss_b:.4f}</div>
            <div style="color: {delta_color}">Avg Delta: {avg_delta:+.4f}</div>
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-box" style="background-color: rgb(77, 255, 77)"></div>
                <span>Better than avg</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background-color: rgb(255, 255, 255)"></div>
                <span>= Avg delta</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background-color: rgb(255, 77, 77)"></div>
                <span>Worse than avg</span>
            </div>
            <div class="legend-item" style="margin-left: 20px;">
                <span style="color: #aaa;">Saturation:</span>
                <input type="range" id="saturation-slider" min="500" max="1000" value="1000" step="1" style="width: 200px; vertical-align: middle;">
                <span id="saturation-value" style="color: #fff; min-width: 45px; display: inline-block;">100.0%</span>
            </div>
        </div>
    </div>
    <div class="content">
        {''.join(html_content)}
    </div>
    <script>
        const svgOverlay = document.getElementById('svg-overlay');
        const words = document.querySelectorAll('.word');

        // Group words by their data-word attribute
        const wordGroups = {{}};
        words.forEach(word => {{
            const wordText = word.getAttribute('data-word');
            if (!wordGroups[wordText]) {{
                wordGroups[wordText] = [];
            }}
            wordGroups[wordText].push(word);
        }});

        function clearLines() {{
            svgOverlay.innerHTML = '';
            words.forEach(w => w.classList.remove('highlighted'));
        }}

        function drawLines(hoveredWord) {{
            clearLines();

            const wordText = hoveredWord.getAttribute('data-word');
            const wordId = parseInt(hoveredWord.getAttribute('data-word-id'));
            const sameWords = wordGroups[wordText] || [];

            // Find previous occurrences
            const previousWords = sameWords.filter(w => {{
                const id = parseInt(w.getAttribute('data-word-id'));
                return id < wordId;
            }});

            if (previousWords.length === 0) return;

            // Highlight all same words
            sameWords.forEach(w => w.classList.add('highlighted'));

            // Get position of hovered word
            const hoveredRect = hoveredWord.getBoundingClientRect();
            const hoveredX = hoveredRect.left + hoveredRect.width / 2;
            const hoveredY = hoveredRect.top + hoveredRect.height / 2;

            // Draw lines to previous occurrences
            previousWords.forEach(prevWord => {{
                const prevRect = prevWord.getBoundingClientRect();
                const prevX = prevRect.left + prevRect.width / 2;
                const prevY = prevRect.top + prevRect.height / 2;

                // Create curved line (quadratic bezier)
                const midX = (hoveredX + prevX) / 2;
                const midY = Math.min(hoveredY, prevY) - 30;

                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('class', 'link-line');
                path.setAttribute('d', `M ${{prevX}} ${{prevY}} Q ${{midX}} ${{midY}} ${{hoveredX}} ${{hoveredY}}`);
                svgOverlay.appendChild(path);

                // Add dots at endpoints
                const dot1 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                dot1.setAttribute('class', 'link-dot');
                dot1.setAttribute('cx', prevX);
                dot1.setAttribute('cy', prevY);
                dot1.setAttribute('r', 4);
                svgOverlay.appendChild(dot1);

                const dot2 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                dot2.setAttribute('class', 'link-dot');
                dot2.setAttribute('cx', hoveredX);
                dot2.setAttribute('cy', hoveredY);
                dot2.setAttribute('r', 4);
                svgOverlay.appendChild(dot2);
            }});
        }}

        // Add event listeners
        words.forEach(word => {{
            word.addEventListener('mouseenter', () => drawLines(word));
            word.addEventListener('mouseleave', clearLines);
        }});

        // Clear lines on scroll (positions change)
        window.addEventListener('scroll', clearLines);

        // Tooltip functionality for token info
        const tooltip = document.getElementById('tooltip');
        const tokenSpans = document.querySelectorAll('.token');

        tokenSpans.forEach(token => {{
            token.addEventListener('mouseenter', (e) => {{
                const qwen = token.getAttribute('data-qwen') || 'N/A';
                const rwkv = token.getAttribute('data-rwkv') || 'N/A';
                const bytes = token.getAttribute('data-bytes') || '';
                const lossA = token.getAttribute('data-loss-a') || '';
                const lossB = token.getAttribute('data-loss-b') || '';

                tooltip.innerHTML = `
                    <div><span class="label">Bytes:</span> <span class="bytes">${{bytes || '(empty)'}}</span></div>
                    <div><span class="label">Loss A:</span> <span class="loss-a">${{lossA || '(empty)'}}</span></div>
                    <div><span class="label">Loss B:</span> <span class="loss-b">${{lossB || '(empty)'}}</span></div>
                    <hr style="border-color: #555; margin: 6px 0;">
                    <div><span class="label">Qwen:</span> <span class="qwen">${{qwen || '(empty)'}}</span></div>
                    <div><span class="label">RWKV:</span> <span class="rwkv">${{rwkv || '(empty)'}}</span></div>
                `;
                tooltip.style.display = 'block';
            }});

            token.addEventListener('mousemove', (e) => {{
                const x = e.clientX + 15;
                const y = e.clientY + 15;
                tooltip.style.left = x + 'px';
                tooltip.style.top = y + 'px';
            }});

            token.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});
        }});

        // Saturation slider functionality
        const avgDelta = {avg_delta};
        const slider = document.getElementById('saturation-slider');
        const saturationValue = document.getElementById('saturation-value');

        // Collect all deltas for percentile calculation
        const allDeltas = [];
        tokenSpans.forEach(token => {{
            const delta = parseFloat(token.getAttribute('data-delta'));
            if (!isNaN(delta)) allDeltas.push(delta);
        }});

        function percentile(arr, p) {{
            const sorted = [...arr].sort((a, b) => a - b);
            const idx = (p / 100) * (sorted.length - 1);
            const lower = Math.floor(idx);
            const upper = Math.ceil(idx);
            if (lower === upper) return sorted[lower];
            return sorted[lower] + (sorted[upper] - sorted[lower]) * (idx - lower);
        }}

        function deltaToColor(delta, avgDelta, maxDeviation) {{
            if (maxDeviation === 0) return 'rgb(255, 255, 255)';
            const deviation = delta - avgDelta;
            let normalized = Math.max(-1, Math.min(1, deviation / maxDeviation));
            let r, g, b;
            if (normalized < 0) {{
                const intensity = -normalized;
                r = Math.round(255 * (1 - intensity * 0.7));
                g = 255;
                b = Math.round(255 * (1 - intensity * 0.7));
            }} else {{
                const intensity = normalized;
                r = 255;
                g = Math.round(255 * (1 - intensity * 0.7));
                b = Math.round(255 * (1 - intensity * 0.7));
            }}
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        function updateColors(percentileValue) {{
            const deviations = allDeltas.map(d => Math.abs(d - avgDelta));
            const maxDeviation = Math.max(percentile(deviations, percentileValue), 1e-6);
            tokenSpans.forEach(token => {{
                const delta = parseFloat(token.getAttribute('data-delta'));
                if (!isNaN(delta)) {{
                    token.style.backgroundColor = deltaToColor(delta, avgDelta, maxDeviation);
                }}
            }});
        }}

        slider.addEventListener('input', (e) => {{
            const val = parseInt(e.target.value) / 10;
            saturationValue.textContent = val.toFixed(1) + '%';
            updateColors(val);
        }});
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def visualize_comparison(
    model_a_path: str,
    model_b_path: str,
    output_dir: str,
    sample_index: Optional[int] = None,
    font_size: int = 14,
    max_width: int = 1200,
):
    """
    Generate image visualizations comparing two models' byte-wise losses.

    Args:
        model_a_path: Path to model A's evaluation results JSON
        model_b_path: Path to model B's evaluation results JSON (baseline)
        output_dir: Directory to save image files
        sample_index: Optional specific sample index to visualize (None = all)
        font_size: Font size for text rendering
        max_width: Maximum image width
    """
    # Load results
    results_a = load_results(model_a_path)
    results_b = load_results(model_b_path)

    # Validate
    assert "single_sample_texts" in results_a, \
        "Model A results missing 'single_sample_texts'. Enable track_single_sample_byte_wise_data=True"
    assert "single_sample_texts" in results_b, \
        "Model B results missing 'single_sample_texts'. Enable track_single_sample_byte_wise_data=True"

    texts_a = results_a["single_sample_texts"]
    texts_b = results_b["single_sample_texts"]
    losses_a = results_a["single_sample_byte_wise_losses"]
    losses_b = results_b["single_sample_byte_wise_losses"]

    assert len(texts_a) == len(texts_b), \
        f"Sample count mismatch: A has {len(texts_a)}, B has {len(texts_b)}"
    assert texts_a == texts_b, \
        "Sample texts do not match between models. Ensure same dataset was used."

    # Extract clean model names from paths
    model_a_name = extract_model_name(results_a.get("model_name_or_path", "Model A"))
    model_b_name = extract_model_name(results_b.get("model_name_or_path", "Model B (Baseline)"))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine which samples to process
    if sample_index is not None:
        indices = [sample_index]
    else:
        indices = range(len(texts_a))

    for idx in indices:
        text = texts_a[idx]
        byte_losses_a = losses_a[idx]
        byte_losses_b = losses_b[idx]

        # Generate PNG image
        png_path = os.path.join(output_dir, f"sample_{idx:04d}.png")
        generate_image(
            sample_index=idx,
            text=text,
            byte_losses_a=byte_losses_a,
            byte_losses_b=byte_losses_b,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            output_path=png_path,
            font_size=font_size,
            max_width=max_width,
        )

        # Generate HTML file
        html_path = os.path.join(output_dir, f"sample_{idx:04d}.html")
        generate_html(
            sample_index=idx,
            text=text,
            byte_losses_a=byte_losses_a,
            byte_losses_b=byte_losses_b,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            output_path=html_path,
        )

        print(f"Generated: {png_path}, {html_path}")

    print(f"\nVisualization complete. {len(list(indices))} sample(s) saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize byte-wise loss comparison between two models"
    )
    parser.add_argument(
        "--model_a",
        type=str,
        required=True,
        help="Path to model A's evaluation results JSON file",
    )
    parser.add_argument(
        "--model_b",
        type=str,
        required=True,
        help="Path to model B's (baseline) evaluation results JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save PNG visualization files (default: ./visualizations)",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Specific sample index to visualize (default: all samples)",
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=14,
        help="Font size for text rendering (default: 14)",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=1200,
        help="Maximum image width in pixels (default: 1200)",
    )

    args = parser.parse_args()

    visualize_comparison(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        output_dir=args.output_dir,
        sample_index=args.sample_index,
        font_size=args.font_size,
        max_width=args.max_width,
    )


if __name__ == "__main__":
    main()
