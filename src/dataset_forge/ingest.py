from __future__ import annotations

import re
from dataclasses import dataclass

from dataset_forge.models import SourceDocument


@dataclass(slots=True)
class SourceSegment:
    segment_id: str
    source_id: str
    title: str
    text: str
    speaker: str | None
    index: int


_SPEAKER_LINE_RE = re.compile(r"^(?P<speaker>[A-Za-z][A-Za-z0-9 _.-]{0,48}):\s*(?P<text>.+)$")


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def segment_sources(sources: list[SourceDocument]) -> list[SourceSegment]:
    segments: list[SourceSegment] = []
    for source in sources:
        normalized = normalize_text(source.text)
        speaker_segments = _speaker_segments(source, normalized)
        if speaker_segments:
            segments.extend(speaker_segments)
            continue
        for index, block in enumerate(_paragraph_blocks(normalized)):
            if len(block.split()) < 10:
                continue
            segments.append(
                SourceSegment(
                    segment_id=f"{source.source_id}:block:{index}",
                    source_id=source.source_id,
                    title=source.title,
                    text=block,
                    speaker=None,
                    index=index,
                )
            )
    return segments


def _speaker_segments(source: SourceDocument, text: str) -> list[SourceSegment]:
    segments: list[SourceSegment] = []
    pending_speaker: str | None = None
    pending_lines: list[str] = []
    index = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _SPEAKER_LINE_RE.match(line)
        if match:
            if pending_lines:
                segment_text = " ".join(pending_lines).strip()
                if len(segment_text.split()) >= 8:
                    segments.append(
                        SourceSegment(
                            segment_id=f"{source.source_id}:turn:{index}",
                            source_id=source.source_id,
                            title=source.title,
                            text=segment_text,
                            speaker=pending_speaker,
                            index=index,
                        )
                    )
                    index += 1
            pending_speaker = match.group("speaker").strip()
            pending_lines = [match.group("text").strip()]
        elif pending_lines:
            pending_lines.append(line)
    if pending_lines:
        segment_text = " ".join(pending_lines).strip()
        if len(segment_text.split()) >= 8:
            segments.append(
                SourceSegment(
                    segment_id=f"{source.source_id}:turn:{index}",
                    source_id=source.source_id,
                    title=source.title,
                    text=segment_text,
                    speaker=pending_speaker,
                    index=index,
                )
            )
    return segments if len(segments) >= 2 else []


def _paragraph_blocks(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) > 1:
        return blocks
    sentences = re.split(r"(?<=[.!?])\s+", text)
    grouped: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        current.append(sentence.strip())
        if sum(len(item.split()) for item in current) >= 45:
            grouped.append(" ".join(current))
            current = []
    if current:
        grouped.append(" ".join(current))
    return grouped
