#!/usr/bin/env python3
"""Check tracked Markdown's local links, heading anchors, and fenced blocks.

External URLs are intentionally not fetched: historical source references may
be immutable, private, or subject to network instability.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*$")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)", re.DOTALL)


def visible_markdown(source: str, path: Path) -> tuple[str, list[str]]:
    """Blank fenced code while retaining line numbers for link diagnostics."""
    output = []
    errors = []
    marker = ""
    opened_at = 0
    for number, line in enumerate(source.splitlines(), 1):
        match = FENCE.match(line)
        if marker:
            if match and match.group(1)[0] == marker[0] and len(match.group(1)) >= len(marker):
                if not line[match.end() :].strip():
                    marker = ""
            output.append("")
        elif match:
            marker = match.group(1)
            opened_at = number
            output.append("")
        else:
            output.append(line)
    if marker:
        errors.append(f"{path}:{opened_at}: unclosed fenced code block")
    return "\n".join(output), errors


def anchor_ids(text: str) -> set[str]:
    """Apply the GitHub-style heading-slug rules used by project headings."""
    counts: dict[str, int] = {}
    anchors = set()
    for line in text.splitlines():
        match = HEADING.match(line)
        if not match:
            continue
        heading = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(1)).strip()
        heading = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        heading = re.sub(r"<[^>]*>", "", heading)
        heading = html.unescape(heading).lower().replace("`", "")
        heading = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        number = counts.get(heading, 0)
        anchors.add(f"{heading}-{number}" if number else heading)
        counts[heading] = number + 1
    return anchors


def check_markdown(root: Path, paths: list[Path]) -> list[str]:
    errors = []
    content = {}
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            errors.append(f"{path}: cannot read: {error}")
            continue
        shown, fence_errors = visible_markdown(source, path)
        errors.extend(fence_errors)
        content[path.resolve()] = shown

    cached_anchors = {path: anchor_ids(shown) for path, shown in content.items()}
    for path, shown in content.items():
        for match in LINK.finditer(shown):
            raw = match.group(1).strip()
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1]
            # Markdown link titles are not repository paths.
            raw = re.split(r'\s+["\']', raw, maxsplit=1)[0]
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or raw.startswith("//"):
                continue
            line = shown.count("\n", 0, match.start()) + 1
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not target.is_relative_to(root):
                errors.append(f"{path}:{line}: link leaves repository: {raw}")
            elif not target.exists():
                errors.append(f"{path}:{line}: missing local target: {raw}")
            elif parsed.fragment and target.suffix.lower() == ".md":
                fragment = unquote(parsed.fragment)
                if fragment not in cached_anchors.get(target, set()):
                    errors.append(f"{path}:{line}: missing Markdown anchor: {raw}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=root, capture_output=True, check=True,
    )
    paths = [(root / item.decode("utf-8")).resolve() for item in result.stdout.split(b"\0") if item]
    errors = check_markdown(root, paths)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"Documentation check failed: {len(errors)} error(s).", file=sys.stderr)
        return 1
    print(f"Documentation check passed: {len(paths)} Markdown files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
