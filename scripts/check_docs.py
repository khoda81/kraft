#!/usr/bin/env python3
"""Check relative Markdown file links (not anchors or external URLs)."""
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

root = Path(__file__).resolve().parents[1]
failures = []
paths = [*root.glob("*.md"), *root.joinpath("docs").rglob("*.md"), root / "artifacts/README.md"]
for path in paths:
    for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", path.read_text()):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        if not (path.parent / unquote(parsed.path)).exists():
            failures.append(f"{path.relative_to(root)}: missing {target}")
if failures:
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
print(f"Checked relative file links in {len(paths)} Markdown files.")
