"""Expose the useful tail of a unittest log as a GitHub check annotation."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Emit one escaped workflow-command annotation for a failed test run."""
    path = Path(sys.argv[1])
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    excerpt = "\n".join(lines[-120:])
    escaped = (
        excerpt.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    )
    print(f"::error title=Unit test failure::{escaped}")


if __name__ == "__main__":
    main()
