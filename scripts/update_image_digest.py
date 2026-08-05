#!/usr/bin/env python3
"""Update the GitOps image digest and release ID without rewriting YAML."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DIGEST = re.compile(r"^(\s*digest:\s*)[^#\n]+(\s*(?:#.*)?)$", re.MULTILINE)
RELEASE = re.compile(r"^(\s*releaseId:\s*)[^#\n]+(\s*(?:#.*)?)$", re.MULTILINE)


def update(path: Path, digest: str, release_id: str) -> None:
    text = path.read_text(encoding="utf-8")
    replaced, digest_count = DIGEST.subn(rf"\g<1>{digest}\g<2>", text, count=1)
    replaced, release_count = RELEASE.subn(
        rf"\g<1>{release_id}\g<2>", replaced, count=1
    )
    if digest_count != 1 or release_count != 1:
        raise ValueError(f"{path} must contain one image.digest and one releaseId")
    path.write_text(replaced, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest", required=True, help="sha256:<64 hex chars>")
    parser.add_argument("--release-id", required=True, help="Git SHA or release ID")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.digest):
        parser.error("--digest must be a sha256 digest")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{6,62}", args.release_id):
        parser.error("--release-id is not a safe Kubernetes name fragment")
    for raw in args.paths:
        update(Path(raw), args.digest, args.release_id.lower())


if __name__ == "__main__":
    main()
