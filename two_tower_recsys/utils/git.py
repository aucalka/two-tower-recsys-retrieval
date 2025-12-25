from __future__ import annotations

import subprocess


def get_git_commit_id() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
