"""Git utilities."""

import subprocess

from arxiv_classifier.utils.logger import get_logger

logger = get_logger(__name__)


def get_git_commit() -> str:
    """Get current git commit hash.

    Returns:
        Git commit hash or "unknown" if git is not available
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:
        logger.warning(f"Failed to get git commit: {e}")
        return "unknown"
