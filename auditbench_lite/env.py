"""Load configuration from a .env file."""

from pathlib import Path
from typing import Optional

_loaded = False


def load_env() -> Optional[Path]:
    """Load the first .env found in the cwd or any parent directory.

    Returns:
        Path to the loaded .env file, or None if no file was found.
    """
    global _loaded
    if _loaded:
        return None

    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    for directory in [Path.cwd(), *Path.cwd().parents]:
        env_file = directory / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            _loaded = True
            return env_file

    return None
