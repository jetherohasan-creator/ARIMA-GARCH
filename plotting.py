"""
Resilient figure saving.

On Windows, when the project lives inside a OneDrive-synced folder, OneDrive can
transiently lock/dehydrate a PNG while it syncs, so ``fig.savefig`` may fail with
``OSError: [Errno 22] Invalid argument``. A single locked plot should not crash
the whole pipeline (the forecasts/CSVs are the important output), so we retry a
couple of times and then skip with a warning.
"""
from __future__ import annotations

import logging
import time

import matplotlib.pyplot as plt

log = logging.getLogger("plotting")


def safe_savefig(fig, path: str, dpi: int = 110, retries: int = 4) -> bool:
    for attempt in range(retries + 1):
        try:
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            return True
        except OSError as e:
            if attempt < retries:
                log.warning("Could not write %s (%s); retrying…", path, e)
                time.sleep(0.8)
            else:
                log.error("Skipping plot %s after %d tries (%s). This is usually "
                          "a OneDrive file lock — pause OneDrive sync or move the "
                          "project out of OneDrive.", path, retries + 1, e)
    plt.close(fig)
    return False


def safe_to_csv(df, path: str, retries: int = 4, **kwargs) -> bool:
    """Write a DataFrame/Series to CSV, retrying through transient OneDrive
    PermissionError / OSError locks. Returns True on success."""
    for attempt in range(retries + 1):
        try:
            df.to_csv(path, **kwargs)
            return True
        except (OSError, PermissionError) as e:
            if attempt < retries:
                log.warning("Could not write %s (%s); retrying…", path, e)
                time.sleep(0.8)
            else:
                log.error("FAILED to write %s after %d tries (%s). This is "
                          "usually a OneDrive file lock — pause OneDrive sync "
                          "(right-click tray icon → Pause syncing) or move the "
                          "project out of OneDrive, then re-run.",
                          path, retries + 1, e)
    return False
