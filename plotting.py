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


def safe_savefig(fig, path: str, dpi: int = 110, retries: int = 2) -> bool:
    for attempt in range(retries + 1):
        try:
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            return True
        except OSError as e:
            if attempt < retries:
                log.warning("Could not write %s (%s); retrying…", path, e)
                time.sleep(0.6)
            else:
                log.error("Skipping plot %s after %d tries (%s). This is usually "
                          "a OneDrive file lock — consider moving the project out "
                          "of OneDrive.", path, retries + 1, e)
    plt.close(fig)
    return False
