"""Progress lines and summaries for the stage scripts: printed, and written to the stage's log in its artifact
folder, so every artifact folder holds the log of the run that wrote it.

    log.to_file(path)   start the stage's log (overwritten on every run)
    log.log(message)    a timestamped progress line
    log.text(block)     a multi-line summary or report, as it is
"""

import os
import time

_file = None


def to_file(path):
    global _file
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _file = open(path, "w")


def text(block):
    print(block, flush=True)
    if _file is not None:
        _file.write(block + "\n")
        _file.flush()


def log(message):
    text(f"{time.strftime('%H:%M:%S')} {message}")
