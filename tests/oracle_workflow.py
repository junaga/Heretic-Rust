"""Test-only upstream oracle. This file is never loaded by the Rust application."""
import sys
from pathlib import Path

upstream, config, output = sys.argv[1:]
sys.path.insert(0, str(Path(upstream) / "src"))
sys.argv = ["heretic"]

import heretic.main as original

# Replace only interactive input and distribution lookup. All model/search/export
# operations remain the unmodified pinned upstream implementation.
original.version = lambda _: "1.4.0"
selected = False
saved = False


def choose(message, choices):
    global selected, saved
    if message == "Which trial do you want to use?" and not selected:
        selected = True
        return choices[0].value
    if message == "What do you want to do with the decensored model?" and not saved:
        saved = True
        return "Save the model to a local folder"
    return None


original.prompt_select = choose
original.prompt_path = lambda _: output
original.run()
