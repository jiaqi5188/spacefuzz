"""Shared test setup: no playback delays in any test, whatever order the modules load in
(ui_config reads SPACEFUZZ_ANIM_SCALE once, at import)."""
import os

os.environ.setdefault("SPACEFUZZ_ANIM_SCALE", "0")
# The optional AI hint (ui/assist.py) stays off in every test, even with a key in .env.
os.environ["SPACEFUZZ_ASSIST"] = "off"
