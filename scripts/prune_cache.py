"""List (or delete) engine cache files that the current code no longer reads.

The full-profile cache of the demo scenarios is committed (see .gitignore) so a fresh clone
shows the demo results at once, offline. Cache file names carry a hash of the inputs and of
the engine code, so every code change leaves stale files behind; run this before a commit.

    python scripts/prune_cache.py                  # dry run: list stale full-profile files
    python scripts/prune_cache.py --delete         # delete them
    python scripts/prune_cache.py --delete --all   # also every other profile's cache (local
                                                   # runs take minutes to rebuild)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spacefuzz.fuzzing import adjuster  # noqa: E402
from spacefuzz.fuzzing import engine as en  # noqa: E402
from spacefuzz.fuzzing import midday  # noqa: E402
from spacefuzz.fuzzing import mutator as mu  # noqa: E402
import ui_config  # noqa: E402

DEMO_SCENARIOS = ("main_sleep_debt_eva", "radiation_spe", "backup_high_co2")


def current_names(profile: str = "full", midday_too: bool = True) -> set[str]:
    """Forecast, adjustment (and midday demo) cache names the current code would read (same
    keys as engine.forecast, adjuster.adjust and midday.forecast_midday)."""
    prof = en.profile(profile)
    keys = []
    for name in DEMO_SCENARIOS:
        sc = mu.Scenario.load(name)
        keys += [(f"{name}_{prof['name']}_", en._cache_key(sc, prof)),
                 adjuster.cache_key(name, sc, prof)]
    if midday_too:
        cut = ui_config.MIDDAY_CHECK_MIN
        for name, values in ui_config.MIDDAY_EXAMPLE.items():     # the midday demo result
            try:
                sc = midday.scenario_for(name, values, profile)
            except midday.MorningMissing:     # no morning result yet: no midday file is current
                continue
            keys.append(midday.cache_key(name, sc, prof, values, cut))
    return {n for stem, key in keys for n in en.cache_names(stem, key)}


def stale(cache_dir: Path = en.CACHE_DIR, all_profiles: bool = False) -> list[str]:
    """Files the current code no longer reads: full-profile ones, or every profile's."""
    keep = current_names()
    pattern = "*.json" if all_profiles else "*_full_*.json"
    return sorted(p.name for p in Path(cache_dir).glob(pattern) if p.name not in keep)


def prune(cache_dir: Path = en.CACHE_DIR, delete: bool = False,
          all_profiles: bool = False) -> list[str]:
    old = stale(cache_dir, all_profiles)
    for name in old:
        if delete:
            (Path(cache_dir) / name).unlink()
    return old


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delete", action="store_true", help="delete the stale files")
    ap.add_argument("--all", action="store_true",
                    help="also other profiles' cache (default: full profile only)")
    args = ap.parse_args()
    old = prune(delete=args.delete, all_profiles=args.all)
    keep = sorted(current_names())
    print(f"current ({len(keep)}): " + ", ".join(keep))
    print(f"{'deleted' if args.delete else 'stale (dry run)'}: {len(old)} files")
    missing = [n for n in keep if not (Path(en.CACHE_DIR) / n).exists()]
    if missing:
        print("missing, re-run the full profile: " + ", ".join(missing))


if __name__ == "__main__":
    main()
