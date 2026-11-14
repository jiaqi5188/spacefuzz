"""Tests for scripts/prune_cache.py: keep the current full-profile cache, list the rest."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prune_cache", ROOT / "scripts" / "prune_cache.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def test_current_names_cover_forecast_and_adjust_per_demo_scenario(monkeypatch):
    monkeypatch.setattr(pc.en, "_fluctuation_source", lambda: "wesad")   # same on a clone
    names = pc.current_names(midday_too=False)
    assert len(names) == 2 * len(pc.DEMO_SCENARIOS)
    for n in pc.DEMO_SCENARIOS:
        assert sum(x.startswith(f"{n}_full_") for x in names) == 2


def test_stale_never_includes_a_current_file(tmp_path, monkeypatch):
    keep = sorted(pc.current_names(midday_too=False))
    pc_current = pc.current_names
    monkeypatch.setattr(pc, "current_names", lambda *a, **k: pc_current(midday_too=False))
    for name in keep + ["main_sleep_debt_eva_full_000000000000.json",
                        "main_sleep_debt_eva_local_111111111111.json"]:
        (tmp_path / name).write_text("{}")
    assert pc.stale(tmp_path) == ["main_sleep_debt_eva_full_000000000000.json"]  # local kept
    assert set(pc.stale(tmp_path, all_profiles=True)) == {
        "main_sleep_debt_eva_full_000000000000.json", "main_sleep_debt_eva_local_111111111111.json"}
    pc.prune(tmp_path, delete=False)
    assert len(list(tmp_path.iterdir())) == len(keep) + 2          # dry run keeps everything
    pc.prune(tmp_path, delete=True)                                # default: full profile only
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        keep + ["main_sleep_debt_eva_local_111111111111.json"])
    pc.prune(tmp_path, delete=True, all_profiles=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == keep


def test_the_committed_cache_is_current(monkeypatch):
    """Every result the demo reads is in cache/ under the current key (a stale cache would only
    make the app tests skip, so this one does not skip). The committed files are real-WESAD
    results, also on a clone without the series."""
    monkeypatch.setattr(pc.en, "_fluctuation_source", lambda: "wesad")
    have = {p.name for p in pc.en.CACHE_DIR.glob("*_full_*.json")}
    missing = pc.current_names() - have
    assert not missing, f"stale or missing full-profile cache: {sorted(missing)}"


def test_a_fresh_clone_keeps_the_committed_cache(monkeypatch):
    """Without the WESAD series the committed (real-WESAD) files are still current."""
    with_real = pc.current_names()
    monkeypatch.setattr(pc.en, "_fluctuation_source", lambda: "wesad_params")
    assert with_real <= pc.current_names()
