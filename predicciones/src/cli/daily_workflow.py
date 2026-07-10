"""
Guided workflows for modules 1 and 2.

Module 1 (Predict) focuses on selecting or generating a usable fixture CSV.
Module 2 (Daily Pipeline) orchestrates data refresh, derived rebuilds, fixture
generation, and optional predictions.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIRS = [
    PROJECT_ROOT / "data" / "fixtures",
    PROJECT_ROOT / "predicciones" / "data" / "fixtures",
]
AUTO_FIXTURE_DIR = PROJECT_ROOT / "data" / "fixtures" / "auto"
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"

REQUIRED_FIXTURE_COLUMNS = {"home_team", "away_team"}


def _today() -> datetime:
    return datetime.now()


def _date_key(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def _display_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _read_fixture_file(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def discover_fixture_files() -> List[Dict[str, Any]]:
    """Find existing CSV fixture files and summarize them for menu selection."""
    fixtures: List[Dict[str, Any]] = []
    seen = set()
    for fixture_dir in FIXTURE_DIRS:
        if not fixture_dir.exists():
            continue
        for path in sorted(fixture_dir.rglob("*.csv")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            df = _read_fixture_file(path)
            fixtures.append({
                "path": path,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "date": path.stem,
                "matches": len(df),
                "valid": validate_fixture_df(df)[0],
                "competitions": ", ".join(sorted(df.get("competition", pd.Series(dtype=str)).dropna().astype(str).unique())[:3]),
                "preview": df.head(5),
            })
    return fixtures


def validate_fixture_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = sorted(REQUIRED_FIXTURE_COLUMNS - set(df.columns))
    if len(df) == 0:
        return False, ["fixture file has no rows"]
    if missing:
        return False, [f"missing required column: {col}" for col in missing]
    return True, []


def render_fixture_preview(console: Console, df: pd.DataFrame, title: str = "Fixture Preview", limit: int = 10) -> None:
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Match", style="white")
    table.add_column("Date", style="dim")
    table.add_column("Competition", style="dim")
    rows = df.head(limit)
    for idx, row in rows.iterrows():
        table.add_row(
            str(idx + 1),
            f"{row.get('home_team', 'Home')} vs {row.get('away_team', 'Away')}",
            str(row.get("date", ""))[:10],
            str(row.get("competition") or row.get("league") or ""),
        )
    console.print(table)
    if len(df) > limit:
        console.print(f"[dim]... and {len(df) - limit} more match(es)[/dim]")


def _fetch_fixtures_for_date(date_key: str) -> pd.DataFrame:
    from predicciones.scripts.fetch_daily_fixtures import fetch_fixtures_for_date

    return fetch_fixtures_for_date(date_key)


def _save_auto_fixture(df: pd.DataFrame, label: str) -> Path:
    AUTO_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    output_path = AUTO_FIXTURE_DIR / f"{safe_label}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def detect_today_fixtures() -> Tuple[pd.DataFrame, str]:
    date_key = _date_key(_today())
    df = _fetch_fixtures_for_date(date_key)
    return df, f"today_{date_key}"


def detect_upcoming_fixtures(days: int = 7) -> Tuple[pd.DataFrame, str]:
    frames = []
    start = _today()
    for offset in range(days):
        date_key = _date_key(start + timedelta(days=offset))
        df = _fetch_fixtures_for_date(date_key)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), f"upcoming_{_date_key(start)}"
    return pd.concat(frames, ignore_index=True).drop_duplicates(), f"upcoming_{_date_key(start)}_{days}d"


def load_all_detected_fixtures() -> Tuple[pd.DataFrame, str]:
    frames = []
    for fixture in discover_fixture_files():
        df = _read_fixture_file(fixture["path"])
        if len(df) > 0:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), "all_detected"
    return pd.concat(frames, ignore_index=True).drop_duplicates(), "all_detected"


def filter_by_competition(df: pd.DataFrame, console: Console) -> pd.DataFrame:
    column = "competition" if "competition" in df.columns else "league" if "league" in df.columns else None
    if not column:
        console.print("[yellow]No competition column available in detected fixtures.[/yellow]")
        return pd.DataFrame()
    competitions = sorted(df[column].dropna().astype(str).unique())
    if not competitions:
        console.print("[yellow]No competitions found in detected fixtures.[/yellow]")
        return pd.DataFrame()
    for idx, competition in enumerate(competitions, 1):
        console.print(f"  [cyan]{idx}.[/cyan] {competition}")
    choice = _ask_index("Select competition", len(competitions))
    if choice is None:
        return pd.DataFrame()
    return df[df[column].astype(str) == competitions[choice]]


def filter_by_team(df: pd.DataFrame, console: Console) -> pd.DataFrame:
    teams = sorted(set(df.get("home_team", pd.Series(dtype=str)).dropna().astype(str)) | set(df.get("away_team", pd.Series(dtype=str)).dropna().astype(str)))
    if not teams:
        console.print("[yellow]No teams found in detected fixtures.[/yellow]")
        return pd.DataFrame()
    for idx, team in enumerate(teams, 1):
        console.print(f"  [cyan]{idx}.[/cyan] {team}")
    choice = _ask_index("Select team", len(teams))
    if choice is None:
        return pd.DataFrame()
    team = teams[choice]
    return df[(df["home_team"].astype(str) == team) | (df["away_team"].astype(str) == team)]


def _ask_index(prompt: str, count: int, allow_back: bool = True) -> Optional[int]:
    while True:
        raw = Prompt.ask(f"[cyan]{prompt} (1-{count})[/cyan]").strip()
        upper = raw.upper()
        if upper == "Q":
            raise KeyboardInterrupt
        if allow_back and upper == "B":
            return None
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        back = "B to go back, " if allow_back else ""
        print(f"Invalid option. Enter 1-{count}, {back}or Q to exit.")


def run_prediction_for_fixture(console: Console, fixture_path: Path, verbose: bool = False) -> None:
    from predicciones.src.cli.commands import predict_command

    predict_command(str(fixture_path), "output/predictions", verbose)


def run_predict_menu(console: Optional[Console] = None) -> None:
    active_console = console or Console()
    while True:
        active_console.print(Panel(
            "[bold]Predict[/bold]\n\n"
            "  [cyan]1.[/cyan] Predict from auto-generated fixtures\n"
            "  [cyan]2.[/cyan] Predict from existing fixture file\n"
            "  [cyan]B.[/cyan] Back\n"
            "  [cyan]Q.[/cyan] Exit",
            title="Predict",
            border_style="blue",
        ))
        choice = Prompt.ask("[cyan]Choose an option[/cyan]").strip().upper()
        if choice in {"B", "Q"}:
            return
        if choice == "1":
            _run_auto_predict(active_console)
        elif choice == "2":
            _run_existing_fixture_predict(active_console)
        else:
            active_console.print("[yellow]Invalid option. Please choose 1, 2, B, or Q.[/yellow]")


def _run_auto_predict(console: Console) -> None:
    while True:
        console.print(Panel(
            "  [cyan]1.[/cyan] Today's fixtures\n"
            "  [cyan]2.[/cyan] Upcoming fixtures\n"
            "  [cyan]3.[/cyan] By competition\n"
            "  [cyan]4.[/cyan] By team\n"
            "  [cyan]5.[/cyan] All detected fixtures\n"
            "  [cyan]B.[/cyan] Back\n"
            "  [cyan]Q.[/cyan] Exit",
            title="Auto-generated Fixtures",
            border_style="cyan",
        ))
        choice = Prompt.ask("[cyan]Choose fixture source[/cyan]").strip().upper()
        if choice == "Q":
            raise KeyboardInterrupt
        if choice == "B":
            return

        if choice == "1":
            df, label = detect_today_fixtures()
        elif choice == "2":
            df, label = detect_upcoming_fixtures()
        elif choice in {"3", "4"}:
            df, label = load_all_detected_fixtures()
            if len(df) == 0:
                df, label = detect_upcoming_fixtures()
            df = filter_by_competition(df, console) if choice == "3" else filter_by_team(df, console)
            label = f"{label}_filtered"
        elif choice == "5":
            df, label = load_all_detected_fixtures()
        else:
            console.print("[yellow]Invalid option.[/yellow]")
            continue

        if len(df) == 0:
            console.print("[yellow]No fixtures found for this selection.[/yellow]")
            continue
        valid, issues = validate_fixture_df(df)
        render_fixture_preview(console, df, "Detected Fixtures")
        if not valid:
            console.print("[red]Fixture selection is not valid:[/red]")
            for issue in issues:
                console.print(f"  - {issue}")
            continue
        if Confirm.ask("[cyan]Generate fixture file and run predictions?[/cyan]", default=True):
            path = _save_auto_fixture(df, label)
            console.print(f"[green]✓ Fixture file generated: {path}[/green]")
            run_prediction_for_fixture(console, path)


def _run_existing_fixture_predict(console: Console) -> None:
    fixtures = discover_fixture_files()
    if not fixtures:
        console.print("[yellow]No fixture files found.[/yellow]")
        console.print("[dim]Use auto-generated fixtures instead.[/dim]")
        return

    table = Table(title="Existing Fixture Files", show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Path", style="white")
    table.add_column("Matches", justify="right")
    table.add_column("Valid", justify="center")
    table.add_column("Competitions", style="dim")
    for idx, fixture in enumerate(fixtures, 1):
        table.add_row(
            str(idx),
            fixture["relative_path"],
            str(fixture["matches"]),
            "yes" if fixture["valid"] else "no",
            fixture["competitions"],
        )
    console.print(table)
    console.print("[dim]B = back, Q = exit[/dim]")
    try:
        idx = _ask_index("Select fixture file", len(fixtures))
    except KeyboardInterrupt:
        return
    if idx is None:
        return
    fixture = fixtures[idx]
    df = _read_fixture_file(fixture["path"])
    render_fixture_preview(console, df, "Fixture File Preview")
    valid, issues = validate_fixture_df(df)
    if not valid:
        console.print("[red]Selected fixture is not valid:[/red]")
        for issue in issues:
            console.print(f"  - {issue}")
        return
    if Confirm.ask("[cyan]Run predictions for this fixture file?[/cyan]", default=True):
        run_prediction_for_fixture(console, fixture["path"])


def update_raw_data_sources(console: Console) -> Dict[str, Any]:
    """Refresh currently actionable raw data sources without asking for IDs."""
    results: Dict[str, Any] = {}
    today = _date_key(_today())
    console.print("[bold]Updating raw fixture source for today[/bold]")
    try:
        df, label = detect_today_fixtures()
        path = _save_auto_fixture(df, label)
        results["today_fixtures"] = {"status": "ok", "rows": len(df), "path": path}
        console.print(f"[green]✓ Fixtures refreshed: {path} ({len(df)} match(es))[/green]")
    except Exception as exc:
        results["today_fixtures"] = {"status": "error", "message": str(exc)}
        console.print(f"[red]Fixture refresh failed: {exc}[/red]")
    return results


def rebuild_derived_datasets(console: Console) -> Dict[str, Any]:
    """Run available derived dataset rebuilders and report outputs."""
    results: Dict[str, Any] = {}
    commands = [
        ("player_match_stats", [sys.executable, "predicciones/scripts/regenerate_player_match_stats.py", "--all-from-team-stats"]),
    ]
    for name, cmd in commands:
        console.print(f"[bold]Rebuilding {name}[/bold]")
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=120)
            status = "ok" if proc.returncode == 0 else "error"
            results[name] = {"status": status, "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]}
            if proc.returncode == 0:
                console.print(f"[green]✓ {name} rebuilt[/green]")
            else:
                console.print(f"[red]{name} failed[/red]\n{proc.stderr[-1000:]}")
        except Exception as exc:
            results[name] = {"status": "error", "message": str(exc)}
            console.print(f"[red]{name} failed: {exc}[/red]")
    for path in [DERIVED_DIR / "player_match_stats.jsonl", DERIVED_DIR / "match_events.jsonl", DERIVED_DIR / "team_match_stats.jsonl"]:
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            console.print(f"[dim]{path}: {path.stat().st_size} bytes, updated {mtime}[/dim]")
    return results


def detect_generate_today_fixtures(console: Console) -> Optional[Path]:
    df, label = detect_today_fixtures()
    if len(df) == 0:
        console.print("[yellow]No fixtures found for today.[/yellow]")
        return None
    render_fixture_preview(console, df, "Today's Fixtures")
    path = _save_auto_fixture(df, label)
    console.print(f"[green]✓ Today's fixture file ready: {path}[/green]")
    return path


def run_full_daily_pipeline(console: Console, with_predictions: bool = False) -> Optional[Path]:
    update_raw_data_sources(console)
    rebuild_derived_datasets(console)
    fixture_path = detect_generate_today_fixtures(console)
    if with_predictions and fixture_path:
        run_prediction_for_fixture(console, fixture_path)
    elif fixture_path:
        console.print("[green]✓ Ready to predict.[/green]")
    return fixture_path


def run_daily_pipeline_menu(console: Optional[Console] = None) -> None:
    active_console = console or Console()
    while True:
        active_console.print(Panel(
            "[bold]Run Daily Pipeline[/bold]\n\n"
            "  [cyan]1.[/cyan] Update raw data sources\n"
            "  [cyan]2.[/cyan] Rebuild derived datasets\n"
            "  [cyan]3.[/cyan] Detect/generate today fixtures\n"
            "  [cyan]4.[/cyan] Run full daily pipeline\n"
            "  [cyan]5.[/cyan] Run full daily pipeline + predictions\n"
            "  [cyan]B.[/cyan] Back\n"
            "  [cyan]Q.[/cyan] Exit",
            title="Daily Pipeline",
            border_style="blue",
        ))
        choice = Prompt.ask("[cyan]Choose an option[/cyan]").strip().upper()
        if choice in {"B", "Q"}:
            return
        if choice == "1":
            update_raw_data_sources(active_console)
        elif choice == "2":
            rebuild_derived_datasets(active_console)
        elif choice == "3":
            detect_generate_today_fixtures(active_console)
        elif choice == "4":
            run_full_daily_pipeline(active_console, with_predictions=False)
        elif choice == "5":
            run_full_daily_pipeline(active_console, with_predictions=True)
        else:
            active_console.print("[yellow]Invalid option. Please choose 1-5, B, or Q.[/yellow]")
