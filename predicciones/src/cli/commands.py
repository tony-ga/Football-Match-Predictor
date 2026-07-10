"""
CLI Commands for Football Prediction System

This module implements all command functions used by app.py.
Each function wraps existing project functionality.
Includes helper functions for discovering available data.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import csv
import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, IntPrompt, Confirm

console = Console()

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Helper Functions for Data Discovery
# =============================================================================

def list_available_fixtures() -> List[Dict[str, Any]]:
    """List available fixture files in data/fixtures directories."""
    fixtures = []
    
    # Check both possible locations
    fixture_dirs = [
        project_root / "data" / "fixtures",
        project_root / "predicciones" / "data" / "fixtures",
    ]
    
    for fixture_dir in fixture_dirs:
        if fixture_dir.exists():
            for file in sorted(fixture_dir.glob("*.csv")):
                try:
                    # Read first line to get date from filename or content
                    date_str = file.stem  # e.g., "20250715"
                    
                    # Count matches in file
                    with open(file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        match_count = sum(1 for _ in reader)
                    
                    fixtures.append({
                        "path": str(file.relative_to(project_root)),
                        "date": date_str,
                        "matches": match_count,
                        "size_kb": round(file.stat().st_size / 1024, 1),
                    })
                except Exception:
                    continue
    
    return fixtures


def list_available_timelines() -> List[Dict[str, Any]]:
    """
    List available timeline data from ALL available sources in the project.
    
    Searches:
    - output/*.json files with timeline/event data
    - output/*timeline*.json files
    - output/*recap*.json files (match summaries)
    - output/advanced_team_stats*.json
    - data/examples/*.json
    - Any JSON file containing 'events' or 'match' keys
    
    Returns deduplicated list prioritizing World Cup matches.
    """
    timelines = []
    
    # Define all search locations
    search_patterns = [
        # Output directory patterns
        (project_root / "output", "*.json"),
        (project_root / "predicciones" / "output", "*.json"),
        # Data examples
        (project_root / "data" / "examples", "*.json"),
        (project_root / "predicciones" / "data" / "examples", "*.json"),
    ]
    
    processed_ids = set()
    
    for base_dir, pattern in search_patterns:
        if not base_dir.exists():
            continue
            
        for file in base_dir.glob(pattern):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract event_id
                event_id = data.get('event_id')
                if not event_id:
                    continue
                
                # Skip if already processed
                if event_id in processed_ids:
                    continue
                processed_ids.add(event_id)
                
                # Extract match info from various possible locations
                match_info = data.get('match', {})
                if not match_info:
                    # Try to build from root level
                    match_info = {
                        'short_name': data.get('short_name', ''),
                        'home_team': data.get('home_team', ''),
                        'away_team': data.get('away_team', ''),
                        'status': data.get('status', ''),
                        'date': data.get('date', ''),
                    }
                
                # Build fixture name
                if match_info.get('short_name'):
                    fixture = match_info['short_name']
                elif match_info.get('home_team') and match_info.get('away_team'):
                    fixture = f"{match_info['home_team']} vs {match_info['away_team']}"
                else:
                    fixture = f"Match {event_id}"
                
                # Count events if available
                events_count = 0
                if 'events' in data:
                    events_count = len(data['events'])
                elif 'sources' in data and isinstance(data['sources'], dict):
                    events_count = data['sources'].get('total_events', 0)
                
                # Determine competition/source type
                competition = data.get('league', 'N/A')
                source_type = "timeline" if 'events' in data else "recap" if 'team_stats' in data else "stats"
                
                # Check if World Cup related
                is_world_cup = (
                    competition == 'fifa.world' or
                    'fifa' in str(file).lower() or
                    'world' in str(file).lower() or
                    any(team in fixture for team in ['Argentina', 'Brasil', 'France', 'England', 'España', 'Alemania', 'Italy'])
                )
                
                timelines.append({
                    "match_id": str(event_id),
                    "fixture": fixture,
                    "date": match_info.get('date', 'N/A') or 'N/A',
                    "competition": competition,
                    "events": events_count,
                    "status": match_info.get('status', 'N/A'),
                    "source_type": source_type,
                    "path": str(file.relative_to(project_root)),
                    "is_world_cup": is_world_cup,
                })
                
            except Exception as e:
                logger.debug(f"Error processing {file}: {e}")
                continue
    
    # Sort: World Cup matches first, then by event count (more events = richer data)
    timelines.sort(key=lambda x: (not x['is_world_cup'], -x['events']))
    
    return timelines


def list_available_teams() -> List[str]:
    """
    Extract unique team names from ALL available sources, prioritizing World Cup national teams.
    
    Searches:
    - Fixture CSVs in data/fixtures
    - Prediction CSVs in output/daily_predictions
    - JSON outputs with match/team data
    - ratings_wc2026.json for official World Cup teams
    
    Uses team_normalization module to deduplicate teams with aliases (e.g., Francia/France).
    Returns sorted list with World Cup national teams first, using Spanish display names.
    """
    from predicciones.src.utils.team_normalization import get_unique_teams_for_menu
    
    # First, try to get unique teams from the normalization module
    # This handles deduplication of aliases like Francia/France
    unique_teams = get_unique_teams_for_menu()
    
    if unique_teams:
        # Return only display names (Spanish preferred)
        return [display_name for display_name, _ in unique_teams]
    
    # Fallback to original behavior if normalization fails
    teams = set()
    world_cup_teams = set()
    
    # Priority 1: Load from ratings_wc2026.json (official World Cup teams)
    wc_ratings_paths = [
        project_root / "data" / "ratings_wc2026.json",
        project_root / "predicciones" / "data" / "ratings_wc2026.json",
    ]
    
    for wc_path in wc_ratings_paths:
        if wc_path.exists():
            try:
                with open(wc_path, 'r', encoding='utf-8') as f:
                    wc_data = json.load(f)
                wc_teams = wc_data.get('teams', {})
                # Prefer Spanish names (they are primary keys in ratings file)
                world_cup_teams.update(wc_teams.keys())
            except Exception:
                continue
    
    # Priority 2: From fixtures CSVs
    fixture_dirs = [
        project_root / "data" / "fixtures",
        project_root / "predicciones" / "data" / "fixtures",
    ]
    
    for fixture_dir in fixture_dirs:
        if fixture_dir.exists():
            for file in fixture_dir.glob("*.csv"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if 'home_team' in row and row['home_team']:
                                teams.add(row['home_team'])
                            if 'away_team' in row and row['away_team']:
                                teams.add(row['away_team'])
                except Exception:
                    continue
    
    # Priority 3: From prediction CSVs
    pred_dirs = [
        project_root / "output" / "daily_predictions",
        project_root / "predicciones" / "output" / "daily_predictions",
    ]
    
    for pred_dir in pred_dirs:
        if pred_dir.exists():
            for file in pred_dir.glob("*.csv"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if 'home_team' in row and row['home_team']:
                                teams.add(row['home_team'])
                            if 'away_team' in row and row['away_team']:
                                teams.add(row['away_team'])
                except Exception:
                    continue
    
    # Priority 4: From JSON outputs
    json_dirs = [
        project_root / "output",
        project_root / "predicciones" / "output",
    ]
    
    for json_dir in json_dirs:
        if json_dir.exists():
            for file in json_dir.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check for teams in various formats
                    if 'match' in data:
                        match_info = data['match']
                        if match_info.get('home_team'):
                            teams.add(match_info['home_team'])
                        if match_info.get('away_team'):
                            teams.add(match_info['away_team'])
                    
                    if 'home_team' in data and data['home_team']:
                        teams.add(data['home_team'])
                    if 'away_team' in data and data['away_team']:
                        teams.add(data['away_team'])
                    
                    # Check rosters/teams arrays
                    if 'teams' in data and isinstance(data['teams'], list):
                        for team_entry in data['teams']:
                            if isinstance(team_entry, dict):
                                team_name = team_entry.get('team_name', '')
                                if team_name:
                                    teams.add(team_name)
                except Exception:
                    continue
    
    # Combine: World Cup teams first, then others
    all_teams = sorted(list(teams))
    wc_team_list = sorted(list(world_cup_teams))
    
    # Put World Cup teams at the front, remove duplicates
    result = []
    seen = set()
    
    for team in wc_team_list:
        if team not in seen:
            result.append(team)
            seen.add(team)
    
    for team in all_teams:
        if team not in seen:
            result.append(team)
            seen.add(team)
    
    return result


def list_available_reports() -> List[Dict[str, Any]]:
    """List available daily reports."""
    reports = []
    
    report_dirs = [
        project_root / "output" / "daily_reports",
        project_root / "predicciones" / "output" / "daily_reports",
    ]
    
    for report_dir in report_dirs:
        if report_dir.exists():
            for file in sorted(report_dir.glob("*.md")):
                try:
                    date_str = file.stem.split('_')[0]  # e.g., "20250715"
                    reports.append({
                        "date": date_str,
                        "type": "report",
                        "path": str(file.relative_to(project_root)),
                        "size_kb": round(file.stat().st_size / 1024, 1),
                    })
                except Exception:
                    continue
    
    return reports


def _is_valid_date_format(date_str: str) -> bool:
    """
    Validate if a string is a valid date format for the pipeline.
    
    Accepts:
        - YYYYMMDD (8 digits)
        - YYYY-MM-DD (10 characters with dashes)
    
    Returns True if valid, False otherwise.
    """
    if not date_str or not isinstance(date_str, str):
        return False
    
    # Check YYYYMMDD format (8 digits)
    if len(date_str) == 8 and date_str.isdigit():
        try:
            from datetime import datetime
            datetime.strptime(date_str, "%Y%m%d")
            return True
        except ValueError:
            return False
    
    # Check YYYY-MM-DD format (10 characters)
    if len(date_str) == 10:
        try:
            from datetime import datetime
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    return False


def list_available_predictions() -> List[Dict[str, Any]]:
    """List available prediction files with valid date formats only."""
    predictions = []
    
    pred_dirs = [
        project_root / "output" / "daily_predictions",
        project_root / "predicciones" / "output" / "daily_predictions",
    ]
    
    for pred_dir in pred_dirs:
        if pred_dir.exists():
            for file in sorted(pred_dir.glob("*.csv")):
                try:
                    date_str = file.stem.split('_')[0]
                    # Only include if it's a valid date format
                    if not _is_valid_date_format(date_str):
                        continue
                    predictions.append({
                        "date": date_str,
                        "type": "predictions",
                        "path": str(file.relative_to(project_root)),
                        "size_kb": round(file.stat().st_size / 1024, 1),
                    })
                except Exception:
                    continue
    
    return predictions


def list_recent_outputs(limit: int = 10) -> List[Dict[str, Any]]:
    """List recent output files across all output directories."""
    outputs = []
    
    output_dirs = [
        project_root / "output",
        project_root / "predicciones" / "output",
    ]
    
    for output_dir in output_dirs:
        if output_dir.exists():
            for file in output_dir.rglob("*"):
                if file.is_file() and file.suffix in ['.csv', '.json', '.md']:
                    try:
                        stat = file.stat()
                        outputs.append({
                            "name": file.name,
                            "type": file.suffix[1:].upper() if file.suffix else "file",
                            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                            "size_kb": round(stat.st_size / 1024, 1),
                            "path": str(file.relative_to(project_root)),
                        })
                    except Exception:
                        continue
    
    # Sort by modification time (newest first)
    outputs.sort(key=lambda x: x["date"], reverse=True)
    
    return outputs[:limit]


def parse_date(date_str: str) -> str:
    """
    Parse and normalize date string to YYYY-MM-DD format.
    
    Accepts:
        - YYYYMMDD (e.g., 20250715)
        - YYYY-MM-DD (e.g., 2025-07-15)
    
    Returns:
        Date string in YYYY-MM-DD format.
    """
    from datetime import datetime
    
    if len(date_str) == 8 and date_str.isdigit():
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")
    
    if len(date_str) == 10:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}")
    
    raise ValueError(f"Unrecognized date format: {date_str}. Use YYYYMMDD or YYYY-MM-DD")


def get_config_sections() -> List[str]:
    """Get available configuration sections from config file."""
    from predicciones.src.utils.config_loader import config
    
    if not config:
        return []
    
    return sorted([k for k in config.keys() if isinstance(config[k], dict)])


# ================================================
# New functions for automatic fixtures and pipeline
# ================================================

def detect_available_matches(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    competition: Optional[str] = None,
    team: Optional[str] = None
) -> pd.DataFrame:
    """
    Detect available matches from data sources or existing fixture files.
    
    Args:
        from_date: Start date in YYYY-MM-DD or YYYYMMDD format
        to_date: End date in YYYY-MM-DD or YYYYMMDD format
        competition: Competition filter
        team: Team filter
        
    Returns:
        DataFrame with available matches
    """
    from datetime import datetime, timedelta
    
    # If no dates specified, default to today + next 7 days
    if not from_date:
        from_date = datetime.now().strftime("%Y-%m-%d")
    else:
        from_date = parse_date(from_date)
        
    if not to_date:
        to_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    else:
        to_date = parse_date(to_date)
        
    # Try to fetch fixtures for date range
    all_matches = []
    
    # First, check existing fixture files
    fixtures = list_available_fixtures()
    if fixtures:
        for fix in fixtures:
            try:
                fix_date = parse_date(fix.get('date', ''))
                if from_date <= fix_date <= to_date:
                    # Read the fixture file
                    fix_path = Path(fix['path'])
                    if fix_path.is_file():
                        df_fix = pd.read_csv(fix_path)
                        all_matches.append(df_fix)
            except:
                continue
                
    # If no existing fixtures, try to fetch new ones
    if not all_matches:
        try:
            from predicciones.scripts.fetch_daily_fixtures import fetch_fixtures_for_date
            current_date = datetime.strptime(from_date, "%Y-%m-%d")
            end_date = datetime.strptime(to_date, "%Y-%m-%d")
            
            while current_date <= end_date:
                try:
                    df_day = fetch_fixtures_for_date(current_date.strftime("%Y-%m-%d"))
                    if len(df_day) > 0:
                        all_matches.append(df_day)
                except:
                    pass
                current_date += timedelta(days=1)
        except:
            pass
            
    if all_matches:
        df = pd.concat(all_matches, ignore_index=True)
    else:
        # Return empty DataFrame with standard columns
        df = pd.DataFrame(columns=['match_id', 'home_team', 'away_team', 'competition', 
                                  'date', 'kickoff_datetime', 'neutral_venue'])
        
    # Apply filters
    if competition:
        comp_col = 'competition' if 'competition' in df.columns else 'league'
        if comp_col in df.columns:
            df = df[df[comp_col].str.contains(competition, case=False, na=False)]
            
    if team:
        if 'home_team' in df.columns and 'away_team' in df.columns:
            df = df[
                df['home_team'].str.contains(team, case=False, na=False) | 
                df['away_team'].str.contains(team, case=False, na=False)
            ]
            
    return df


def build_fixture_file_from_matches(
    matches_df: pd.DataFrame,
    filename: Optional[str] = None
) -> Path:
    """
    Build a fixture file from a DataFrame of matches.
    
    Args:
        matches_df: DataFrame with match data
        filename: Optional filename (without .csv)
        
    Returns:
        Path to the saved fixture file
    """
    if filename is None:
        from datetime import datetime
        filename = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    output_dir = project_root / "predicciones" / "data" / "fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{filename}.csv"
    matches_df.to_csv(output_path, index=False)
    
    return output_path


def preview_fixture_selection(matches_df: pd.DataFrame, console: Console) -> bool:
    """
    Display a preview of selected matches and ask for confirmation.
    
    Args:
        matches_df: DataFrame of matches
        console: Rich Console object
        
    Returns:
        True if user confirms, False otherwise
    """
    if len(matches_df) == 0:
        console.print("[yellow]No matches to preview![/yellow]")
        return False
        
    console.print(f"\n[bold blue]Preview of {len(matches_df)} matches:[/bold blue]")
    
    table = Table(title="Matches", show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Date", style="white")
    table.add_column("Home Team", style="cyan")
    table.add_column("Away Team", style="yellow")
    table.add_column("Competition", style="dim")
    
    for idx, row in matches_df.head(20).iterrows():
        table.add_row(
            str(idx + 1),
            row.get('date', 'N/A'),
            row.get('home_team', 'N/A'),
            row.get('away_team', 'N/A'),
            row.get('competition', row.get('league', 'N/A'))
        )
        
    console.print(table)
    
    if len(matches_df) > 20:
        console.print(f"\n[dim]... and {len(matches_df) - 20} more matches[/dim]")
        
    return Confirm.ask("\n[cyan]Do you want to proceed with these matches?[/cyan]", default=True)


def update_raw_sources(console: Console, verbose: bool = False) -> Dict[str, Any]:
    """
    Update raw data sources (fixtures, etc.).
    
    Args:
        console: Rich Console object
        verbose: Enable verbose output
        
    Returns:
        Dict with update status
    """
    console.print("\n[bold blue]Updating raw data sources...[/bold blue]")
    
    status = {
        'success': True,
        'messages': [],
        'updated_files': []
    }
    
    try:
        from predicciones.scripts.fetch_daily_fixtures import fetch_fixtures_for_date, save_fixtures
        from datetime import datetime, timedelta
        
        # Update today and next 3 days
        for i in range(4):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                df = fetch_fixtures_for_date(date)
                if len(df) > 0:
                    path = save_fixtures(df, date)
                    status['messages'].append(f"Updated fixtures for {date}: {len(df)} matches")
                    status['updated_files'].append(str(path))
                else:
                    status['messages'].append(f"No fixtures found for {date}")
            except Exception as e:
                status['success'] = False
                status['messages'].append(f"Error updating {date}: {str(e)}")
                
    except Exception as e:
        status['success'] = False
        status['messages'].append(f"Error updating raw sources: {str(e)}")
        
    for msg in status['messages']:
        if "Error" in msg:
            console.print(f"[red]{msg}[/red]")
        else:
            console.print(f"[green]✓ {msg}[/green]")
            
    return status


def rebuild_derived_datasets(console: Console, verbose: bool = False) -> Dict[str, Any]:
    """
    Rebuild derived datasets (match events, player stats, etc.).
    
    Args:
        console: Rich Console object
        verbose: Enable verbose output
        
    Returns:
        Dict with rebuild status
    """
    console.print("\n[bold blue]Rebuilding derived datasets...[/bold blue]")
    
    status = {
        'success': True,
        'messages': [],
        'updated_files': []
    }
    
    # Check if regeneration scripts exist
    scripts_to_run = []
    match_events_script = project_root / "predicciones" / "scripts" / "regenerate_match_events.py"
    player_stats_script = project_root / "predicciones" / "scripts" / "regenerate_player_match_stats.py"
    enrich_script = project_root / "predicciones" / "scripts" / "enrich_and_regenerate.py"
    
    if enrich_script.is_file():
        scripts_to_run.append(("Enrich & Regenerate", enrich_script))
    else:
        if match_events_script.is_file():
            scripts_to_run.append(("Match Events", match_events_script))
        if player_stats_script.is_file():
            scripts_to_run.append(("Player Stats", player_stats_script))
            
    if not scripts_to_run:
        status['messages'].append("No derived dataset scripts found")
        console.print("[yellow]No derived dataset scripts found[/yellow]")
        return status
        
    for name, script_path in scripts_to_run:
        try:
            console.print(f"[dim]Running {name}...[/dim]")
            
            # Use subprocess to run the script
            import subprocess
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                status['messages'].append(f"Successfully rebuilt {name}")
                console.print(f"[green]✓ Successfully rebuilt {name}[/green]")
                if verbose and result.stdout:
                    console.print(f"[dim]{result.stdout}[/dim]")
            else:
                status['success'] = False
                status['messages'].append(f"Error rebuilding {name}: {result.stderr}")
                console.print(f"[red]✗ Error rebuilding {name}[/red]")
                if verbose:
                    console.print(f"[red]{result.stderr}[/red]")
                    
        except Exception as e:
            status['success'] = False
            status['messages'].append(f"Error running {name}: {str(e)}")
            console.print(f"[red]✗ Error running {name}: {str(e)}[/red]")
            
    return status


def detect_or_generate_daily_fixtures(
    console: Console,
    date_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect or generate fixtures for today or a specific date.
    
    Args:
        console: Rich Console object
        date_str: Optional date string
        
    Returns:
        Dict with fixture status
    """
    from datetime import datetime
    
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        
    console.print(f"\n[bold blue]Detecting/generating fixtures for {date_str}...[/bold blue]")
    
    status = {
        'success': True,
        'messages': [],
        'fixture_path': None,
        'match_count': 0
    }
    
    try:
        from predicciones.scripts.fetch_daily_fixtures import fetch_fixtures_for_date, save_fixtures
        
        df = fetch_fixtures_for_date(date_str)
        
        if len(df) > 0:
            path = save_fixtures(df, date_str)
            status['fixture_path'] = str(path)
            status['match_count'] = len(df)
            status['messages'].append(f"Generated fixture file with {len(df)} matches")
            console.print(f"[green]✓ Generated fixture file with {len(df)} matches: {path}[/green]")
            
            # Show preview
            preview_fixture_selection(df, console)
        else:
            status['messages'].append(f"No fixtures found for {date_str}")
            console.print(f"[yellow]No fixtures found for {date_str}[/yellow]")
            
    except Exception as e:
        status['success'] = False
        status['messages'].append(f"Error detecting/generating fixtures: {str(e)}")
        console.print(f"[red]Error detecting/generating fixtures: {str(e)}[/red]")
        
    return status


def run_full_daily_pipeline(
    console: Console,
    date_str: Optional[str] = None,
    run_predictions: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run the full daily pipeline.
    
    Args:
        console: Rich Console object
        date_str: Optional date string
        run_predictions: Whether to run predictions after pipeline
        verbose: Enable verbose output
        
    Returns:
        Dict with pipeline status
    """
    console.print("\n[bold blue]Running full daily pipeline...[/bold blue]")
    
    pipeline_status = {
        'success': True,
        'steps': {}
    }
    
    # Step 1: Update raw sources
    pipeline_status['steps']['raw_sources'] = update_raw_sources(console, verbose)
    if not pipeline_status['steps']['raw_sources']['success']:
        pipeline_status['success'] = False
        
    # Step 2: Rebuild derived datasets
    pipeline_status['steps']['derived_datasets'] = rebuild_derived_datasets(console, verbose)
    if not pipeline_status['steps']['derived_datasets']['success']:
        pipeline_status['success'] = False
        
    # Step 3: Detect/generate daily fixtures
    pipeline_status['steps']['fixtures'] = detect_or_generate_daily_fixtures(console, date_str)
    if not pipeline_status['steps']['fixtures']['success']:
        pipeline_status['success'] = False
        
    # Step 4: Run predictions if requested
    if run_predictions and pipeline_status['steps']['fixtures']['fixture_path']:
        console.print("\n[bold blue]Running predictions...[/bold blue]")
        try:
            predict_command(
                pipeline_status['steps']['fixtures']['fixture_path'],
                "output/predictions",
                verbose,
                date_str
            )
            pipeline_status['steps']['predictions'] = {'success': True}
        except Exception as e:
            pipeline_status['steps']['predictions'] = {'success': False, 'error': str(e)}
            pipeline_status['success'] = False
            console.print(f"[red]Error running predictions: {str(e)}[/red]")
            
    console.print("\n[bold green]✓ Full pipeline completed![/bold green]")
    
    return pipeline_status


def choose_from_table(
    console: Console,
    items: List[Dict[str, Any]],
    title: str,
    columns: List[str],
    column_map: Dict[str, str],
    allow_manual: bool = True,
    manual_prompt: str = "Enter index or manual value"
) -> Optional[Any]:
    """Display items in a table and let user choose by index."""
    if not items:
        if allow_manual:
            return None
        return None
    
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right", style="cyan")
    for col in columns:
        table.add_column(column_map.get(col, col), style="white")
    
    for idx, item in enumerate(items, 1):
        row_data = [str(idx)]
        for col in columns:
            row_data.append(str(item.get(col, 'N/A')))
        table.add_row(*row_data)
    
    console.print(table)
    
    if allow_manual and len(items) > 0:
        choice = Prompt.ask(
            f"\n[cyan]{manual_prompt} (1-{len(items)}, or type value)[/cyan]"
        )
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
        except ValueError:
            # User entered a manual value
            return choice
        
        return None
    elif len(items) > 0:
        choice = IntPrompt.ask(
            f"\n[cyan]Select option (1-{len(items)})[/cyan]",
            default=1
        )
        idx = choice - 1
        if 0 <= idx < len(items):
            return items[idx]
        return None
    
    return None


def choose_from_list(
    console: Console,
    items: List[Any],
    title: str,
    allow_manual: bool = True,
    allow_multiple: bool = False
) -> Optional[Any]:
    """Display items as a numbered list and let user choose."""
    if not items:
        return None
    
    console.print(f"\n[bold]{title}[/bold]")
    for idx, item in enumerate(items, 1):
        console.print(f"  [cyan]{idx}.[/cyan] {item}")
    
    if allow_manual:
        choice = Prompt.ask(
            f"\n[cyan]Select (1-{len(items)}) or type custom value[/cyan]"
        )
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
        except ValueError:
            return choice
        
        return None
    else:
        choice = IntPrompt.ask(
            f"\n[cyan]Select option (1-{len(items)})[/cyan]",
            default=1
        )
        idx = choice - 1
        if 0 <= idx < len(items):
            return items[idx]
        return None


def predict_command(
    fixture: str,
    output_dir: str = "output/predictions",
    verbose: bool = False,
    date: Optional[str] = None,
) -> None:
    """
    Generate predictions from a fixture CSV file.
    
    Uses the pipeline.predict module for prediction logic.
    """
    console.print(f"[bold blue]Running predictions for fixture: {fixture}[/bold blue]")
    
    fixture_path = Path(fixture)
    if not fixture_path.exists():
        console.print(f"[bold red]Error: Fixture file not found: {fixture}[/bold red]")
        return
    
    try:
        # Import the prediction pipeline
        from predicciones.src.pipeline.predict import predict_match_pipeline
        from predicciones.src.ingestion.csv_loader import load_fixtures_csv
        
        # Load fixtures
        console.print("[dim]Loading fixtures...[/dim]")
        fixtures = load_fixtures_csv(fixture_path)
        
        if len(fixtures) == 0:
            console.print("[yellow]No fixtures found in file.[/yellow]")
            return
        
        console.print(f"[green]Found {len(fixtures)} fixture(s)[/green]")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        for idx, fixture_row in fixtures.iterrows():
            home_team = fixture_row.get('home_team', 'Unknown')
            away_team = fixture_row.get('away_team', 'Unknown')
            
            console.print(f"\n[bold]Processing: {home_team} vs {away_team}[/bold]")
            
            try:
                # Run prediction pipeline
                match_date = date or fixture_row.get('date', datetime.now().strftime("%Y-%m-%d"))
                
                # Extract competition context (support both 'competition' and 'league' columns)
                competition_name = fixture_row.get('competition') or fixture_row.get('league', 'International Friendly')
                competition_slug = fixture_row.get('league', None)
                
                response = predict_match_pipeline(
                    home_team=home_team,
                    away_team=away_team,
                    match_date=match_date,
                    neutral_venue=fixture_row.get('neutral_venue', False),
                    competition_name=competition_name,
                    competition_slug=competition_slug,
                    refresh_data=False,
                    api_source="auto"
                )
                results.append(response)
                
                if verbose and response:
                    predictions = response.get('predictions', {})
                    # Handle both old (p_home_win_markov) and new (1x2) formats
                    p_home = predictions.get('p_home_win_markov') or predictions.get('1x2', {}).get('home', 0) * 100
                    p_draw = predictions.get('p_draw_markov') or predictions.get('1x2', {}).get('draw', 0) * 100
                    p_away = predictions.get('p_away_win_markov') or predictions.get('1x2', {}).get('away', 0) * 100
                    console.print(f"[dim]1X2: Home {p_home:.1f}% | Draw {p_draw:.1f}% | Away {p_away:.1f}%[/dim]")
                    
            except Exception as e:
                console.print(f"[red]Error predicting {home_team} vs {away_team}: {e}[/red]")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Save results
        if results:
            import pandas as pd
            output_file = output_path / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Flatten results for CSV, including fixture metadata
            flat_results = []
            for idx, (r, fixture_row) in enumerate(zip(results, fixtures.itertuples(index=False))):
                flat = {}
                
                # Add fixture metadata first (critical for identification)
                flat['match_id'] = idx + 1
                flat['home_team'] = fixture_row.home_team if hasattr(fixture_row, 'home_team') else fixture_row[2] if len(fixture_row) > 2 else 'Unknown'
                flat['away_team'] = fixture_row.away_team if hasattr(fixture_row, 'away_team') else fixture_row[3] if len(fixture_row) > 3 else 'Unknown'
                flat['competition'] = fixture_row.league if hasattr(fixture_row, 'league') else (fixture_row[1] if len(fixture_row) > 1 else 'N/A')
                flat['date'] = fixture_row.date if hasattr(fixture_row, 'date') else (fixture_row[0] if len(fixture_row) > 0 else 'N/A')
                flat['kickoff_datetime'] = fixture_row.kickoff_datetime if hasattr(fixture_row, 'kickoff_datetime') else (fixture_row[4] if len(fixture_row) > 4 else 'N/A')
                flat['neutral_venue'] = fixture_row.neutral_venue if hasattr(fixture_row, 'neutral_venue') else (fixture_row[5] if len(fixture_row) > 5 else False)
                
                # Add response metadata if available
                if 'metadata' in r:
                    flat.update(r['metadata'])
                
                # Add predictions with proper formatting
                if 'predictions' in r:
                    preds = r['predictions']
                    # Format 1x2 as dict string
                    if '1x2' in preds and isinstance(preds['1x2'], dict):
                        flat['1x2'] = str(preds['1x2'])
                    # Format btts as dict string
                    if 'btts' in preds and isinstance(preds['btts'], dict):
                        flat['btts'] = str(preds['btts'])
                    # Format over_under as dict string
                    if 'over_under' in preds and isinstance(preds['over_under'], dict):
                        flat['over_under'] = str(preds['over_under'])
                    # Format clean_sheets as dict string
                    if 'clean_sheets' in preds and isinstance(preds['clean_sheets'], dict):
                        flat['clean_sheets'] = str(preds['clean_sheets'])
                    # Format team_totals as dict string
                    if 'team_totals' in preds and isinstance(preds['team_totals'], dict):
                        flat['team_totals'] = str(preds['team_totals'])
                    # Format correct_scores as dict string
                    if 'correct_scores' in preds and isinstance(preds['correct_scores'], dict):
                        flat['correct_scores'] = str(preds['correct_scores'])
                    # Format halftime as dict string
                    if 'halftime' in preds and isinstance(preds['halftime'], dict):
                        flat['halftime'] = str(preds['halftime'])
                    # Format home_goals_distribution as list string
                    if 'home_goals_distribution' in preds and isinstance(preds['home_goals_distribution'], list):
                        flat['home_goals_distribution'] = str(preds['home_goals_distribution'])
                    # Format away_goals_distribution as list string
                    if 'away_goals_distribution' in preds and isinstance(preds['away_goals_distribution'], list):
                        flat['away_goals_distribution'] = str(preds['away_goals_distribution'])
                    # Add expected_goals
                    if 'expected_goals' in preds:
                        flat['expected_goals'] = str(preds['expected_goals'])
                    # Add sanity_flags
                    if 'sanity_flags' in preds:
                        flat['sanity_flags'] = str(preds['sanity_flags'])
                
                flat_results.append(flat)
            
            df_results = pd.DataFrame(flat_results)
            
            # Ensure column order: metadata first, then predictions
            base_columns = ['match_id', 'home_team', 'away_team', 'competition', 'date', 'kickoff_datetime', 'neutral_venue']
            prediction_columns = ['1x2', 'btts', 'over_under', 'clean_sheets', 'team_totals', 'correct_scores', 'halftime', 'home_goals_distribution', 'away_goals_distribution', 'expected_goals', 'sanity_flags']
            
            # Get all columns, ensuring base columns come first
            all_cols = base_columns + [c for c in prediction_columns if c in df_results.columns] + [c for c in df_results.columns if c not in base_columns and c not in prediction_columns]
            
            # Only keep columns that exist
            final_columns = [c for c in all_cols if c in df_results.columns]
            
            df_results = df_results[final_columns]
            df_results.to_csv(output_file, index=False)
            console.print(f"\n[green]✓ Predictions saved to {output_file}[/green]")
        else:
            console.print("[yellow]No predictions generated.[/yellow]")
            
    except ImportError as e:
        console.print(f"[bold red]Error importing prediction module: {e}[/bold red]")
        console.print("[dim]Make sure all dependencies are installed.[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error running predictions: {e}[/bold red]")
        if verbose:
            import traceback
            traceback.print_exc()


def pipeline_command(
    date: str,
    output_dir: str = "output/daily",
    verbose: bool = False,
    skip_validation: bool = False,
) -> None:
    """
    Run the complete daily prediction pipeline.
    
    Wraps the run_daily_pipeline script functionality.
    """
    console.print(f"[bold blue]Running daily pipeline for date: {date}[/bold blue]")
    
    try:
        from predicciones.scripts.run_daily_pipeline import run_daily_pipeline
        
        outputs = run_daily_pipeline(
            date_str=date,
            config=None,
            verbose=verbose
        )
        
        # Handle different status outcomes
        status = outputs.get('status', 'unknown')
        
        if status == 'no_fixtures':
            console.print("\n[bold yellow]⚠️  PIPELINE ABORTED: No fixtures found[/bold yellow]")
            console.print(f"[yellow]{outputs.get('message', 'No fixtures found for selected date')}[/yellow]")
            console.print(f"\n[dim]Fixtures file (empty): {outputs['fixtures']}[/dim]")
            console.print("\n[bold]Reason:[/bold] Missing API keys or no fixtures returned from APIs")
            console.print("\n[bold]To fix this:[/bold]")
            console.print("  1. Set FOOTBALL_DATA_TOKEN environment variable, OR")
            console.print("  2. Set API_FOOTBALL_KEY environment variable, OR")
            console.print("  3. Use an existing dated fixture file with matches")
            
        elif status == 'no_predictions':
            console.print("\n[bold yellow]⚠️  PIPELINE COMPLETED WITH NO PREDICTIONS[/bold yellow]")
            console.print(f"[yellow]{outputs.get('message', 'No predictions could be generated')}[/yellow]")
            console.print(f"\n📋 Fixtures found: [green]{outputs.get('fixtures_count', 0)}[/green]")
            console.print(f"📊 Predictions:   [yellow]{outputs.get('predictions_count', 0)}[/yellow]")
            console.print(f"\n[dim]Fixtures file: {outputs['fixtures']}[/dim]")
            console.print(f"[dim]Predictions file (empty): {outputs['predictions']}[/dim]")
            
        else:
            console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")
            console.print(f"\nGenerated files:")
            console.print(f"  📋 Fixtures:      {outputs['fixtures']} ({outputs.get('fixtures_count', 0)} matches)")
            console.print(f"  📊 Predictions:   {outputs['predictions']} ({outputs.get('predictions_count', 0)} matches)")
            console.print(f"  📝 Report (MD):   {outputs['report_md']}")
            console.print(f"  📈 Summary (CSV): {outputs['report_csv']}")
        
    except ImportError:
        # Fallback: manual pipeline execution
        console.print("[dim]Running pipeline manually...[/dim]")
        
        # Step 1: Fetch fixtures
        console.print("\n[bold]Step 1: Fetching fixtures[/bold]")
        try:
            from predicciones.scripts.fetch_daily_fixtures import fetch_fixtures_for_date, save_fixtures
            import pandas as pd
            
            df = fetch_fixtures_for_date(date)
            if len(df) == 0:
                console.print("[yellow]No fixtures found for this date.[/yellow]")
                console.print("\n[bold red]Pipeline aborted - no fixtures to process.[/bold red]")
                return
            
            fixture_path = save_fixtures(df, date)
            console.print(f"[green]✓ Saved {len(df)} fixtures to {fixture_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error fetching fixtures: {e}[/red]")
            return
        
        # Step 2: Run predictions would go here
        console.print("\n[bold]Step 2: Running predictions[/bold]")
        console.print("[dim]Predictions would be generated here.[/dim]")
        
        # Step 3: Generate report
        console.print("\n[bold]Step 3: Generating report[/bold]")
        try:
            from predicciones.scripts.generate_daily_report import generate_daily_report
            
            # Find predictions file
            normalized_date = date.replace('-', '')
            if len(normalized_date) == 8:
                predictions_path = project_root / "output" / "daily_predictions" / f"{normalized_date}_predictions.csv"
            else:
                console.print("[yellow]Could not locate predictions file.[/yellow]")
                return
            
            if predictions_path.exists():
                md_path, csv_path = generate_daily_report(predictions_path, date)
                console.print(f"[green]✓ Report generated: {md_path}[/green]")
            else:
                console.print("[yellow]Predictions file not found. Skipping report generation.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error generating report: {e}[/red]")
            
    except Exception as e:
        console.print(f"[bold red]Error running pipeline: {e}[/bold red]")
        if verbose:
            import traceback
            traceback.print_exc()


def players_command(
    team: str,
    max_matches: int = 10,
    output_format: str = "table",
    mode: str = "summary",  # "summary", "cards", "timeline"
) -> None:
    """
    Get player statistics for a selected team with multiple submodes.
    
    Uses the espn_player_stats module which reads from JSONL derived data.
    Normalizes team name to handle Spanish/English aliases.
    
    Submodes:
    - summary: Accumulated stats (GP, Min, Gls, Ast, YC, RC, Shots)
    - cards: Historical cards per player  
    - timeline: Chronological card timeline by match
    
    Args:
        team: Team name (Spanish or English accepted)
        max_matches: Maximum matches to consider
        output_format: Output format ("table", "csv", "json")
        mode: One of "summary", "cards", "timeline"
    """
    # Normalize team name for display
    from predicciones.src.utils.team_normalization import normalize_team_name
    canonical_team = normalize_team_name(team)
    
    console.print(f"[bold blue]Player Statistics - {canonical_team}[/bold blue]")
    console.print(f"[dim]Mode: {mode} | Max matches: {max_matches}[/dim]")
    
    try:
        from predicciones.scripts.espn_player_stats import (
            fetch_extended_player_stats,
            fetch_team_roster_stats,
            format_output_table,
            format_output_csv,
            format_output_json,
            format_card_timeline_table,
            format_card_timeline_csv,
        )
        
        # Fetch data based on mode
        if mode == "summary":
            console.print("[dim]Fetching accumulated roster stats...[/dim]")
            player_stats = fetch_extended_player_stats(canonical_team, mode="summary")
            display_mode = "roster"
            
            # Show matches used info - read directly from player_match_stats.jsonl
            if player_stats:
                from pathlib import Path
                import json
                
                # Resolve path relative to project root (workspace)
                # Strategy: go up from commands.py to find data/derived
                current_file = Path(__file__).resolve()
                # Navigate: /workspace/predicciones/src/cli/commands.py -> /workspace
                project_root = current_file.parent.parent.parent.parent
                player_stats_path = project_root / "data" / "derived" / "player_match_stats.jsonl"
                
                # Debug info
                console.print(f"[dim]Player stats path: {player_stats_path} (exists={player_stats_path.exists()})[/dim]")
                
                # Get unique matches for this team from player_match_stats.jsonl
                matches_used = []
                competitions = set()
                league_slugs = set()
                if player_stats_path.exists():
                    # Convert canonical_team (Spanish display name) to JSONL team name (English)
                    from predicciones.src.utils.team_normalization import get_jsonl_team_name
                    jsonl_team = get_jsonl_team_name(canonical_team)
                    
                    with open(player_stats_path, 'r', encoding='utf-8') as f:
                        seen_events = set()
                        for line in f:
                            record = json.loads(line)
                            if record.get("team", "").lower() == jsonl_team.lower():
                                eid = record.get("event_id")
                                if eid and eid not in seen_events:
                                    seen_events.add(eid)
                                    comp = record.get("competition", "Unknown")
                                    league = record.get("league_slug", "")
                                    matches_used.append({
                                        "event_id": eid,
                                        "date": record.get("date", ""),
                                        "competition": comp,
                                        "league_slug": league,
                                        "opponent": record.get("opponent", ""),
                                        "home_or_away": record.get("home_or_away", ""),
                                    })
                                    if comp and comp != "Unknown":
                                        competitions.add(comp)
                                    if league:
                                        league_slugs.add(league)
                
                # Display matches used
                if matches_used:
                    console.print("\n[bold]Matches used:[/bold]")
                    for m in sorted(matches_used, key=lambda x: x.get("date", "")):
                        date_str = (m.get("date", "") or "")[:10]
                        comp = m.get("competition", "Unknown")
                        league = m.get("league_slug", "")
                        opponent = m.get("opponent", "")
                        hoa = m.get("home_or_away", "")
                        home_display = "(H)" if hoa == "home" else "(A)" if hoa == "away" else ""
                        
                        if opponent:
                            console.print(f"  [dim]{m.get('event_id')} | {date_str} | {comp} | {league} | {canonical_team} vs {opponent} {home_display}[/dim]")
                        else:
                            console.print(f"  [dim]{m.get('event_id')} | {date_str} | {comp} | {league}[/dim]")
                    
                    if competitions:
                        console.print(f"\n[bold]Competitions included:[/bold] {', '.join(sorted(competitions))}")
                    if league_slugs:
                        console.print(f"[bold]League slugs:[/bold] {', '.join(sorted(league_slugs))}")
                    
        elif mode == "cards":
            console.print("[dim]Fetching card history per player...[/dim]")
            player_stats = fetch_extended_player_stats(canonical_team, mode="cards", max_matches=max_matches)
            display_mode = "cards"
        elif mode == "timeline":
            console.print(f"[dim]Fetching chronological card timeline (max {max_matches} matches)...[/dim]")
            player_stats = fetch_extended_player_stats(canonical_team, mode="timeline", max_matches=max_matches)
            display_mode = "timeline"
        else:
            console.print(f"[yellow]Unknown mode: {mode}. Using 'summary'.[/yellow]")
            player_stats = fetch_extended_player_stats(canonical_team, mode="summary")
            display_mode = "roster"
        
        if not player_stats:
            console.print(f"[yellow]No data found for team: {team} (mode: {mode})[/yellow]")
            return
        
        # Log stats summary
        console.print(f"[green]✓ Found {len(player_stats)} records[/green]")
        
        # Display based on format
        if output_format == "table":
            if display_mode == "timeline":
                table_str = format_card_timeline_table(player_stats)
            else:
                table_str = format_output_table(player_stats, mode=display_mode)
            console.print(table_str)
        elif output_format == "csv":
            if display_mode == "timeline":
                csv_str = format_card_timeline_csv(player_stats)
            else:
                csv_str = format_output_csv(player_stats, mode=display_mode)
            # Save CSV to file
            from pathlib import Path
            output_dir = Path("output/player_stats")
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{canonical_team.replace(' ', '_')}_{mode}_{max_matches}.csv"
            output_path = output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(csv_str)
            console.print(f"[green]✓ Player stats saved to {output_path}[/green]")
        elif output_format == "json":
            json_str = format_output_json(player_stats)
            console.print(json_str)
        else:
            console.print(f"[yellow]Unknown output format: {output_format}[/yellow]")
            
    except ImportError as e:
        console.print(f"[bold red]Error: Player stats module not available: {e}[/bold red]")
        console.print("[dim]This feature requires ESPN-derived JSONL data.[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error fetching player stats: {e}[/bold red]")
        import traceback
        if console.is_terminal:
            traceback.print_exc()


def _display_player_stats_table(player_stats: list) -> None:
    """Display player stats in a Rich table."""
    if not player_stats:
        console.print("[yellow]No player statistics available.[/yellow]")
        return
    
    table = Table(title="Player Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Player", style="cyan")
    table.add_column("Position", justify="center")
    table.add_column("Matches", justify="right")
    table.add_column("Goals", justify="right")
    table.add_column("Assists", justify="right")
    table.add_column("Tackles", justify="right")
    table.add_column("Interceptions", justify="right")
    
    for player in player_stats:
        table.add_row(
            player.get('name', 'N/A'),
            player.get('position', 'N/A'),
            str(player.get('matches', 0)),
            str(player.get('goals', 0)),
            str(player.get('assists', 0)),
            str(player.get('tackles', 0)),
            str(player.get('interceptions', 0)),
        )
    
    console.print(table)


def _save_player_stats_csv(player_stats: list, team: str) -> None:
    """Save player stats to CSV."""
    import pandas as pd
    
    output_dir = Path("output/player_stats")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(player_stats)
    filename = f"{team.replace(' ', '_')}_players.csv"
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    console.print(f"[green]✓ Player stats saved to {output_path}[/green]")


def _display_player_stats_json(player_stats: list) -> None:
    """Display player stats as JSON."""
    import json
    console.print(json.dumps(player_stats, indent=2))


def timelines_command(
    match_id: Optional[str] = None,
    output_dir: str = "output/timelines",
) -> None:
    """
    Get timeline data for past matches.

    Without match_id, opens the guided team -> match -> timeline browser.
    """
    if not match_id:
        from predicciones.src.cli.match_timeline import run_match_timeline_menu

        run_match_timeline_menu(console)
        return

    console.print(f"[bold blue]Fetching timeline for match: {match_id}[/bold blue]")
    
    try:
        from predicciones.scripts.match_timeline import fetch_match_timeline
        
        timeline_data = fetch_match_timeline(match_id)
        
        if not timeline_data:
            console.print("[yellow]No timeline data found for this match.[/yellow]")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save timeline
        import json
        output_file = output_path / f"timeline_{match_id}.json"
        with open(output_file, 'w') as f:
            json.dump(timeline_data, f, indent=2)
        
        console.print(f"[green]✓ Timeline saved to {output_file}[/green]")
        
        # Display summary
        events = timeline_data.get('events', [])
        console.print(f"\n[bold]Timeline Summary:[/bold] {len(events)} events")
        
        if events:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Minute", justify="right")
            table.add_column("Type", style="cyan")
            table.add_column("Description")
            
            for event in events[:10]:  # Show first 10 events
                table.add_row(
                    str(event.get('minute', '?')),
                    event.get('type', 'N/A'),
                    event.get('description', '')[:50],
                )
            
            console.print(table)
            if len(events) > 10:
                console.print(f"[dim]... and {len(events) - 10} more events[/dim]")
                
    except ImportError as e:
        console.print(f"[bold red]Error: Timeline module not available: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error fetching timeline: {e}[/bold red]")


def fixtures_command(
    date: str,
    competition: Optional[str] = None,
    output_format: str = "table",
) -> None:
    """
    Get fixtures for a specific date.
    
    Uses the fetch_daily_fixtures script functionality.
    """
    console.print(f"[bold blue]Fetching fixtures for date: {date}[/bold blue]")
    if competition:
        console.print(f"[dim]Filtering by competition: {competition}[/dim]")
    
    try:
        from predicciones.scripts.fetch_daily_fixtures import fetch_fixtures_for_date
        
        df = fetch_fixtures_for_date(date)
        
        if competition:
            df = df[df.get('competition', '').str.contains(competition, case=False, na=False)]
        
        if len(df) == 0:
            console.print("[yellow]No fixtures found for this date.[/yellow]")
            return
        
        if output_format == "table":
            _display_fixtures_table(df)
        elif output_format == "csv":
            _save_fixtures_csv(df, date)
        else:
            console.print(f"[yellow]Unknown output format: {output_format}[/yellow]")
            
    except ImportError as e:
        console.print(f"[bold red]Error: Fixtures module not available: {e}[/bold red]")
        console.print("[dim]Make sure API keys are configured if using live data.[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error fetching fixtures: {e}[/bold red]")


def _display_fixtures_table(df) -> None:
    """Display fixtures in a Rich table."""
    table = Table(title=f"Fixtures ({len(df)} matches)", show_header=True, header_style="bold magenta")
    table.add_column("Time", justify="center")
    table.add_column("Home Team", style="cyan")
    table.add_column("Away Team", style="yellow")
    table.add_column("Competition", style="dim")
    
    for _, row in df.iterrows():
        table.add_row(
            row.get('time', 'N/A'),
            row.get('home_team', 'N/A'),
            row.get('away_team', 'N/A'),
            row.get('competition', 'N/A'),
        )
    
    console.print(table)


def _save_fixtures_csv(df, date: str) -> None:
    """Save fixtures to CSV."""
    output_dir = Path("data/fixtures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{date.replace('-', '')}.csv"
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    console.print(f"[green]✓ Fixtures saved to {output_path}[/green]")


def daily_report_command(
    date: str,
    output_dir: str = "output/reports",
    include_analysis: bool = True,
) -> None:
    """
    Generate daily prediction report.
    
    Uses the generate_daily_report script functionality.
    """
    console.print(f"[bold blue]Generating daily report for: {date}[/bold blue]")
    
    try:
        from predicciones.scripts.generate_daily_report import generate_daily_report
        
        # Find predictions file
        normalized_date = date.replace('-', '')
        if len(normalized_date) == 8:
            predictions_path = project_root / "output" / "daily_predictions" / f"{normalized_date}_predictions.csv"
        else:
            predictions_path = Path(date)  # Assume direct path
        
        if not predictions_path.exists():
            console.print(f"[yellow]Predictions file not found: {predictions_path}[/yellow]")
            console.print("[dim]Run the pipeline first to generate predictions.[/dim]")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        md_path, csv_path = generate_daily_report(predictions_path, date, output_path)
        
        console.print(f"\n[green]✓ Report generated successfully![/green]")
        console.print(f"  📝 Markdown: {md_path}")
        console.print(f"  📊 Summary:  {csv_path}")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error generating report: {e}[/bold red]")


def lambda_analysis_command(
    num_matches: int = 50,
    output_dir: str = "output/lambda_validation",
    threshold_home: float = 3.0,
    threshold_away: float = 2.5,
    threshold_total: float = 5.0,
) -> None:
    """
    Analyze lambda distribution across matches.
    
    Uses the analyze_lambda_distribution script functionality.
    """
    console.print(f"[bold blue]Analyzing lambda distribution across {num_matches} matches[/bold blue]")
    console.print(f"[dim]Thresholds: Home > {threshold_home}, Away > {threshold_away}, Total > {threshold_total}[/dim]")
    
    try:
        from predicciones.scripts.analyze_lambda_distribution import LambdaDistributionAnalyzer
        
        analyzer = LambdaDistributionAnalyzer()
        
        # Generate test matches
        from predicciones.scripts.analyze_lambda_distribution import generate_test_matches
        matches = generate_test_matches(n_matches=num_matches)
        
        console.print("[dim]Running predictions...[/dim]")
        results = []
        for match in matches:
            result = analyzer.predict_match(match)
            results.append(result)
        
        # Analyze distribution
        console.print("[dim]Analyzing distribution...[/dim]")
        metrics = analyzer.analyze_distribution(results)
        
        # Save results
        analyzer.save_results(results, metrics)
        
        # Display summary
        console.print("\n[bold green]✓ Analysis complete![/bold green]")
        console.print(f"\n[bold]Lambda Distribution Summary:[/bold]")
        console.print(f"  λ_home mean:   {metrics['lambda_home_mean']:.4f}")
        console.print(f"  λ_home median: {metrics['lambda_home_median']:.4f}")
        console.print(f"  λ_away mean:   {metrics['lambda_away_mean']:.4f}")
        console.print(f"  λ_away median: {metrics['lambda_away_median']:.4f}")
        console.print(f"  λ_total mean:  {metrics['lambda_total_mean']:.4f}")
        console.print(f"  λ_total median:{metrics['lambda_total_median']:.4f}")
        
        console.print(f"\n[bold]Threshold Exceedances:[/bold]")
        console.print(f"  Home > {threshold_home}: {metrics['n_home_above_threshold']} ({metrics['pct_home_above_threshold']:.1f}%)")
        console.print(f"  Away > {threshold_away}: {metrics['n_away_above_threshold']} ({metrics['pct_away_above_threshold']:.1f}%)")
        console.print(f"  Total > {threshold_total}: {metrics['n_total_above_threshold']} ({metrics['pct_total_above_threshold']:.1f}%)")
        
        # Generate report
        report_content = analyzer.generate_report(results, metrics)
        report_path = Path(output_dir) / "report.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        console.print(f"\n[green]✓ Full report saved to {report_path}[/green]")
        
    except ImportError as e:
        console.print(f"[bold red]Error: Lambda analysis module not available: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error running lambda analysis: {e}[/bold red]")


def backtest_command(
    num_matches: int = 200,
    output_dir: str = "output/calibration_eval",
    compare_markov: bool = True,
) -> None:
    """
    Run backtest/calibration evaluation.
    
    Uses the backtest_temporal_calibration_v2 script functionality.
    """
    console.print(f"[bold blue]Running backtest with {num_matches} matches[/bold blue]")
    if compare_markov:
        console.print("[dim]Comparing baseline vs Markov-aware models[/dim]")
    else:
        console.print("[dim]Using baseline model only[/dim]")
    
    try:
        from predicciones.scripts.backtest_temporal_calibration_v2 import (
            run_backtest,
            BASELINE_CONFIG,
            MARKOV_AWARE_CONFIG,
        )
        
        configs = [BASELINE_CONFIG]
        if compare_markov:
            configs.append(MARKOV_AWARE_CONFIG)
        
        console.print("[dim]Running backtest...[/dim]")
        results = run_backtest(
            model_configs=configs,
            n_matches=num_matches,
            output_dir=Path(output_dir),
        )
        
        console.print("\n[bold green]✓ Backtest complete![/bold green]")
        
        # Display summary
        if 'summary' in results:
            console.print(f"\n[bold]Results Summary:[/bold]")
            for metric, value in results['summary'].items():
                console.print(f"  {metric}: {value}")
        
        console.print(f"\n[green]✓ Results saved to {output_dir}[/green]")
        
    except ImportError as e:
        console.print(f"[bold red]Error: Backtest module not available: {e}[/bold red]")
        console.print("[dim]Make sure all evaluation dependencies are installed.[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error running backtest: {e}[/bold red]")


def recent_command(
    limit: int = 10,
    file_type: str = "all",
    show_summary: bool = False,
) -> None:
    """
    View recently generated files and reports.
    """
    console.print(f"[bold blue]Showing {limit} recent files[/bold blue]")
    if file_type != "all":
        console.print(f"[dim]Filtering by type: {file_type}[/dim]")
    
    # Define output directories to search
    output_dirs = {
        'predictions': project_root / "output" / "predictions",
        'reports': project_root / "output" / "daily_reports",
        'daily': project_root / "output" / "daily",
        'lambda': project_root / "output" / "lambda_validation",
        'backtest': project_root / "output" / "calibration_eval",
    }
    
    # Collect recent files
    recent_files = []
    
    for dir_name, dir_path in output_dirs.items():
        if not dir_path.exists():
            continue
        
        # Filter by type if specified
        if file_type != "all" and file_type not in dir_name:
            continue
        
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                recent_files.append({
                    'path': file_path,
                    'type': dir_name,
                    'mtime': file_path.stat().st_mtime,
                })
    
    # Sort by modification time (newest first)
    recent_files.sort(key=lambda x: x['mtime'], reverse=True)
    recent_files = recent_files[:limit]
    
    if not recent_files:
        console.print("[yellow]No recent files found.[/yellow]")
        return
    
    # Display table
    table = Table(title=f"Recent Files ({len(recent_files)} items)", show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan", justify="center")
    table.add_column("Filename", style="white")
    table.add_column("Modified", style="dim")
    
    for file_info in recent_files:
        file_path = file_info['path']
        mtime = datetime.fromtimestamp(file_info['mtime']).strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            file_info['type'],
            file_path.name,
            mtime,
        )
    
    console.print(table)
    
    # Show summary of latest file if requested
    if show_summary and recent_files:
        latest = recent_files[0]
        console.print(f"\n[bold]Latest File Summary:[/bold] {latest['path']}")
        
        try:
            if latest['path'].suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(latest['path'], nrows=5)
                console.print(f"[dim]Columns: {', '.join(df.columns.tolist())}[/dim]")
                console.print(f"[dim]Rows: {len(pd.read_csv(latest['path']))} total[/dim]")
            elif latest['path'].suffix == '.json':
                import json
                with open(latest['path'], 'r') as f:
                    data = json.load(f)
                console.print(f"[dim]Keys: {', '.join(str(k) for k in data.keys())}[/dim]")
            elif latest['path'].suffix == '.md':
                with open(latest['path'], 'r') as f:
                    lines = f.readlines()[:10]
                console.print("[dim]First lines:[/dim]")
                for line in lines:
                    console.print(f"  {line.rstrip()}")
        except Exception as e:
            console.print(f"[dim]Could not read file: {e}[/dim]")


def config_command(
    show_all: bool = False,
    section: Optional[str] = None,
    edit: bool = False,
) -> None:
    """
    View or edit system configuration.
    """
    console.print("[bold blue]System Configuration[/bold blue]")
    
    try:
        from predicciones.src.utils.config_loader import config
        
        if not config:
            console.print("[yellow]No configuration file found.[/yellow]")
            console.print("[dim]Create a config file at: configs/model_config.yaml[/dim]")
            return
        
        if edit:
            console.print("[dim]Opening config file in editor...[/dim]")
            import subprocess
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "model_config.yaml"
            if config_path.exists():
                subprocess.call([os.environ.get('EDITOR', 'nano'), str(config_path)])
            else:
                console.print("[yellow]Config file not found. Creating default...[/yellow]")
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    f.write("# Football Predictor Configuration\n")
                console.print(f"[green]Created: {config_path}[/green]")
            return
        
        if section:
            if section in config:
                console.print(f"\n[bold]{section}:[/bold]")
                _display_config_section(config[section])
            else:
                console.print(f"[yellow]Section '{section}' not found in config.[/yellow]")
                console.print(f"[dim]Available sections: {', '.join(config.keys())}[/dim]")
        elif show_all:
            console.print("\n[bold]Full Configuration:[/bold]")
            _display_config_full(config)
        else:
            # Show summary
            console.print("\n[bold]Configuration Summary:[/bold]")
            console.print(f"  Sections: {', '.join(config.keys())}")
            console.print(f"\n[dim]Use --all to show full config or --section <name> for specific section[/dim]")
        
    except ImportError as e:
        console.print(f"[bold red]Error loading config: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


def _display_config_section(section_data: dict, indent: int = 0) -> None:
    """Display a config section."""
    prefix = "  " * indent
    for key, value in section_data.items():
        if isinstance(value, dict):
            console.print(f"{prefix}[bold]{key}:[/bold]")
            _display_config_section(value, indent + 1)
        else:
            console.print(f"{prefix}{key}: {value}")


def _display_config_full(config: dict, indent: int = 0) -> None:
    """Display full config recursively."""
    _display_config_section(config, indent)
