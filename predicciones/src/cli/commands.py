"""
CLI Commands for Football Prediction System

This module implements all command functions used by app.py.
Each function wraps existing project functionality.
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


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
                
                response = predict_match_pipeline(
                    home_team=home_team,
                    away_team=away_team,
                    match_date=match_date,
                    neutral_venue=fixture_row.get('neutral_venue', False),
                    refresh_data=False,
                    api_source="auto"
                )
                results.append(response)
                
                if verbose and response:
                    predictions = response.get('predictions', {})
                    p_home = predictions.get('p_home_win_markov', 0) * 100
                    p_draw = predictions.get('p_draw_markov', 0) * 100
                    p_away = predictions.get('p_away_win_markov', 0) * 100
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
            
            # Flatten results for CSV
            flat_results = []
            for r in results:
                flat = {}
                if 'metadata' in r:
                    flat.update(r['metadata'])
                if 'predictions' in r:
                    flat.update(r['predictions'])
                flat_results.append(flat)
            
            df_results = pd.DataFrame(flat_results)
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
        
        console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")
        console.print(f"\nGenerated files:")
        console.print(f"  📋 Fixtures:      {outputs['fixtures']}")
        console.print(f"  📊 Predictions:   {outputs['predictions']}")
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
) -> None:
    """
    Get player statistics for a selected team.
    
    Uses the match_player_defensive_stats script logic.
    """
    console.print(f"[bold blue]Fetching player stats for: {team}[/bold blue]")
    
    try:
        from predicciones.scripts.match_player_defensive_stats import (
            fetch_team_players,
            compute_player_stats,
        )
        
        # Fetch team data
        console.print("[dim]Fetching team data...[/dim]")
        team_data = fetch_team_players(team, max_matches=max_matches)
        
        if not team_data:
            console.print(f"[yellow]No data found for team: {team}[/yellow]")
            return
        
        # Compute player stats
        console.print("[dim]Computing player statistics...[/dim]")
        player_stats = compute_player_stats(team_data)
        
        if output_format == "table":
            _display_player_stats_table(player_stats)
        elif output_format == "csv":
            _save_player_stats_csv(player_stats, team)
        elif output_format == "json":
            _display_player_stats_json(player_stats)
        else:
            console.print(f"[yellow]Unknown output format: {output_format}[/yellow]")
            
    except ImportError as e:
        console.print(f"[bold red]Error: Player stats module not available: {e}[/bold red]")
        console.print("[dim]This feature requires ESPN API integration.[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error fetching player stats: {e}[/bold red]")


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
    match_id: str,
    output_dir: str = "output/timelines",
) -> None:
    """
    Get timeline data for past matches.
    
    Uses the match_timeline script functionality.
    """
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
