"""
Interactive Menu for Football Prediction System

Provides a text-based interactive menu interface using Rich.
Implements guided selection with data discovery before asking for manual input.
"""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.text import Text
from rich.table import Table

console = Console()


class InteractiveMenu:
    """Interactive CLI menu for the Football Prediction System."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.running = True
        
    def run(self) -> None:
        """Main menu loop."""
        while self.running:
            self._show_main_menu()
            
    def _show_main_menu(self) -> None:
        """Display the main menu and handle user input."""
        self.console.print()
        self.console.print(Panel(
            "[bold]Main Menu[/bold]\n"
            "Select an option:\n\n"
            "  [cyan]1.[/cyan] Predict (from fixture file)\n"
            "  [cyan]2.[/cyan] Run Daily Pipeline\n"
            "  [cyan]3.[/cyan] Player Statistics\n"
            "  [cyan]4.[/cyan] Match Timeline\n"
            "  [cyan]5.[/cyan] View Fixtures\n"
            "  [cyan]6.[/cyan] Generate Daily Report\n"
            "  [cyan]7.[/cyan] Lambda Analysis\n"
            "  [cyan]8.[/cyan] Backtest/Calibration\n"
            "  [cyan]9.[/cyan] Recent Files\n"
            "  [cyan]10.[/cyan] Configuration\n"
            "  [cyan]0.[/cyan] Exit\n",
            title="⚽ Football Prediction System",
            border_style="blue",
        ))
        
        choice = Prompt.ask("\n[cyan]Enter your choice[/cyan]", choices=[str(i) for i in range(11)], default="0")
        
        if choice == "0":
            self._exit_menu()
        elif choice == "1":
            self._predict_menu()
        elif choice == "2":
            self._pipeline_menu()
        elif choice == "3":
            self._players_menu()
        elif choice == "4":
            self._timelines_menu()
        elif choice == "5":
            self._fixtures_menu()
        elif choice == "6":
            self._daily_report_menu()
        elif choice == "7":
            self._lambda_analysis_menu()
        elif choice == "8":
            self._backtest_menu()
        elif choice == "9":
            self._recent_files_menu()
        elif choice == "10":
            self._config_menu()
        else:
            self.console.print("[yellow]Invalid choice. Please try again.[/yellow]")
    
    def _exit_menu(self) -> None:
        """Exit the interactive menu."""
        if Confirm.ask("\n[cyan]Are you sure you want to exit?[/cyan]", default=False):
            self.running = False
            self.console.print("\n[bold green]Goodbye![/bold green]")
    
    def _predict_menu(self) -> None:
        """New Predict menu with auto-generated and existing fixture options."""
        from predicciones.src.cli.commands import (
            list_available_fixtures,
            detect_available_matches,
            build_fixture_file_from_matches,
            preview_fixture_selection,
            predict_command,
        )
        from datetime import datetime
        
        while True:
            self.console.print()
            self.console.print(Panel(
                "[bold]Predict Menu[/bold]\n"
                "1. Predict from auto-generated fixtures\n"
                "2. Predict from existing fixture file\n"
                "B. Back\n"
                "Q. Exit",
                title="⚽ Predict",
                border_style="blue"
            ))
            
            choice = Prompt.ask("\n[cyan]Enter your choice[/cyan]", default="1").strip().lower()
            
            if choice == 'q':
                self._exit_menu()
                return
            elif choice == 'b':
                return
            elif choice == '1':
                # Auto-generated fixtures
                self._predict_auto_generated()
            elif choice == '2':
                # Existing fixture file
                self._predict_existing()
            else:
                self.console.print("[yellow]Invalid choice. Please try again.[/yellow]")

    def _predict_auto_generated(self) -> None:
        """Submenu for auto-generated fixture options."""
        from predicciones.src.cli.commands import (
            detect_available_matches,
            build_fixture_file_from_matches,
            preview_fixture_selection,
            predict_command,
        )
        from datetime import datetime, timedelta

        while True:
            self.console.print()
            self.console.print(Panel(
                "[bold]Auto-generated Fixtures[/bold]\n"
                "1. Today's fixtures\n"
                "2. Upcoming fixtures (next 7 days)\n"
                "3. Fixtures by competition\n"
                "4. Fixtures by team\n"
                "5. All available fixtures\n"
                "B. Back\n"
                "Q. Exit",
                title="⚽ Auto Fixtures",
                border_style="blue"
            ))
            
            choice = Prompt.ask("\n[cyan]Enter your choice[/cyan]", default="1").strip().lower()
            
            if choice == 'q':
                self._exit_menu()
                return
            elif choice == 'b':
                return
            elif choice == '1':
                # Today's fixtures
                today = datetime.now().strftime("%Y-%m-%d")
                df = detect_available_matches(from_date=today, to_date=today)
            elif choice == '2':
                # Upcoming fixtures
                today = datetime.now().strftime("%Y-%m-%d")
                next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
                df = detect_available_matches(from_date=today, to_date=next_week)
            elif choice == '3':
                # By competition
                competition = Prompt.ask("[cyan]Enter competition name[/cyan]")
                df = detect_available_matches(competition=competition)
            elif choice == '4':
                # By team
                team = Prompt.ask("[cyan]Enter team name[/cyan]")
                df = detect_available_matches(team=team)
            elif choice == '5':
                # All available fixtures
                df = detect_available_matches()
            else:
                self.console.print("[yellow]Invalid choice. Please try again.[/yellow]")
                continue

            # Preview and confirm
            if len(df) == 0:
                self.console.print("[yellow]No matches found for this selection![/yellow]")
                continue

            if preview_fixture_selection(df, self.console):
                # Build fixture file
                fixture_path = build_fixture_file_from_matches(df)
                self.console.print(f"[green]✓ Fixture file created at {fixture_path}[/green]")
                
                # Run predictions
                output_dir = Prompt.ask(
                    "[cyan]Output directory[/cyan]",
                    default="output/predictions"
                )
                verbose = Confirm.ask(
                    "[cyan]Enable verbose output?[/cyan]",
                    default=False
                )
                
                predict_command(str(fixture_path), output_dir, verbose)
                return

    def _predict_existing(self) -> None:
        """Submenu for selecting existing fixture file."""
        from predicciones.src.cli.commands import (
            list_available_fixtures,
            preview_fixture_selection,
            predict_command,
        )
        import pandas as pd

        while True:
            fixtures = list_available_fixtures()
            
            if not fixtures:
                self.console.print("\n[yellow]No fixture files found![/yellow]")
                self.console.print("[dim]Try auto-generated fixtures instead.[/dim]")
                return
            
            self.console.print("\n[green]✓ Found available fixture files:[/green]")
            
            table = Table(title="Available Fixtures", show_header=True, header_style="bold magenta")
            table.add_column("#", justify="right", style="cyan")
            table.add_column("Date", style="white")
            table.add_column("Matches", justify="right")
            table.add_column("Size (KB)", justify="right")
            table.add_column("Path", style="dim")
            
            for idx, fix in enumerate(fixtures, 1):
                table.add_row(
                    str(idx),
                    fix.get('date', 'N/A'),
                    str(fix.get('matches', 0)),
                    str(fix.get('size_kb', 0)),
                    fix.get('path', 'N/A')[:50],
                )
            
            self.console.print(table)
            self.console.print("\nB. Back\nQ. Exit")
            
            choice = Prompt.ask(
                f"\n[cyan]Select fixture (1-{len(fixtures)})[/cyan]"
            ).strip().lower()
            
            if choice == 'q':
                self._exit_menu()
                return
            elif choice == 'b':
                return
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(fixtures):
                    fixture_path = fixtures[idx]['path']
                    
                    # Preview the fixture
                    df = pd.read_csv(fixture_path)
                    if preview_fixture_selection(df, self.console):
                        # Run predictions
                        output_dir = Prompt.ask(
                            "[cyan]Output directory[/cyan]",
                            default="output/predictions"
                        )
                        verbose = Confirm.ask(
                            "[cyan]Enable verbose output?[/cyan]",
                            default=False
                        )
                        predict_command(fixture_path, output_dir, verbose)
                        return
                else:
                    self.console.print("[yellow]Invalid selection. Please try again.[/yellow]")
            except ValueError:
                self.console.print("[yellow]Please enter a valid number.[/yellow]")
    
    def _pipeline_menu(self) -> None:
        """New Daily Pipeline menu with multiple options."""
        from predicciones.src.cli.commands import (
            update_raw_sources,
            rebuild_derived_datasets,
            detect_or_generate_daily_fixtures,
            run_full_daily_pipeline,
        )
        from datetime import datetime
        
        while True:
            self.console.print()
            self.console.print(Panel(
                "[bold]Daily Pipeline Menu[/bold]\n"
                "1. Update raw data sources\n"
                "2. Rebuild derived datasets\n"
                "3. Detect/generate today fixtures\n"
                "4. Run full daily pipeline\n"
                "5. Run full daily pipeline + predictions\n"
                "B. Back\n"
                "Q. Exit",
                title="⚽ Daily Pipeline",
                border_style="blue"
            ))
            
            choice = Prompt.ask("\n[cyan]Enter your choice[/cyan]", default="4").strip().lower()
            
            if choice == 'q':
                self._exit_menu()
                return
            elif choice == 'b':
                return
            elif choice == '1':
                verbose = Confirm.ask("[cyan]Enable verbose output?[/cyan]", default=False)
                update_raw_sources(self.console, verbose)
            elif choice == '2':
                verbose = Confirm.ask("[cyan]Enable verbose output?[/cyan]", default=False)
                rebuild_derived_datasets(self.console, verbose)
            elif choice == '3':
                date_str = Prompt.ask(
                    "[cyan]Enter date (YYYY-MM-DD or YYYYMMDD, leave empty for today)[/cyan]",
                    default=""
                ).strip()
                if not date_str:
                    date_str = None
                detect_or_generate_daily_fixtures(self.console, date_str)
            elif choice == '4':
                date_str = Prompt.ask(
                    "[cyan]Enter date (YYYY-MM-DD or YYYYMMDD, leave empty for today)[/cyan]",
                    default=""
                ).strip()
                if not date_str:
                    date_str = None
                verbose = Confirm.ask("[cyan]Enable verbose output?[/cyan]", default=False)
                run_full_daily_pipeline(self.console, date_str, run_predictions=False, verbose=verbose)
            elif choice == '5':
                date_str = Prompt.ask(
                    "[cyan]Enter date (YYYY-MM-DD or YYYYMMDD, leave empty for today)[/cyan]",
                    default=""
                ).strip()
                if not date_str:
                    date_str = None
                verbose = Confirm.ask("[cyan]Enable verbose output?[/cyan]", default=False)
                run_full_daily_pipeline(self.console, date_str, run_predictions=True, verbose=verbose)
            else:
                self.console.print("[yellow]Invalid choice. Please try again.[/yellow]")
    
    def _players_menu(self) -> None:
        """Player statistics menu with team selection from available data."""
        self.console.print("\n[bold]Player Statistics[/bold]")
        
        # Import helper
        from predicciones.src.cli.commands import list_available_teams
        
        # Get available teams
        teams = list_available_teams()
        
        if teams:
            self.console.print(f"\n[green]✓ Found {len(teams)} teams in available data:[/green]")
            
            # Show teams in columns
            for idx, team in enumerate(teams, 1):
                self.console.print(f"  [cyan]{idx}.[/cyan] {team}")
            
            choice = Prompt.ask(
                f"\n[cyan]Select team (1-{len(teams)}) or type custom team name[/cyan]"
            )
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(teams):
                    team = teams[idx]
                else:
                    team = choice
            except ValueError:
                team = choice
        else:
            self.console.print("\n[yellow]No teams found in available data.[/yellow]")
            team = Prompt.ask("[cyan]Enter team name[/cyan]")
        
        max_matches = IntPrompt.ask(
            "[cyan]Max matches to consider[/cyan]",
            default=10
        )
        
        output_format = Prompt.ask(
            "[cyan]Output format[/cyan]",
            choices=["table", "csv", "json"],
            default="table"
        )
        
        # Import and run command
        from predicciones.src.cli.commands import players_command
        players_command(team, max_matches, output_format)
    
    def _timelines_menu(self) -> None:
        """Match timeline menu with guided team and match selection."""
        from predicciones.src.cli.match_timeline import run_match_timeline_menu

        run_match_timeline_menu(self.console)
    
    def _fixtures_menu(self) -> None:
        """Fixtures menu with date selection from available data."""
        self.console.print("\n[bold]View Fixtures[/bold]")
        
        # Import helpers
        from predicciones.src.cli.commands import list_available_fixtures, list_available_predictions
        
        # Get available dates
        fixtures = list_available_fixtures()
        predictions = list_available_predictions()
        
        available_dates = set()
        for f in fixtures:
            available_dates.add(f.get('date', ''))
        for p in predictions:
            available_dates.add(p.get('date', ''))
        
        available_dates = sorted([d for d in available_dates if d])
        
        if available_dates:
            self.console.print(f"\n[green]✓ Found fixtures for {len(available_dates)} dates:[/green]")
            for idx, date in enumerate(available_dates, 1):
                self.console.print(f"  [cyan]{idx}.[/cyan] {date}")
            
            choice = Prompt.ask(
                f"\n[cyan]Select date (1-{len(available_dates)}) or enter new date (YYYYMMDD)[/cyan]"
            )
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(available_dates):
                    date = available_dates[idx]
                else:
                    date = choice
            except ValueError:
                date = choice
        else:
            from datetime import datetime
            suggested_date = datetime.now().strftime("%Y%m%d")
            self.console.print(f"[dim]Suggested: {suggested_date}[/dim]")
            date = Prompt.ask(
                "[cyan]Date (YYYYMMDD)[/cyan]",
                default=suggested_date
            )
        
        competition = Prompt.ask(
            "[cyan]Competition filter (optional, leave empty for all)[/cyan]",
            default=""
        ) or None
        
        output_format = Prompt.ask(
            "[cyan]Output format[/cyan]",
            choices=["table", "csv"],
            default="table"
        )
        
        # Import and run command
        from predicciones.src.cli.commands import fixtures_command
        fixtures_command(date, competition, output_format)
    
    def _daily_report_menu(self) -> None:
        """Daily report menu with selection from available predictions."""
        self.console.print("\n[bold]Generate Daily Report[/bold]")
        
        # Import helpers
        from predicciones.src.cli.commands import list_available_predictions, list_available_reports
        
        # Get available prediction dates
        predictions = list_available_predictions()
        existing_reports = [r.get('date') for r in list_available_reports()]
        
        if predictions:
            self.console.print(f"\n[green]✓ Found predictions for {len(predictions)} dates:[/green]")
            
            table = Table(title="Available Predictions for Reports", show_header=True, header_style="bold magenta")
            table.add_column("#", justify="right", style="cyan")
            table.add_column("Date", style="white")
            table.add_column("Status", style="yellow")
            table.add_column("Size (KB)", justify="right")
            
            for idx, pred in enumerate(predictions, 1):
                date = pred.get('date', '')
                has_report = "✓ Report exists" if date in existing_reports else "Needs report"
                status_style = "green" if date in existing_reports else "yellow"
                
                table.add_row(
                    str(idx),
                    date,
                    f"[{status_style}]{has_report}[/{status_style}]",
                    str(pred.get('size_kb', 0)),
                )
            
            self.console.print(table)
            
            choice = Prompt.ask(
                f"\n[cyan]Select date (1-{len(predictions)}) or enter date (YYYYMMDD)[/cyan]"
            )
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(predictions):
                    date = predictions[idx]['date']
                else:
                    date = choice
            except ValueError:
                date = choice
        else:
            self.console.print("\n[yellow]No prediction files found.[/yellow]")
            from datetime import datetime
            date = Prompt.ask(
                "[cyan]Enter date (YYYYMMDD)[/cyan]",
                default=datetime.now().strftime("%Y%m%d")
            )
        
        output_dir = Prompt.ask(
            "[cyan]Output directory[/cyan]",
            default="output/reports"
        )
        
        include_analysis = Confirm.ask(
            "[cyan]Include detailed analysis?[/cyan]",
            default=True
        )
        
        # Import and run command
        from predicciones.src.cli.commands import daily_report_command
        daily_report_command(date, output_dir, include_analysis)
    
    def _lambda_analysis_menu(self) -> None:
        """Lambda analysis menu with presets and suggestions."""
        self.console.print("\n[bold]Lambda Distribution Analysis[/bold]")
        
        # Load config for suggestions
        from predicciones.src.utils.config_loader import config
        
        default_num_matches = 50
        default_threshold_home = 3.0
        default_threshold_away = 2.5
        default_threshold_total = 5.0
        
        if config:
            dc_config = config.get('dixon_coles', {})
            default_threshold_home = float(dc_config.get('min_lambda', 3.0))
            default_threshold_away = float(dc_config.get('min_lambda', 2.5))
            default_threshold_total = float(dc_config.get('max_lambda', 5.0))
        
        # Offer presets
        self.console.print("\n[cyan]Presets:[/cyan]")
        self.console.print("  [cyan]1.[/cyan] Quick (20 matches)")
        self.console.print("  [cyan]2.[/cyan] Standard (50 matches)")
        self.console.print("  [cyan]3.[/cyan] Comprehensive (200 matches)")
        self.console.print("  [cyan]4.[/cyan] Custom")
        
        preset = Prompt.ask(
            "\n[cyan]Select preset[/cyan]",
            choices=["1", "2", "3", "4"],
            default="2"
        )
        
        if preset == "1":
            num_matches = 20
        elif preset == "2":
            num_matches = 50
        elif preset == "3":
            num_matches = 200
        else:
            num_matches = IntPrompt.ask(
                "[cyan]Number of matches to analyze[/cyan]",
                default=default_num_matches
            )
        
        output_dir = Prompt.ask(
            "[cyan]Output directory[/cyan]",
            default="output/lambda_validation"
        )
        
        self.console.print(f"\n[dim]Using thresholds from config (adjust if needed)[/dim]")
        
        threshold_home = float(Prompt.ask(
            "[cyan]Home lambda threshold[/cyan]",
            default=str(default_threshold_home)
        ))
        
        threshold_away = float(Prompt.ask(
            "[cyan]Away lambda threshold[/cyan]",
            default=str(default_threshold_away)
        ))
        
        threshold_total = float(Prompt.ask(
            "[cyan]Total lambda threshold[/cyan]",
            default=str(default_threshold_total)
        ))
        
        # Import and run command
        from predicciones.src.cli.commands import lambda_analysis_command
        lambda_analysis_command(num_matches, output_dir, threshold_home, threshold_away, threshold_total)
    
    def _backtest_menu(self) -> None:
        """Backtest menu with presets."""
        self.console.print("\n[bold]Backtest / Calibration Evaluation[/bold]")
        
        # Offer presets
        self.console.print("\n[cyan]Presets:[/cyan]")
        self.console.print("  [cyan]1.[/cyan] Quick (50 matches)")
        self.console.print("  [cyan]2.[/cyan] Standard (200 matches)")
        self.console.print("  [cyan]3.[/cyan] Full (1000 matches)")
        self.console.print("  [cyan]4.[/cyan] Custom")
        
        preset = Prompt.ask(
            "\n[cyan]Select preset[/cyan]",
            choices=["1", "2", "3", "4"],
            default="2"
        )
        
        if preset == "1":
            num_matches = 50
        elif preset == "2":
            num_matches = 200
        elif preset == "3":
            num_matches = 1000
        else:
            num_matches = IntPrompt.ask(
                "[cyan]Number of matches for backtest[/cyan]",
                default=200
            )
        
        output_dir = Prompt.ask(
            "[cyan]Output directory[/cyan]",
            default="output/calibration_eval"
        )
        
        compare_markov = Confirm.ask(
            "[cyan]Compare baseline vs Markov-aware?[/cyan]",
            default=True
        )
        
        # Import and run command
        from predicciones.src.cli.commands import backtest_command
        backtest_command(num_matches, output_dir, compare_markov)
    
    def _recent_files_menu(self) -> None:
        """Recent files menu with enhanced display."""
        self.console.print("\n[bold]Recent Files[/bold]")
        
        # Import helper
        from predicciones.src.cli.commands import list_recent_outputs
        
        limit = IntPrompt.ask(
            "[cyan]Number of files to show[/cyan]",
            default=10
        )
        
        file_type = Prompt.ask(
            "[cyan]File type filter[/cyan]",
            choices=["all", "predictions", "reports", "metrics"],
            default="all"
        )
        
        # Get recent files
        outputs = list_recent_outputs(limit=limit * 2)  # Get more to filter
        
        # Filter by type if needed
        if file_type != "all":
            filtered = []
            for out in outputs:
                path_lower = out.get('path', '').lower()
                name_lower = out.get('name', '').lower()
                
                if file_type == "predictions" and ("prediction" in path_lower or "daily_predictions" in path_lower):
                    filtered.append(out)
                elif file_type == "reports" and ("report" in path_lower or ".md" in out.get('type', '')):
                    filtered.append(out)
                elif file_type == "metrics" and ("metric" in path_lower or "calibration" in path_lower):
                    filtered.append(out)
            
            outputs = filtered
        
        if outputs:
            self.console.print(f"\n[green]✓ Found {len(outputs)} recent files:[/green]")
            
            table = Table(title="Recent Output Files", show_header=True, header_style="bold magenta")
            table.add_column("#", justify="right", style="cyan")
            table.add_column("Type", justify="center")
            table.add_column("Name", style="white")
            table.add_column("Date", style="dim")
            table.add_column("Size (KB)", justify="right")
            table.add_column("Path", style="dim")
            
            for idx, out in enumerate(outputs[:limit], 1):
                table.add_row(
                    str(idx),
                    out.get('type', 'FILE'),
                    out.get('name', 'N/A')[:30],
                    out.get('date', 'N/A'),
                    str(out.get('size_kb', 0)),
                    out.get('path', 'N/A')[:40],
                )
            
            self.console.print(table)
            
            show_summary = Confirm.ask(
                "\n[cyan]Show summary of latest file?[/cyan]",
                default=False
            )
            
            if show_summary and outputs:
                latest = outputs[0]
                self.console.print(f"\n[bold]Latest file:[/bold] {latest['path']}")
                self.console.print(f"[dim]Type: {latest['type']}, Size: {latest['size_kb']} KB, Date: {latest['date']}[/dim]")
        else:
            self.console.print("\n[yellow]No recent output files found.[/yellow]")
    
    def _config_menu(self) -> None:
        """Configuration menu with section selection."""
        self.console.print("\n[bold]Configuration[/bold]")
        
        # Import helper
        from predicciones.src.cli.commands import get_config_sections
        
        sections = get_config_sections()
        
        if sections:
            self.console.print(f"\n[green]✓ Available configuration sections ({len(sections)}):[/green]")
            
            for idx, section in enumerate(sections, 1):
                self.console.print(f"  [cyan]{idx}.[/cyan] {section}")
            
            self.console.print(f"  [cyan]0.[/cyan] Show all sections")
            
            choice = Prompt.ask(
                f"\n[cyan]Select section (0-{len(sections)})[/cyan]",
                choices=[str(i) for i in range(len(sections) + 1)],
                default="0"
            )
            
            if choice == "0":
                show_all = True
                section = None
            else:
                show_all = False
                idx = int(choice) - 1
                if 0 <= idx < len(sections):
                    section = sections[idx]
                else:
                    section = None
        else:
            self.console.print("\n[yellow]Could not load configuration sections.[/yellow]")
            show_all = Confirm.ask(
                "[cyan]Show all configuration?[/cyan]",
                default=False
            )
            section = None
        
        # Import and run command
        from predicciones.src.cli.commands import config_command
        config_command(show_all, section)
