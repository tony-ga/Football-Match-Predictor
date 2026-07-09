"""
Interactive Menu for Football Prediction System

Provides a text-based interactive menu interface using Rich.
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
        """Predict from fixture file menu."""
        self.console.print("\n[bold]Predict Matches from Fixture File[/bold]")
        
        fixture_path = Prompt.ask(
            "[cyan]Path to fixture CSV file[/cyan]",
            default="data/fixtures/test.csv"
        )
        
        output_dir = Prompt.ask(
            "[cyan]Output directory[/cyan]",
            default="output/predictions"
        )
        
        verbose = Confirm.ask(
            "[cyan]Enable verbose output?[/cyan]",
            default=False
        )
        
        # Import and run command
        from predicciones.src.cli.commands import predict_command
        predict_command(fixture_path, output_dir, verbose)
    
    def _pipeline_menu(self) -> None:
        """Daily pipeline menu."""
        self.console.print("\n[bold]Run Daily Pipeline[/bold]")
        
        date = Prompt.ask(
            "[cyan]Date to process (YYYYMMDD)[/cyan]",
            default=""
        )
        
        if not date:
            self.console.print("[yellow]Date is required.[/yellow]")
            return
        
        output_dir = Prompt.ask(
            "[cyan]Output directory[/cyan]",
            default="output/daily"
        )
        
        verbose = Confirm.ask(
            "[cyan]Enable verbose output?[/cyan]",
            default=False
        )
        
        # Import and run command
        from predicciones.src.cli.commands import pipeline_command
        pipeline_command(date, output_dir, verbose)
    
    def _players_menu(self) -> None:
        """Player statistics menu."""
        self.console.print("\n[bold]Player Statistics[/bold]")
        
        team = Prompt.ask(
            "[cyan]Team name[/cyan]"
        )
        
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
        """Match timeline menu."""
        self.console.print("\n[bold]Match Timeline[/bold]")
        
        match_id = Prompt.ask(
            "[cyan]Match ID or fixture[/cyan]"
        )
        
        output_dir = Prompt.ask(
            "[cyan]Output directory[/cyan]",
            default="output/timelines"
        )
        
        # Import and run command
        from predicciones.src.cli.commands import timelines_command
        timelines_command(match_id, output_dir)
    
    def _fixtures_menu(self) -> None:
        """Fixtures menu."""
        self.console.print("\n[bold]View Fixtures[/bold]")
        
        date = Prompt.ask(
            "[cyan]Date (YYYYMMDD)[/cyan]"
        )
        
        competition = Prompt.ask(
            "[cyan]Competition filter (optional)[/cyan]",
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
        """Daily report menu."""
        self.console.print("\n[bold]Generate Daily Report[/bold]")
        
        date = Prompt.ask(
            "[cyan]Date (YYYYMMDD)[/cyan]"
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
        """Lambda analysis menu."""
        self.console.print("\n[bold]Lambda Distribution Analysis[/bold]")
        
        num_matches = IntPrompt.ask(
            "[cyan]Number of matches to analyze[/cyan]",
            default=50
        )
        
        output_dir = Prompt.ask(
            "[cyan]Output directory[/cyan]",
            default="output/lambda_validation"
        )
        
        threshold_home = float(Prompt.ask(
            "[cyan]Home lambda threshold[/cyan]",
            default="3.0"
        ))
        
        threshold_away = float(Prompt.ask(
            "[cyan]Away lambda threshold[/cyan]",
            default="2.5"
        ))
        
        threshold_total = float(Prompt.ask(
            "[cyan]Total lambda threshold[/cyan]",
            default="5.0"
        ))
        
        # Import and run command
        from predicciones.src.cli.commands import lambda_analysis_command
        lambda_analysis_command(num_matches, output_dir, threshold_home, threshold_away, threshold_total)
    
    def _backtest_menu(self) -> None:
        """Backtest menu."""
        self.console.print("\n[bold]Backtest / Calibration Evaluation[/bold]")
        
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
        """Recent files menu."""
        self.console.print("\n[bold]Recent Files[/bold]")
        
        limit = IntPrompt.ask(
            "[cyan]Number of files to show[/cyan]",
            default=10
        )
        
        file_type = Prompt.ask(
            "[cyan]File type filter[/cyan]",
            choices=["all", "predictions", "reports", "metrics"],
            default="all"
        )
        
        show_summary = Confirm.ask(
            "[cyan]Show summary of latest file?[/cyan]",
            default=False
        )
        
        # Import and run command
        from predicciones.src.cli.commands import recent_command
        recent_command(limit, file_type, show_summary)
    
    def _config_menu(self) -> None:
        """Configuration menu."""
        self.console.print("\n[bold]Configuration[/bold]")
        
        show_all = Confirm.ask(
            "[cyan]Show all configuration?[/cyan]",
            default=False
        )
        
        section = None
        if not show_all:
            section = Prompt.ask(
                "[cyan]Specific section (or leave empty for summary)[/cyan]",
                default=""
            ) or None
        
        # Import and run command
        from predicciones.src.cli.commands import config_command
        config_command(show_all, section)
