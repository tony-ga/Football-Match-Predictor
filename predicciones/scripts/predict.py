#!/usr/bin/env python
import argparse
import json
import logging
from pathlib import Path
from rich.console import Console
from rich.pretty import pprint

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.json_parser import load_match_json
from src.pipeline.predict import predict_match_pipeline
from src.utils.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

def main():
    parser = argparse.ArgumentParser(description="Predict a match using real-time API or local data.")
    parser.add_argument("json_file", type=str, help="Path to the match JSON file")
    parser.add_argument("--refresh-data", action="store_true", help="Force refresh data from API (ignore cache)")
    parser.add_argument("--source", type=str, default="auto", choices=["api_football", "football_data", "open_football", "auto"], help="Preferred API source")
    args = parser.parse_args()

    console.print(f"[bold blue]Loading match metadata from {args.json_file}[/bold blue]")
    try:
        match_input = load_match_json(args.json_file)
    except Exception as e:
        console.print(f"[bold red]Error parsing JSON: {e}[/bold red]")
        return

    home_team = match_input.metadata.home_team
    away_team = match_input.metadata.away_team
    match_date = match_input.metadata.date
    neutral_venue = match_input.metadata.neutral_venue

    console.print(f"[bold yellow]Running Prediction Pipeline for {home_team} vs {away_team}...[/bold yellow]")
    
    try:
        response = predict_match_pipeline(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            neutral_venue=neutral_venue,
            refresh_data=args.refresh_data,
            api_source=args.source
        )
    except Exception as e:
        console.print(f"[bold red]Error running prediction pipeline: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return

    console.print("\n[bold green]=== FINAL PREDICTION OUTPUT ===[/bold green]")
    
    # Prune giant distributions for display to keep it neat
    display_response = response.copy()
    display_markets = display_response.get("predictions", {}).copy()
    if 'home_goals_distribution' in display_markets:
        del display_markets['home_goals_distribution']
    if 'away_goals_distribution' in display_markets:
        del display_markets['away_goals_distribution']
    display_response["predictions"] = display_markets

    def format_percentages(data, path=""):
        """
        Format numeric values for display.
        
        - Probabilities (0-1 range) are shown as percentages like "29.11%"
        - Lambdas, expected goals, and totals remain as floats
        - player_props probability fields are formatted as percentages
        
        Args:
            data: The data structure to format
            path: Current path in the data structure (for context-aware formatting)
            
        Returns:
            Formatted data structure with percentages as strings
        """
        if isinstance(data, dict):
            return {k: format_percentages(v, path=f"{path}.{k}") for k, v in data.items()}
        elif isinstance(data, list):
            return [format_percentages(v, path) for v in data]
        elif isinstance(data, float):
            # Keep lambdas, expected goals, and totals as numeric
            if any(keyword in path.lower() for keyword in [
                "lambda", "expected_goals", "expected_total", 
                "weight", "xg", "goals_per_match"
            ]):
                return round(data, 3)
            
            # Check if this is a probability field in player_props
            # that should be displayed as percentage
            if "probability" in path.lower() or "prob" in path.lower():
                # Ensure it's in 0-1 range before converting to percentage
                if 0 <= data <= 1.0:
                    return f"{data * 100:.2f}%"
                elif data > 1.0 and data <= 100:
                    # Already a percentage, just format
                    return f"{data:.2f}%"
            
            # General case: assume 0-1 probability for other fields
            # Skip fields that look like counts or IDs
            if any(keyword in path.lower() for keyword in [
                "count", "id", "size", "sample", "matches", "goals_recent"
            ]):
                return round(data, 4)
                
            # Default: format as percentage for probabilities
            if 0 <= data <= 1.0:
                return f"{data * 100:.2f}%"
            elif data > 1.0 and data <= 100:
                return f"{data:.2f}%"
            
            return round(data, 4)
        return data

    def replace_team_names(data, home_name, away_name):
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                new_k = str(k).replace('home', home_name).replace('away', away_name)
                new_dict[new_k] = replace_team_names(v, home_name, away_name)
            return new_dict
        elif isinstance(data, list):
            return [replace_team_names(i, home_name, away_name) for i in data]
        return data

    display_response = format_percentages(display_response)
    display_response = replace_team_names(display_response, home_team, away_team)
    pprint(display_response)

if __name__ == "__main__":
    main()

