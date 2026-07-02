# Real-time Football API Integration Walkthrough

I have implemented a robust real-time API ingestion and profiling layer that integrates seamlessly with our Dixon-Coles prediction engine.

## What Was Added & Changed

### 1. Unified Team Mapping (`src/data/team_mappings.json`)
- Maps standard and common national team names (e.g. "USA", "United States", "Democratic Republic of the Congo", "Congo DR") to their respective unique IDs for both `api-sports.io` and `football-data.org`.

### 2. HTTP API Client (`src/data/api_client.py`)
- **Fallback Hierarchy:** Tries API-Football first $\rightarrow$ falls back to Football-Data.org $\rightarrow$ falls back to Open-Football's static JSON endpoints $\rightarrow$ falls back to local historical datasets.
- **Robust Caching:** Uses `cachetools.TTLCache` for in-memory caching (1-hour TTL) to prevent repeated requests, and saves response files locally to `data/cache/{date}_{endpoint}.json` for offline usage.
- **Rate-Limiting & Retries:** Employs `tenacity` retry with exponential backoff to handle rate limits and temporary connection issues.
- **Fuzzy Resolution:** Leverages `fuzzywuzzy` Levenshtein-distance matching to resolve team names with a minimum threshold of 80% similarity.

### 3. Match Feature Builder & Time Decay (`src/data/feature_builder.py`)
- Computes decay-weighted average goals, corners, and cards.
- **Time Decay:** Applies an exponential decay weight:
  \[
  W = e^{-\phi \times \text{days\_ago}}
  \]
  where $\phi = 0.0065$. Automatically filters out matches older than 1095 days (36 months).
- **World Cup Boosts:** Matches from the World Cup are weighted more heavily:
  - Group Stage: $\times 1.3$
  - R16 / Quarter Finals: $\times 1.5$
  - Semis / Finals: $\times 1.8$

### 4. Unified Prediction Pipeline (`src/pipeline/predict.py`)
- Coordinates the entire execution flow:
  1. Resolves team profiles via API fetch (or cached files).
  2. Applies Dixon-Coles on decay-weighted lambdas.
  3. Derives markets and runs the calibration manager.
  4. Appends sanity check multiplicative guardrails only after calibration.
  5. Formats outputs to include rich team contexts, recent form logs, and data freshness metrics.

### 5. CLI and API Adaptations
- Updated `scripts/predict.py` with flags `--refresh-data` and `--source`.
- Modified `src/api/routes/predict.py` to route all incoming prediction requests through the new unified pipeline.
- Added comprehensive unit tests in `tests/test_api_client.py`.
