# Football Match Prediction System 🏆

Sistema completo de predicción de partidos de fútbol (orientado a selecciones nacionales / Mundial) que genera probabilidades calibradas para múltiples mercados a partir de un JSON prepartido estructurado.

La arquitectura es de dos capas:
1. **Motor Base:** Modelo Dixon-Coles con decaimiento temporal y corrección de empates, que estima $\lambda_{home}$ y $\lambda_{away}$ basándose en factores ofensivos, defensivos, contextuales y de calidad de plantilla. A partir de esto, se genera una matriz de marcadores (scorelines).
2. **Capa de Calibración:** Modelos independientes por mercado (Isotonic Regression o Platt Scaling) para ajustar probabilidades base hacia probabilidades del mundo real.

## Estructura del Proyecto

```text
predicciones/
├── configs/               # model_config.yaml (pesos, parámetros)
├── data/
│   ├── raw/               # CSVs históricos de FIFA
│   ├── processed/         # Datasets listos para entrenar
│   ├── cache/espn/        # Cache de respuestas ESPN (SHA256 filenames)
│   └── derived/           # Dataset derivado para markets (JSONL)
├── src/
│   ├── ingestion/         # Validación (Pydantic), parser JSON y CSV loader
│   ├── features/          # Feature engineering, agregadores de ratings y jugadores
│   ├── models/            # Poisson, Dixon-Coles, Derivación de mercados, Calibración
│   │                      # market_models.py (corners, cards, shots, player props)
│   │                      # market_availability.py (feature gating)
│   ├── data/              # ESPN clients, parsers, stats extraction
│   │                      # espn_stats_parsers.py (team/player/event parsing)
│   │                      # cache_manager.py (Windows/Linux safe caching)
│   ├── domain/            # Market types and output structures
│   │                      # market_types.py (availability, predictions)
│   ├── pipeline/          # predict.py (main), predict_markets.py (alternative markets)
│   ├── sanity/            # Checks de coherencia pre-output (Sanity warnings)
│   ├── simulation/        # Simulaciones Monte Carlo (Opcional)
│   ├── evaluation/        # Brier Score, Log Loss, curvas de confiabilidad
│   ├── utils/             # Funciones de ayuda (e.g. Config loader)
│   └── api/               # API FastAPI para servir el modelo
├── scripts/               # Scripts CLI
│   ├── predict_match.py   # Predicción con modos teams/match/upcoming/json
│   └── build_market_dataset.py  # Construcción de dataset histórico
├── notebooks/             # Entornos de prueba y análisis de datos
├── output/                # Archivos PKL de modelos calibrados y reportes
├── tests/                 # Tests unitarios
├── requirements.txt       # Dependencias
└── README.md              # Este archivo

## Mercados Alternativos (NUEVO)

El sistema ahora soporta predicciones para mercados adicionales más allá del 1X2 y goles:

### Corners (Córners)
- **Total corners:** Over/Under líneas típicas (7.5, 8.5, 9.5, 10.5, 11.5)
- **Team totals:** Over/Under por equipo (3.5, 4.5, 5.5, 6.5)
- **More corners:** Qué equipo tendrá más córners
- **First/Last corner:** (solo si hay datos de eventos a nivel temporal)

### Cards (Tarjetas)
- **Total cards:** Over/Under líneas típicas (3.5, 4.5, 5.5, 6.5)
- **Team totals:** Over/Under por equipo (1.5, 2.5, 3.5)
- **More cards:** Qué equipo recibirá más tarjetas
- **First card:** (solo si hay datos de eventos)

### Shots on Target (Tiros a Puerta)
- **Total SOT:** Over/Under líneas típicas (7.5, 8.5, 9.5, 10.5, 11.5)
- **Team SOT:** Over/Under por equipo (3.5, 4.5, 5.5)

### Player Props (Jugadores)
- **Anytime scorer:** Probabilidad de que un jugador anote en cualquier momento
- **First scorer:** Probabilidad de que un jugador anote el primer gol
- **Player SOT:** Over/Under tiros a puerta por jugador (0.5, 1.5, 2.5)
- **Player assists:** Over/Under asistencias por jugador (0.5)

**Nota sobre disponibilidad:** Los mercados dependen de la cobertura de datos de ESPN. Si no hay suficientes datos históricos con estadísticas específicas, el mercado se marcará como `available: false` con una razón explicativa. No se generan predicciones inventadas.

## Modelos Estadísticos

### Corners & Cards & Shots
- **Modelo base:** Distribución de Poisson para conteos de eventos
- **Más corners/cards:** Aproximación normal a la distribución de Skellam (diferencia de Poissons)
- **Cards:** Incluye shrinkage hacia el promedio de liga para estabilidad

### Player Props
- **Enfoque heurístico/hierárquico:** Combina:
  - Expected goals del equipo
  - Shot share del jugador
  - Goal/assist historical rates
  - Starter probability y minutos jugados
  - Adjuste por rol (penaltis, tiros libres)

## Feature Gating y Availability

Cada mercado tiene requisitos mínimos de datos:

| Mercado | Mínimo muestras | Requiere eventos |
|---------|-----------------|------------------|
| Corners total | 3 partidos con corners | No |
| First corner | 3 partidos | Sí |
| Cards total | 3 partidos con cards | No |
| First card | 3 partidos | Sí |
| Shots SOT | 3 partidos con SOT | No |
| Player props | 3 partidos + lineups | Sí (lineups) |

Cuando un mercado no está disponible, el output incluye:
```json
{
  "available": false,
  "reason": "Insufficient matches with corner data (2 < 3)",
  "sample_size": 2,
  "confidence": "low",
  "data_source": "unavailable"
}
```

## Construcción de Dataset Histórico

Para construir un dataset local de estadísticas históricas:

```bash
# Últimos 180 días
python scripts/build_market_dataset.py --days-back 180 --league fifa.world

# Rango específico
python scripts/build_market_dataset.py --start-date 2025-01-01 --end-date 2025-06-01

# Máximo de partidos a procesar
python scripts/build_market_dataset.py --days-back 90 --max-matches 200
```

Esto descargará datos de ESPN, extraerá estadísticas de equipos y jugadores, y guardará:
- `data/derived/team_match_stats.jsonl` - Estadísticas por partido
- `data/derived/player_match_stats.jsonl` - Estadísticas por jugador
- `data/derived/match_events.jsonl` - Eventos temporales (corners, cards, goles)
- `data/cache/espn/*.json` - Respuestas raw de API (cache)

**Importante:** Los archivos de cache y derived están en `.gitignore` y no se suben al repositorio.

## Uso con Mercados Alternativos

El script `predict_match.py` soporta el flag `--include-markets`:

```bash
# Predicción por nombres de equipo con mercados
python scripts/predict_match.py --mode teams --home "USA" --away "Belgium" --include-markets

# Predicción por event ID con mercados
python scripts/predict_match.py --mode match --event-id 760506 --include-markets

# Próximo partido automático con mercados
python scripts/predict_match.py --mode upcoming --auto-pick 0 --include-markets
```

El output JSON incluirá una sección `markets`:
```json
{
  "predictions": { ... },  // 1X2, BTTS, Over/Under goles (existente)
  "markets": {
    "corners": {
      "available": true,
      "sample_size": 6,
      "confidence": "medium",
      "data_source": "espn_recent_matches",
      "predictions": {
        "total_over_under": {"over_8": 0.65, "under_8": 0.35, ...},
        "team_totals": {"home_over_4": 0.58, ...},
        "more_corners_team": {"home": 0.40, "away": 0.45, "tie": 0.15}
      }
    },
    "cards": {...},
    "shots_on_target": {...},
    "player_props": {...}
  }
}
```

## Advertencias sobre ESPN API

⚠️ **ESPN es una API no oficial con cobertura variable:**

1. **Cobertura parcial:** No todos los partidos tienen estadísticas completas de corners, cards o shots. Algunos solo tienen resultado final.

2. **Formato variable:** La estructura de respuestas puede cambiar entre competiciones o con el tiempo.

3. **Player stats limitados:** Las estadísticas de jugadores no siempre están disponibles, especialmente para ligas menores o partidos antiguos.

4. **Eventos temporales:** Datos como "primer córner" o "primera tarjeta" requieren parsing de plays/events, que puede no estar disponible para todos los partidos.

El sistema está diseñado para manejar estas limitaciones gracefulmente mediante feature gating. Nunca se inventan datos faltantes.

## Testing

Ejecutar tests unitarios:

```bash
cd predicciones
python -c "
from src.domain.market_types import *
from src.models.market_models import *
from src.models.market_availability import *
from src.data.espn_stats_parsers import *
from src.data.cache_manager import *
from src.pipeline.predict_markets import *
print('All imports OK')
"
```

Tests específicos en `tests/test_market_modules.py`.
```

## Instalación

1. Clona el repositorio y ubícate en la carpeta del proyecto.
2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instala los requerimientos:
   ```bash
   pip install -r requirements.txt
   ```

## Uso (Predicción)

Puedes predecir el resultado de un partido y todos sus mercados usando el script CLI.

### 1. Preparar un JSON de Entrada
El sistema espera un JSON con metadata, `team1` y `team2`. Hay un ejemplo completo disponible en `data/examples/argentina_catar.json`.

### 2. Ejecutar la predicción
```bash
python scripts/predict.py data/examples/argentina_catar.json
```

El script mostrará:
1. Los $\lambda$ estimados para cada equipo usando los pesos heurísticos.
2. Los mercados de 1X2, Over/Under, Both Teams To Score (BTTS), Totales de Equipo, Porterías a Cero y Marcadores Correctos.
3. Avisos del *Sanity Checker* si detecta incoherencias.

## Levantando la API

Para integrar el modelo con otros sistemas o dashboards, puedes usar la API basada en `FastAPI`:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Prueba la API enviando un `POST` a `/predict` con el contenido del JSON del partido, o visita la documentación interactiva en `http://localhost:8000/docs`.

##  Flujo y Módulos Core

1. **Ingesta y Validación (`src/ingestion`):** Usa Pydantic v2 para validar estrictamente que el JSON cumpla con todos los factores tácticos, colectivos, de contexto y la lista de jugadores. Rellena los datos ausentes con defaults estadísticos.
2. **Feature Engineering (`src/features`):** 
    - Convierte victorias/empates en *Forma* base y decaimiento.
    - Pondera los jugadores (`player_aggregator.py`) por su titularidad e impacto para derivar un multiplicador `squad_quality`.
    - Deriva *Ratings de Ataque/Defensa* combinando el *xG* base con la eficiencia de finalización táctica.
3. **Dixon-Coles (`src/models/dixon_coles.py`):** Estima el $\lambda$ (goles esperados) de local y visitante. Usa un modo *heurístico* si el modelo no está entrenado (para poder operar out-of-the-box). Genera la matriz de probabilidades de marcadores con corrección para marcadores bajos ($\rho$).
4. **Mercados (`src/models/market_derivation.py`):** Deriva probabilísticamente todo:
    - `1X2` = Suma de la diagonal y triángulos de la matriz.
    - `BTTS` = $1 - P(home=0) - P(away=0) + P(0,0)$
    - `Over/Under`, `Clean Sheets`, etc.
5. **Calibración (`src/models/calibration.py`):** (Activable pos-entrenamiento) Toma los mercados derivados y aplica un filtro `Isotonic` o `Platt`.
6. **Sanity Checker (`src/sanity/sanity_checker.py`):** Dispara advertencias antes de emitir la salida (ej: *"El equipo local es muy favorito pero su probabilidad P(1) es extrañamente baja"*).
