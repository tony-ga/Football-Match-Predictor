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
│   └── examples/          # Ejemplos de JSONs de partidos (Argentina vs Catar)
├── src/
│   ├── ingestion/         # Validación (Pydantic), parser JSON y CSV loader
│   ├── features/          # Feature engineering, agregadores de ratings y jugadores
│   ├── models/            # Poisson, Dixon-Coles, Derivación de mercados, Calibración
│   ├── sanity/            # Checks de coherencia pre-output (Sanity warnings)
│   ├── simulation/        # Simulaciones Monte Carlo (Opcional)
│   ├── evaluation/        # Brier Score, Log Loss, curvas de confiabilidad
│   ├── utils/             # Funciones de ayuda (e.g. Config loader)
│   └── api/               # API FastAPI para servir el modelo
├── scripts/               # Scripts CLI: predict.py, train.py, evaluate.py
├── notebooks/             # Entornos de prueba y análisis de datos
├── output/                # Archivos PKL de modelos calibrados y reportes
├── requirements.txt       # Dependencias
└── README.md              # Este archivo
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
