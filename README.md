# ⚽ Football Prediction System v2.0

Sistema profesional de predicción y análisis de partidos de fútbol con interfaz de consola mejorada (CLI) basada en Typer.

## 🚀 Inicio Rápido

### Ejecutar aplicación interactiva (recomendado)

```bash
python app.py
```

Esto abrirá el menú interactivo donde podrás navegar por todas las opciones.

### Modo CLI directo (usuarios avanzados)

```bash
# Ver ayuda general
python app.py --help

# Generar predicciones desde fixture CSV
python app.py predict --fixture data/fixtures/test.csv --verbose

# Correr pipeline diario completo
python app.py pipeline --date 20260711 --output-dir output/daily

# Ver configuración
python app.py config --all

# Ver archivos recientes
python app.py recent --limit 10
```

## 📋 Menú Interactivo

Al ejecutar `python app.py` sin argumentos, verás el siguiente menú:

```
╭─────────────────────────────────────────────╮
│  ⚽ Football Prediction System              │
│  Professional Match Analysis & Prediction   │
╰─────────────────────────────────────────────╯

Menú Principal
  1   📊 Generar predicciones desde fixture CSV
  2   👥 Player Statistics (estadísticas por jugador/partido)
  3   ⏱️  Obtener timelines de partidos pasados
  4   📅 Obtener fixtures de una fecha
  5   🔄 Correr pipeline diario completo
  6   📄 Generar reporte diario
  7   📈 Analizar distribución de lambdas
  8   🧪 Ejecutar backtest/calibración
  9   📁 Ver archivos/reportes recientes
  10  ⚙️  Configuración
  0   🚪 Salir
```

**Nota:** La opción 2 (Player Statistics) ahora soporta nombres de equipos en español (ej. "Francia", "Marruecos") y reutiliza el script estable `match_player_defensive_stats` para evitar duplicados y fallos de búsqueda.

## 🛠️ Comandos CLI Disponibles

### Predicciones

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `predict` | Generar predicciones desde fixture CSV | `python app.py predict -f data/fixtures/test.csv -v` |
| `pipeline` | Ejecutar pipeline diario completo | `python app.py pipeline -d 20260711` |
| `daily-report` | Generar reporte diario | `python app.py daily-report -d 20260711` |

### Datos y Análisis

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `players` | Estadísticas de jugadores por selección | `python app.py players -t Argentina -m 10` |
| `timelines` | Timeline de eventos de partido | `python app.py timelines -m MATCH_ID` |
| `fixtures` | Fixtures por fecha | `python app.py fixtures -d 20260711` |

### Evaluación

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `lambda-analysis` | Analizar distribución de lambdas | `python app.py lambda-analysis -n 50` |
| `backtest` | Backtest y calibración | `python app.py backtest -n 200` |

### Utilidades

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `recent` | Ver archivos recientes | `python app.py recent -l 10 -s` |
| `config` | Ver/editar configuración | `python app.py config --all` |

## 📁 Estructura del Proyecto

```
/workspace/
├── app.py                      # Entry point principal (Typer CLI)
├── app_config.ini              # Configuración centralizada
├── predicciones/
│   ├── src/
│   │   ├── cli/                # Capa CLI (nueva)
│   │   │   ├── __init__.py     # Exporta InteractiveMenu y comandos
│   │   │   ├── menu.py         # Menú interactivo con Rich
│   │   │   └── commands.py     # Lógica de comandos Typer
│   │   ├── models/             # Modelos de predicción
│   │   ├── pipeline/           # Pipeline de predicción
│   │   ├── features/           # Ingeniería de features
│   │   ├── data/               # Capa de datos
│   │   ├── utils/              # Utilidades
│   │   └── eval/               # Evaluación y métricas
│   └── configs/
│       ├── model_config.yaml   # Configuración del modelo
│       └── daily_predictions_config.json
├── data/                       # Datos de entrada
├── output/                     # Resultados generados
└── scripts/                    # Scripts utilitarios
```

## 🔧 Configuración

### Archivos de Configuración

1. **`app_config.ini`** - Configuración de la aplicación CLI
   - Directorios por defecto
   - Thresholds de alertas
   - Opciones de visualización

2. **`predicciones/configs/model_config.yaml`** - Configuración del modelo
   - Parámetros Dixon-Coles
   - Weights de features
   - Configuración de calibración

### Ver Configuración

```bash
# Ver resumen
python app.py config

# Ver toda la configuración
python app.py config --all

# Ver sección específica
python app.py config --section dixon_coles

# Editar configuración (abre editor)
python app.py config --edit
```

## 📊 Outputs Generados

La aplicación organiza los resultados en subdirectorios bajo `output/`:

| Directorio | Contenido |
|------------|-----------|
| `output/predictions/` | Predicciones de partidos |
| `output/daily/` | Resultados del pipeline diario |
| `output/reports/` | Reportes en Markdown |
| `output/players/` | Estadísticas de jugadores |
| `output/timelines/` | Timelines de partidos |
| `output/lambda_validation/` | Análisis de distribución de lambdas |
| `output/calibration_eval/` | Métricas de backtest |

## 🎯 Flujo de Uso Típico

### Opción 1: Menú Interactivo (Recomendado para nuevos usuarios)

1. Ejecutar `python app.py`
2. Seleccionar opción del menú (ej. `1` para predicciones)
3. Seguir prompts para inputs (ruta de fixture, directorio, etc.)
4. Ver resultados en consola
5. Decidir si continuar o salir

### Opción 2: CLI Directo (Para automatización)

```bash
# Script de ejemplo para procesamiento diario
#!/bin/bash
DATE=$(date +%Y%m%d)

# Correr pipeline
python app.py pipeline --date $DATE --verbose

# Generar reporte
python app.py daily-report --date $DATE

# Ver resultados
python app.py recent --limit 5 --summary
```

## 🔍 Ejemplos Detallados

### Generar Predicciones

```bash
# Desde menú interactivo:
python app.py
# → Seleccionar opción 1
# → Ingresar ruta: data/fixtures/test.csv
# → Ingresar directorio: output/predictions
# → Modo detallado: yes

# Desde CLI:
python app.py predict \
  --fixture data/fixtures/test.csv \
  --output-dir output/predictions \
  --verbose
```

### Analizar Jugadores de una Selección

```bash
# Desde menú interactivo:
python app.py
# → Seleccionar opción 2
# → Elegir selección: Francia (28) - ahora soporta nombres en español
# → Max partidos: 10
# → Formato: table

# Desde CLI:
python app.py players \
  --team France \
  --max-matches 10 \
  --format table

# También acepta aliases en español:
python app.py players \
  --team Francia \
  --max-matches 10 \
  --format table
```

**Nota sobre normalización de equipos:** El sistema ahora soporta nombres de equipos en español e inglés de forma intercambiable. Los siguientes aliases son equivalentes:
- `Francia` = `France`
- `Inglaterra` = `England`  
- `España` = `Spain`
- `Alemania` = `Germany`
- `Marruecos` = `Morocco`
- `Países Bajos` = `Netherlands`
- `Corea del Sur` = `South Korea`
- `Estados Unidos` = `United States` / `USA`

La opción 2 (Player Statistics) reutiliza el script estable `match_player_defensive_stats` y resuelve automáticamente los nombres canónicos antes de buscar datos, evitando duplicados y fallos por idioma.

### Backtest y Calibración

```bash
# Desde menú interactivo:
python app.py
# → Seleccionar opción 8
# → Número de partidos: 200
# → Directorio: output/calibration_eval
# → Comparar Markov: yes

# Desde CLI:
python app.py backtest \
  --num-matches 200 \
  --output-dir output/calibration_eval \
  --compare-markov
```

## 🧩 Integración con Scripts Existentes

Los comandos CLI reutilizan la lógica existente del proyecto:

- `predict` → `predicciones.src.pipeline.predict`
- `pipeline` → `predicciones.scripts.run_daily_pipeline`
- `daily-report` → `predicciones.scripts.generate_daily_report`
- `lambda-analysis` → `predicciones.scripts.analyze_lambda_distribution`
- `backtest` → `predicciones.scripts.backtest_temporal_calibration_v2`
- `players` → `predicciones.scripts.match_player_defensive_stats` (wrapper estable)
- `timelines` → `predicciones.src.data.api.get_match_timeline`
- `fixtures` → `predicciones.src.data.api.get_fixtures_by_date`

Esto mantiene compatibilidad con análisis previos mientras proporciona una interfaz unificada.

### Normalización de Equipos

El sistema incluye un módulo de normalización (`team_normalization.py`) que mapea aliases en español e inglés a nombres canónicos internos:

```python
# Ejemplos de resolución automática
"Francia" → "France"
"England" → "England"  # ya canónico
"Marruecos" → "Morocco"
"España" → "Spain"
```

La opción 2 (Player Statistics) utiliza esta normalización antes de llamar al script `match_player_defensive_stats`, asegurando que:
1. No hay duplicados en el menú (cada equipo aparece una sola vez)
2. Los nombres en español se resuelven correctamente
3. Las estadísticas retornadas son únicas por `(match_id, player_id)`

## 🛡️ Sanity Checks y Validaciones

El sistema incluye validaciones automáticas:

- Thresholds de lambda configurables en `app_config.ini`
- Sanity checks en tiempo real durante predicciones
- Warnings por inconsistencias detectadas

Thresholds por defecto:
- `lambda_home_warning = 3.0`
- `lambda_away_warning = 2.5`
- `lambda_total_warning = 5.0`

## 📝 Logs y Debugging

Para habilitar modo verbose en cualquier comando:

```bash
python app.py predict -f data/fixtures/test.csv --verbose
```

O seleccionar "Modo detallado: yes" en el menú interactivo.

## 🧑‍💻 Desarrollo y Extensión

### Añadir Nuevos Comandos

1. **Nueva función comando**: Agregar en `predicciones/src/cli/commands.py`
   ```python
   @typer_app.command()
   def my_command(param: str = typer.Option(...)):
       # Lógica usando módulos existentes
   ```

2. **Registrar en app.py**: Importar y añadir al Typer app
   ```python
   from predicciones.src.cli.commands import my_command
   app.command()(my_command)
   ```

3. **Opción de menú (opcional)**: Extender `InteractiveMenu.main_options` en `menu.py`

### Arquitectura CLI

- **`app.py`**: Entry point con Typer, define comandos y opciones
- **`menu.py`**: `InteractiveMenu` clase con menú interactivo usando Rich
- **`commands.py`**: Funciones que envuelven lógica existente para Typer

## 📄 Licencia

[Información de licencia]

## 📞 Soporte

Para issues o preguntas, revisar documentación en `output/reports/` o contactar al equipo de desarrollo.

## 🔄 Cambios Recientes

### Normalización de Equipos en Player Statistics (Opción 2)

- **Problema:** La opción Player Statistics fallaba al buscar equipos con nombres en español seleccionados desde el menú (ej. "Francia", "Marruecos").
- **Solución:** 
  - Se implementó un módulo de normalización (`team_normalization.py`) que mapea aliases en español a nombres canónicos internos.
  - El menú ahora muestra cada equipo una sola vez, con su nombre en español como display name.
  - La selección del menú resuelve automáticamente al nombre canónico antes de llamar al script.
  - Se reutiliza el script estable `match_player_defensive_stats` como backend, evitando reimplementaciones defectuosas.
  
- **Equipos soportados con aliases:**
  - Francia ↔ France
  - Inglaterra ↔ England
  - España ↔ Spain
  - Alemania ↔ Germany
  - Marruecos ↔ Morocco
  - Países Bajos ↔ Netherlands
  - Corea del Sur ↔ South Korea
  - Estados Unidos ↔ United States / USA
  - Y más...

- **Validación exitosa:**
  - Francia: datos encontrados correctamente
  - Marruecos: datos encontrados correctamente
  - Argentina: sin registros duplicados
  - Todos los equipos devuelven estadísticas únicas por `(match_id, player_id)`
