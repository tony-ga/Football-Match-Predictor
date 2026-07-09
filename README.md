# ⚽ Football Prediction System v2.0

Sistema profesional de predicción y análisis de partidos de fútbol con interfaz de consola mejorada.

## 🚀 Inicio Rápido

### Ejecutar aplicación interactiva (recomendado)

```bash
python app.py
```

Esto abrirá el menú interactivo donde podrás navegar por todas las opciones.

### Modo CLI directo (usuarios avanzados)

```bash
# Ver ayuda
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
  2   👥 Obtener datos de jugadores de una selección
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
├── app.py                      # Entry point principal
├── app_config.ini              # Configuración centralizada
├── predicciones/
│   ├── src/
│   │   ├── cli/                # Nueva capa CLI
│   │   │   ├── __init__.py
│   │   │   ├── menu.py         # Menú interactivo
│   │   │   └── commands.py     # Lógica de comandos
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
# → Elegir selección: Argentina (1)
# → Max partidos: 10
# → Formato: table

# Desde CLI:
python app.py players \
  --team Argentina \
  --max-matches 10 \
  --format table
```

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

Los comandos CLI reutilizan scripts existentes cuando están disponibles:

- `lambda-analysis` → ejecuta `scripts/analyze_lambda_distribution.py` si existe
- `backtest` → ejecuta `scripts/backtest_temporal_calibration_v2.py` si existe

Esto mantiene compatibilidad con análisis previos mientras proporciona una interfaz unificada.

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

## 🤝 Contribución

La arquitectura está diseñada para ser extensible:

1. **Nuevos comandos**: Agregar función en `predicciones/src/cli/commands.py`
2. **Nuevas opciones de menú**: Extender `InteractiveMenu.main_options` en `menu.py`
3. **Nuevos modelos**: Implementar en `predicciones/src/models/`

## 📄 Licencia

[Información de licencia]

## 📞 Soporte

Para issues o preguntas, revisar documentación en `output/reports/` o contactar al equipo de desarrollo.
