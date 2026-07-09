High lambda_home detected: 3.1261 > 3.0. Match: unknown vs unknown
================================================================================
VALIDACIÓN DE CORRECCIÓN: defense_rating INVERTED FORMULA
================================================================================

RESULTADOS DESPUÉS DE LA CORRECCIÓN:
--------------------------------------------------------------------------------

Norway vs England:
  λ_home=0.932, λ_away=2.028, total=2.960
  P(Norway win)=16.5%, Draw=20.7%, P(England win)=62.8%
  ✓ England correctly favored (62.8% vs 16.5%)

Argentina vs Switzerland:
  λ_home=2.566, λ_away=1.110, total=3.675
  P(Argentina win)=68.7%, Draw=16.9%, P(Switzerland win)=14.4%
  ✓ Argentina correctly favored as expected

Argentina vs Egypt:
  λ_home=3.126, λ_away=0.560, total=3.686
  P(Argentina win)=86.9%, Draw=9.3%, P(Egypt win)=3.9%
  ✓ Argentina correctly favored as expected

================================================================================
EXPLICACIÓN DEL BUG Y LA CORRECCIÓN:
================================================================================

PROBLEMA ORIGINAL:
  - ratings_wc2026.json usa defense como FORTALEZA (>1.0 = buena defensa)
  - La fórmula original usaba: lambda = attack * defense_opponent
  - Esto hacía que una defensa fuerte INFLARA los goles del rival
  - Resultado: England (defensa 1.241) inflaba lambda de Norway

CORRECCIÓN APLICADA:
  - Nueva fórmula: lambda = attack * (1/defense_opponent)
  - Ahora defensa fuerte (>1.0) SUPRIME los goles del rival
  - Documentación actualizada en dixon_coles.py

SEMÁNTICA CORRECTA:
  - attack_rating > 1.0: ataque más fuerte que el promedio
  - defense_rating > 1.0: defensa más fuerte que el promedio (concede MENOS)
  - opponent_defense_factor = 1/defense_rating:
      • defense=1.5 → factor=0.67 (difícil marcarle)
      • defense=0.7 → factor=1.43 (fácil marcarle)

✓ TODAS LAS VALIDACIONES PASARON - CORRECCIÓN EXITOSA


---

## Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `predicciones/src/models/dixon_coles.py` | - Invertida fórmula de defense en `_predict_lambdas_heuristic()`<br>- Actualizada documentación con semántica correcta<br>- Añadidos comentarios explicativos |

## Scripts Creados

| Script | Propósito |
|--------|-----------|
| `scripts/diagnose_lambda_formula.py` | Diagnóstico inicial del bug |
| `scripts/validate_defense_fix.py` | Validación completa de la corrección |

## Impacto

- **Norway vs England**: England pasa de 38.8% → 62.8% win probability
- **Argentina matches**: Correctamente favorita en ambos casos
- **Semántica alineada**: defense_rating > 1.0 ahora correctamente suprime goles del rival

## Recomendación

✅ **CORRECCIÓN APLICADA EXITOSAMENTE**

Esta corrección debe mantenerse ya que:
1. Es un bug lógico, no tuning paramétrico
2. Las predicciones ahora son coherentes con los ratings
3. No requiere re-entrenamiento ni recalibración

---
*Reporte generado: $(date)*
