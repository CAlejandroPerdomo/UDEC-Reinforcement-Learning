# Análisis de la siniestralidad vial de Santiago de Cali (2016–2025) mediante Q-Learning

**Proyecto de Reinforcement Learning aplicado al control de la siniestralidad vial.**

| | |
|---|---|
| **Autores** | Cristian Alejandro Perdomo · Edixon Rubiano |
| **Institución** | Universidad de Cundinamarca (UDEC) |
| **Repositorio** | https://github.com/CAlejandroPerdomo/UDEC-Reinforcement-Learning |
| **Alcance** | Lesionados en accidentes de tránsito, Cali (2016–2025) |
| **Entorno** | Python 3.14 · pandas 3.x · numpy 2.x · matplotlib 3.x |

---

## Descripción

Este proyecto aplica **Q-Learning** (aprendizaje por refuerzo tabular, Watkins, 1992) al
problema de **seleccionar intervenciones de control vial** en Santiago de Cali, con el
objetivo de minimizar el **costo esperado de siniestralidad** (lesionados no mitigados).
El entorno simulado se construye a partir de las **frecuencias empíricas reales** del
registro histórico, por lo que la dinámica del modelo queda trazable a los datos.

> **Nota metodológica:** los resultados corresponden a una **simulación** y **no demuestran
> reducciones reales de accidentes**. Son una propiedad del entorno simulado, no una
> predicción causal.

## Etapas del proyecto

1. **EDA y preparación** — carga, limpieza y análisis exploratorio del dataset
   (51,236 registros → 51,233 preparados).
2. **Formulación del MDP** — 32 estados (franja horaria × tipo de día × actor) y 4 acciones
   de control, con una función de recompensa negativa basada en tasas empíricas.
3. **Entrenamiento** — agente Q-Learning tabular (α=0.1, γ=0.9, ε-greedy con decaimiento).
4. **Evaluación y sensibilidad** — comparación greedy vs. base fija, y prueba de sensibilidad
   de la eficacia (100 %, 70 % y 50 %).

## Estructura del repositorio

```
.
├── analisis_lesionados_cali.ipynb          # Notebook principal (4 etapas)
├── informe_tecnico.pdf                     # Informe técnico de posgrado (≤ 6 págs.)
├── informe_tecnico.tex                     # Fuente LaTeX del informe
├── lesionados-en-santiago-de-cali-del-2016-2025.csv   # Dataset crudo
├── dataset_preparado.csv                   # Dataset preparado (51,233 × 14)
├── graficas/                               # 9 figuras generadas (g1..g9)
├── src/                                    # Código fuente modular
│   ├── __init__.py
│   ├── config.py           # rutas y constantes
│   ├── datos.py            # carga, limpieza y preparación
│   ├── entorno.py          # EntornoVial (entorno simulado)
│   ├── agente.py           # AgenteQLearning
│   ├── entrenamiento.py    # entrenamiento y evaluación
│   ├── visualizacion.py    # generación de gráficas
│   └── main.py             # pipeline ejecutable
└── README.md
```

## Requisitos

```bash
pip install pandas numpy matplotlib nbformat jupyter
```

Para generar el PDF del informe se requiere una distribución LaTeX (p. ej. MiKTeX).

## Uso

### Reproducir el pipeline completo (vía `src`)

```bash
python -m src.main
```

Esto ejecuta de extremo a extremo: prepara el dataset, entrena el agente, evalúa y genera
las 9 gráficas en `graficas/`.

### Usar los módulos de forma programática

```python
from src.datos import cargar_crudo, preparar
from src.entorno import EntornoVial
from src.agente import AgenteQLearning
from src.entrenamiento import construir_datos_rl, entrenar

df = preparar(cargar_crudo())
tasa, p_franja, p_dia, p_actor = construir_datos_rl(df)
entorno = EntornoVial(tasa, p_franja, p_dia, p_actor, horizonte=50, seed=0)
agente = AgenteQLearning(entorno.n_estados, entorno.n_acciones)
historial = entrenar(entorno, agente, n_episodios=1500)
```

### Ejecutar el notebook

```bash
jupyter nbconvert --to notebook --execute analisis_lesionados_cali.ipynb
```

## Modelo

| Componente | Definición |
|---|---|
| **Estados** | 32 = 4 franjas horarias × 2 tipos de día × 4 actores viales |
| **Acciones** | 4 (control de motos, vehículos, protección al peatón, infraestructura ciclista) |
| **Recompensa** | `ρ(s,a) = − tasa[actor\|franja] · (1 − eficacia[a, actor]) · peso[actor]` |
| **Algoritmo** | Q-Learning tabular (α=0.1, γ=0.9, ε₀=1.0 → ε_min=0.01, 1,500 episodios, H=50) |

## Resultados principales

Comparación de la política greedy (ε=0) frente a la base fija, bajo costo sin ruido:

| Eficacia | Costo Q-Learning | Costo base fija | Diferencia |
|---|---|---|---|
| 100 % | 0.0000 | 3.9646 | +100.00 % |
| 70 % | 3.6670 | 6.0458 | +39.35 % |
| 50 % | 6.1235 | 7.4332 | +17.62 % |

La ventaja de Q-Learning es **robusta en dirección** (siempre supera a la base fija) pero
**decreciente en magnitud** conforme se reduce la eficacia asumida de las intervenciones.

## Limitaciones

- El entorno es **simulado**: las transiciones entre estados son independientes (el registro
  histórico no contiene datos de transición causal).
- `eficacia` y `pesos` de severidad son **supuestos de diseño**, no estimados de datos.
- `tipo_dia` forma parte del estado pero **no modifica la recompensa**.
- Los datos presentan **subregistro** en las banderas de víctima.
- No hay validación con intervenciones reales ni contrafactuales.

## Referencias

- Watkins, C. J. C. H. (1992). *Learning from delayed rewards* (Tesis doctoral, Universidad de Cambridge).
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2.ª ed.). The MIT Press.
