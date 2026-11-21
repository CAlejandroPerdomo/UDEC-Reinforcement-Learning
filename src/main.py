# -*- coding: utf-8 -*-
"""Pipeline completo y reproducible del proyecto.

Ejecuta de extremo a extremo:
  1. Carga, limpieza y preparación del dataset.
  2. Construcción de las distribuciones empíricas del estado.
  3. Entrenamiento del agente Q-Learning.
  4. Evaluación comparativa (greedy vs. base fija) y análisis de sensibilidad (100/70/50 %).
  5. Generación de gráficas.
"""
import numpy as np

from .agente import AgenteQLearning
from .config import DIR_GRAFICAS, RUTA_SALIDA
from .datos import cargar_crudo, guardar, preparar
from .entorno import EntornoVial, build_eficacia
from .entrenamiento import (
    construir_datos_rl,
    entrenar,
    entrenar_politica_por_eficacia,
    evaluar_politica,
    politica_base,
    politica_desde_greedy,
)
from . import visualizacion as viz

# Hiperparámetros
N_EPISODIOS = 1500
HORIZONTE = 50
ALPHA = 0.1
GAMMA = 0.9
EPS = 1.0
EPS_MIN = 0.01
EPS_DECAY = 0.995

N_EVAL = 200
H_EVAL = 50
SEED_EVAL = 2024
SEED_ENTRENAMIENTO = 0


def ejecutar(mostrar_graficas: bool = False):
    """Ejecuta el pipeline completo y devuelve un dict con los resultados principales."""
    # 1. Datos
    print("1) Preparación del dataset ...")
    df = preparar(cargar_crudo())
    guardar(df, RUTA_SALIDA)
    print(f"   Dataset preparado: {len(df):,} filas × {df.shape[1]} columnas")

    # EDA
    viz.grafica_por_anio(df, mostrar_graficas)
    viz.grafica_promedio_mes(df, mostrar_graficas)
    viz.grafica_dia_semana(df, mostrar_graficas)
    viz.grafica_por_hora(df, mostrar_graficas)
    viz.grafica_tipo_evento(df, mostrar_graficas)
    viz.grafica_victimas_tipo(df, mostrar_graficas)

    # 2. Datos RL
    print("2) Construcción de distribuciones empíricas ...")
    tasa_por_franja, p_franja, p_dia, p_actor = construir_datos_rl(df)

    # 3. Entrenamiento del agente base (eficacia 100 %)
    print("3) Entrenamiento del agente Q-Learning ...")
    entorno = EntornoVial(tasa_por_franja, p_franja, p_dia, p_actor,
                          horizonte=HORIZONTE, seed=SEED_ENTRENAMIENTO)
    agente = AgenteQLearning(entorno.n_estados, entorno.n_acciones,
                             alpha=ALPHA, gamma=GAMMA, eps=EPS, eps_min=EPS_MIN,
                             eps_decay=EPS_DECAY, seed=SEED_ENTRENAMIENTO)
    hist_recompensa = entrenar(entorno, agente, N_EPISODIOS)
    print(f"   Épsilon final: {agente.eps:.4f}")
    print(f"   Recompensa media (últimos 100): {np.mean(hist_recompensa[-100:]):.3f}")

    viz.grafica_aprendizaje(hist_recompensa, mostrar_graficas)

    greedy = agente.politica_greedy()
    politica_ql = politica_desde_greedy(greedy)

    # 4. Evaluación y sensibilidad
    print("4) Evaluación y análisis de sensibilidad ...")
    resultados = {}
    for etiqueta, diag in [("100%", 1.00), ("70%", 0.70), ("50%", 0.50)]:
        eficacia = EntornoVial.EFICACIA.copy() if diag == 1.0 else build_eficacia(diag)
        if etiqueta == "100%":
            pol_ql = politica_ql  # política de la Sección 9
        else:
            g = entrenar_politica_por_eficacia(tasa_por_franja, p_franja, p_dia, p_actor,
                                               eficacia, n_episodios=N_EPISODIOS,
                                               horizonte=HORIZONTE, seed=7)
            pol_ql = politica_desde_greedy(g)

        rec_q = evaluar_politica(tasa_por_franja, p_franja, p_dia, p_actor,
                                 eficacia, pol_ql, N_EVAL, H_EVAL, SEED_EVAL)
        rec_b = evaluar_politica(tasa_por_franja, p_franja, p_dia, p_actor,
                                 eficacia, politica_base, N_EVAL, H_EVAL, SEED_EVAL)
        mq = float(np.mean(rec_q))
        mb = float(np.mean(rec_b))
        cq = -mq
        cb = -mb
        dp = (cb - cq) / cb * 100.0 if cb != 0 else 0.0
        resultados[etiqueta] = {
            "rec_ql": rec_q, "rec_base": rec_b,
            "media_ql": mq, "media_base": mb,
            "std_ql": float(np.std(rec_q)), "std_base": float(np.std(rec_b)),
            "costo_ql": cq, "costo_base": cb, "diff": dp,
        }
        print(f"   Eficacia {etiqueta}: delta {dp:+.2f}%")

    viz.grafica_sensibilidad(resultados, mostrar_graficas)
    viz.grafica_diferencia(resultados, mostrar_graficas)

    print(f"\nGráficas generadas en: {DIR_GRAFICAS}")
    return {
        "df": df,
        "agente": agente,
        "hist_recompensa": hist_recompensa,
        "resultados": resultados,
    }


if __name__ == "__main__":
    ejecutar()
