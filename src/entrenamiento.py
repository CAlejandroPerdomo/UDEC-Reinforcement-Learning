# -*- coding: utf-8 -*-
"""Funciones de entrenamiento, evaluación y cálculo de distribuciones empíricas."""
import numpy as np
import pandas as pd

from .agente import AgenteQLearning
from .datos import cargar_crudo, preparar
from .entorno import ACTORES, EntornoVial, build_eficacia, franja


def construir_datos_rl(df: pd.DataFrame):
    """Calcula tasas por franja y distribuciones empíricas del estado desde el dataset.

    Devuelve ``(tasa_por_franja, p_franja, p_dia, p_actor)``.
    """
    hora = pd.to_numeric(df["Hora"].astype(str).str.split(":").str[0], errors="coerce")
    df_rl = df.assign(franja=hora.map(franja))

    tasa_por_franja = {}
    for f in range(4):
        bloque = df_rl[df_rl["franja"] == f]
        n = len(bloque)
        tasa_por_franja[f] = {a: (bloque[a].sum() / n) if n else 0.0 for a in ACTORES}

    p_franja = df_rl["franja"].value_counts(normalize=True).sort_index().values

    def tipo_dia(ds):
        return "fin_semana" if ds in ["Viernes", "Sábado", "Domingo"] else "entre_semana"

    p_dia = (
        df_rl["Dia_Semana"].map(tipo_dia)
        .value_counts(normalize=True)
        .reindex(["entre_semana", "fin_semana"])
        .values
    )

    total_flags = df_rl[ACTORES].sum()
    p_actor = (total_flags / total_flags.sum()).values

    return tasa_por_franja, p_franja, p_dia, p_actor


def entrenar(entorno, agente, n_episodios=1500):
    """Entrena el agente sobre el entorno y devuelve el historial de recompensas."""
    historial = []
    for _ in range(n_episodios):
        s = entorno.reset()
        recompensa_total = 0.0
        for _ in range(entorno.H):
            a = agente.elegir_accion(s)
            s_next, r, done, _ = entorno.step(a)
            agente.actualizar(s, a, r, s_next, done)
            s = s_next
            recompensa_total += r
            if done:
                break
        historial.append(recompensa_total)
        agente.decaer()
    return historial


def entrenar_politica_por_eficacia(tasa, p_franja, p_dia, p_actor, eficacia,
                                   n_episodios=1500, horizonte=50, seed=7):
    """Entrena un agente Q-Learning con una eficacia dada y devuelve su política greedy."""
    entorno_local = EntornoVial(tasa, p_franja, p_dia, p_actor,
                                horizonte=horizonte, ruido=0.02, seed=seed,
                                eficacia=eficacia)
    ag = AgenteQLearning(entorno_local.n_estados, entorno_local.n_acciones,
                         alpha=0.1, gamma=0.9, eps=1.0, eps_min=0.01,
                         eps_decay=0.995, seed=seed)
    entrenar(entorno_local, ag, n_episodios)
    return ag.politica_greedy()


def evaluar_politica(tasa, p_franja, p_dia, p_actor, eficacia, politica_fn,
                     n_episodios=200, horizonte=50, seed=0):
    """Evalúa una política devolviendo la recompensa esperada acumulada por episodio.

    Usa ``info["recompensa_media"]`` (sin ruido) y una semilla fija por episodio para
    que distintas políticas enfrenten la MISMA secuencia de estados.
    """
    recompensas = []
    for ep in range(n_episodios):
        ent = EntornoVial(tasa, p_franja, p_dia, p_actor,
                          horizonte=horizonte, ruido=0.02, seed=seed + ep,
                          eficacia=eficacia)
        s = ent.reset()
        acum = 0.0
        for _ in range(horizonte):
            a = politica_fn(s)
            s_next, r, done, info = ent.step(a)
            acum += info["recompensa_media"]
            s = s_next
            if done:
                break
        recompensas.append(acum)
    return np.array(recompensas)


def politica_base(s):
    """Estrategia base fija: siempre 'control de motos' (acción 0)."""
    return 0


def politica_desde_greedy(greedy_arr):
    """Devuelve una función de política a partir de un arreglo greedy."""
    def f(s):
        return int(greedy_arr[s])
    return f
