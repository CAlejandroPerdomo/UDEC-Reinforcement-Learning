# -*- coding: utf-8 -*-
"""Generación de las gráficas del proyecto (EDA, entrenamiento y evaluación)."""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DIR_GRAFICAS, asegurar_directorios

# Paleta de colores del proyecto.
AZUL = "#4C72B0"
VERDE = "#55A868"
ROJO = "#C44E52"
MORADO = "#8172B2"
AMARILLO = "#CCB974"


def _guardar(fig, nombre: str) -> None:
    asegurar_directorios()
    ruta = os.path.join(DIR_GRAFICAS, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grafica_por_anio(df: pd.DataFrame, mostrar: bool = False):
    fig, ax = plt.subplots(figsize=(11, 5))
    ser = df["Año"].value_counts().sort_index()
    ser.plot(kind="bar", color=AZUL, edgecolor="black", ax=ax)
    ax.set_title("Lesionados por año (2016–2025)")
    ax.set_xlabel("Año")
    ax.set_ylabel("Nº de registros")
    ax.tick_params(axis="x", rotation=0)
    _guardar(fig, "g1_por_anio.png")
    if mostrar:
        plt.show()
    return ser


def grafica_promedio_mes(df: pd.DataFrame, mostrar: bool = False):
    fig, ax = plt.subplots(figsize=(11, 5))
    mes_anio = df.groupby(["Año", "Mes"]).size().unstack().mean()
    mes_anio.plot(kind="bar", color=VERDE, edgecolor="black", ax=ax)
    ax.set_title("Promedio mensual de lesionados (normalizado por año)")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Promedio de registros")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], rotation=45)
    _guardar(fig, "g2_promedio_mes.png")
    if mostrar:
        plt.show()
    return mes_anio


def grafica_dia_semana(df: pd.DataFrame, mostrar: bool = False):
    fig, ax = plt.subplots(figsize=(11, 5))
    orden = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    ds = df["Dia_Semana"].str.capitalize().value_counts().reindex(orden)
    ds.plot(kind="bar", color=ROJO, edgecolor="black", ax=ax)
    ax.set_title("Lesionados por día de la semana")
    ax.set_xlabel("Día")
    ax.set_ylabel("Nº de registros")
    ax.tick_params(axis="x", rotation=45)
    _guardar(fig, "g3_dia_semana.png")
    if mostrar:
        plt.show()
    return ds


def grafica_por_hora(df: pd.DataFrame, mostrar: bool = False):
    fig, ax = plt.subplots(figsize=(11, 5))
    hr = pd.to_numeric(df["Hora"].astype(str).str.split(":").str[0], errors="coerce").astype("Int64")
    hr = hr.dropna().astype(int).value_counts().sort_index()
    hr.plot(kind="line", marker="o", color=MORADO, ax=ax)
    ax.set_title("Lesionados por hora del día")
    ax.set_xlabel("Hora")
    ax.set_ylabel("Nº de registros")
    ax.set_xticks(range(0, 24))
    _guardar(fig, "g4_por_hora.png")
    if mostrar:
        plt.show()
    return hr


def grafica_tipo_evento(df: pd.DataFrame, mostrar: bool = False):
    fig, ax = plt.subplots(figsize=(11, 5))
    te = df["Tipo_Evento"].value_counts().head(10)
    te.plot(kind="barh", color=AMARILLO, edgecolor="black", ax=ax)
    ax.set_title("Tipo de evento (clase de accidente) — top 10")
    ax.set_xlabel("Nº de registros")
    _guardar(fig, "g5_tipo_evento.png")
    if mostrar:
        plt.show()
    return te


def grafica_victimas_tipo(df: pd.DataFrame, mostrar: bool = False):
    fig, ax = plt.subplots(figsize=(11, 5))
    flags = ["Automovil", "Moto", "Ciclista", "Peaton"]
    vic = df[flags].sum()
    vic.plot(kind="bar", color=[AZUL, VERDE, ROJO, MORADO], edgecolor="black", ax=ax)
    ax.set_title("Accidentes con víctimas por tipo de actor vial")
    ax.set_ylabel("Nº de accidentes con flag activo")
    ax.tick_params(axis="x", rotation=0)
    _guardar(fig, "g6_victimas_tipo.png")
    if mostrar:
        plt.show()
    return vic


def grafica_aprendizaje(hist_recompensa, mostrar: bool = False):
    fig, ax = plt.subplots(figsize=(11, 5))
    media_movil = np.convolve(hist_recompensa, np.ones(50) / 50, mode="valid")
    ax.plot(media_movil, color=AZUL)
    ax.set_title("Curva de aprendizaje — recompensa total por episodio (suavizada)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa total")
    _guardar(fig, "g7_qlearning_aprendizaje.png")
    if mostrar:
        plt.show()


def grafica_sensibilidad(resultados, mostrar: bool = False):
    """Costo estimado por nivel de eficacia (Q-Learning vs base fija)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    niveles = ["100%", "70%", "50%"]
    costo_ql = [resultados[e]["costo_ql"] for e in niveles]
    costo_base = [resultados[e]["costo_base"] for e in niveles]
    x = np.arange(len(niveles))
    w = 0.35
    b1 = ax.bar(x - w / 2, costo_ql, w, label="Q-Learning (re-entrenado)", color=AZUL, edgecolor="black")
    b2 = ax.bar(x + w / 2, costo_base, w, label="Base fija (motos)", color=ROJO, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(niveles)
    ax.set_title("Costo estimado (sin ruido) por nivel de eficacia")
    ax.set_xlabel("Eficacia sobre el actor objetivo")
    ax.set_ylabel("Costo estimado")
    ax.legend()
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    _guardar(fig, "g8_eval_sensibilidad.png")
    if mostrar:
        plt.show()


def grafica_diferencia(resultados, mostrar: bool = False):
    """Diferencia porcentual de costo por nivel de eficacia."""
    fig, ax = plt.subplots(figsize=(11, 5))
    niveles = ["100%", "70%", "50%"]
    diffs = [resultados[e]["diff"] for e in niveles]
    colores = [VERDE if d > 0 else ROJO for d in diffs]
    bars = ax.bar(niveles, diffs, color=colores, edgecolor="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Diferencia porcentual de costo (positivo = Q-Learning mejor)")
    ax.set_xlabel("Eficacia sobre el actor objetivo")
    ax.set_ylabel("Δ% de costo")
    for b, d in zip(bars, diffs):
        ax.annotate(f"{d:+.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom" if d >= 0 else "top", fontsize=10)
    _guardar(fig, "g9_eval_diferencia.png")
    if mostrar:
        plt.show()
