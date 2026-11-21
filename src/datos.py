# -*- coding: utf-8 -*-
"""Carga, limpieza y preparación del dataset de lesionados de Cali."""
import re

import numpy as np
import pandas as pd

from .config import ENCODING_RAW, RUTA_RAW, SEPARADOR


def cargar_crudo(ruta: str = RUTA_RAW) -> pd.DataFrame:
    """Carga el dataset crudo (Latin-1, separador ';')."""
    return pd.read_csv(ruta, sep=SEPARADOR, encoding=ENCODING_RAW)


def es_mmdd(cad: str) -> bool:
    """Detecta fechas en formato mm/dd/yyyy (primer componente <= 12 y segundo > 12)."""
    try:
        p = cad.split("/")
        return int(p[0]) <= 12 and int(p[1]) > 12
    except Exception:
        return False


def normalizar_12h(cad: str) -> str:
    """Normaliza hora en formato 12h (sufijo ``a.``/``p.``) a 24h ``HH:MM[:SS]``."""
    m = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?\s*([ap])\.", cad.strip(), re.IGNORECASE)
    if not m:
        return cad
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = m.group(3)
    mer = m.group(4).lower()
    if mer == "p" and hh != 12:
        hh += 12
    if mer == "a" and hh == 12:
        hh = 0
    hh %= 24
    return f"{hh:02d}:{mm:02d}" + (f":{ss}" if ss else "")


def fraccion_a_hora(cad: str):
    """Convierte una fracción de día de Excel (``0.5``/``0,5``) a ``datetime.time``."""
    try:
        frac = float(cad.replace(",", "."))
        total_seg = round(frac * 24 * 3600)
        hh = (total_seg // 3600) % 24
        mm = (total_seg % 3600) // 60
        return pd.Timestamp(f"2000-01-01 {hh:02d}:{mm:02d}:00")
    except Exception:
        return pd.NaT


def limpiar_basura(cad: str) -> str:
    """Extrae el patrón HH:MM(:SS) con sufijo opcional a./p. de un texto con ruido."""
    m = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?\s*([ap])?\.", cad.strip(), re.IGNORECASE)
    if not m:
        return cad
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = m.group(3)
    mer = (m.group(4) or "").lower()
    if mer == "p" and hh != 12:
        hh += 12
    if mer == "a" and hh == 12:
        hh = 0
    hh %= 24
    return f"{hh:02d}:{mm:02d}" + (f":{ss}" if ss else "")


def parsear_fecha(serie: pd.Series) -> pd.Series:
    """Parseo de fechas con doble formato dd/mm/yyyy (dominante) y mm/dd/yyyy."""
    fecha_raw = serie.astype(str)
    mask_mmdd = fecha_raw.apply(es_mmdd)
    fecha = pd.to_datetime(fecha_raw, format="%d/%m/%Y", errors="coerce")
    fecha_mmdd = pd.to_datetime(fecha_raw[mask_mmdd], format="%m/%d/%Y", errors="coerce")
    fecha.loc[mask_mmdd] = fecha_mmdd.values
    return fecha


def parsear_hora(serie: pd.Series) -> pd.Series:
    """Parseo de horas con 5 formatos coexistentes a ``datetime.time``."""
    hora_raw = serie.astype(str)
    hora = pd.to_datetime(hora_raw, format="%H:%M", errors="coerce")

    # (a) con segundos
    mask_ss = hora_raw.str.count(":") == 2
    hora.loc[mask_ss] = pd.to_datetime(hora_raw[mask_ss], format="%H:%M:%S", errors="coerce").values

    # (b) 12h a./p.
    aun_nan = hora.isna()
    mask_12h = aun_nan & hora_raw.str.match(r"\d{1,2}:\d{2}(:\d{2})?\s*[ap]\.", case=False, na=False)
    hora.loc[mask_12h] = pd.to_datetime(hora_raw[mask_12h].apply(normalizar_12h), format="%H:%M:%S", errors="coerce").values

    # (c) fracción Excel
    aun_nan = hora.isna()
    mask_num = aun_nan & hora_raw.str.match(r"^[0-9]+[.,][0-9]+$", na=False)
    hora.loc[mask_num] = hora_raw[mask_num].apply(fraccion_a_hora).values

    # (d) zona horaria -05
    aun_nan = hora.isna()
    mask_tz = aun_nan & hora_raw.str.match(r"\d{1,2}:\d{2}:\d{2}-05", na=False)
    hora.loc[mask_tz] = pd.to_datetime(hora_raw[mask_tz].str.replace(r"-05$", "", regex=True), format="%H:%M:%S", errors="coerce").values

    # (e) ruido inicial
    aun_nan = hora.isna()
    hora.loc[aun_nan] = pd.to_datetime(hora_raw[aun_nan].apply(limpiar_basura), format="%H:%M:%S", errors="coerce").values

    return hora.dt.time


def preparar(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza completa y devuelve el dataset preparado."""
    df = df_raw.copy()

    # Renombrar y eliminar columna redundante
    df = df.rename(columns={"Dirección_Reporte": "Direccion", "Tipo_Confirmado.1": "Tipo_Evento"})
    df = df.drop(columns=["Tipo_Confirmado"])

    # Marcador '.' -> NaN
    df = df.replace(".", np.nan)

    # Fechas y horas
    df["Fecha"] = parsear_fecha(df["Fecha"])
    df["Hora"] = parsear_hora(df["Hora"])

    # Banderas de víctimas a 0/1 y total
    flags = ["Automovil", "Moto", "Ciclista", "Peaton"]
    for c in flags:
        df[c] = df[c].astype(float).fillna(0).astype(int)
    df["Total_Victimas"] = df[flags].sum(axis=1)

    # Variables temporales derivadas
    df["Año"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.month
    df["Dia"] = df["Fecha"].dt.day
    df["Dia_Semana"] = df["Fecha"].dt.day_name("spanish")

    # Duplicados y orden
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sort_values(["Fecha", "Hora"], na_position="last").reset_index(drop=True)

    # Tipos finales
    df["Placa Agente de Transito"] = df["Placa Agente de Transito"].astype(str).str.strip()

    return df


def guardar(df: pd.DataFrame, ruta: str) -> None:
    """Guarda el dataset preparado en UTF-8 (con BOM) y separador ';'."""
    df.to_csv(ruta, sep=SEPARADOR, index=False, encoding="utf-8-sig")
