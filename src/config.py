# -*- coding: utf-8 -*-
"""Configuración y rutas del proyecto."""
import os

# Rutas relativas a la raíz del proyecto (donde está `main.py` y `src/`).
RUTA_RAW = "lesionados-en-santiago-de-cali-del-2016-2025.csv"
RUTA_SALIDA = "dataset_preparado.csv"
DIR_GRAFICAS = "graficas"

# Codificación y formato de los archivos
ENCODING_RAW = "latin1"
SEPARADOR = ";"


def asegurar_directorios():
    """Crea el directorio de salida de gráficas si no existe."""
    os.makedirs(DIR_GRAFICAS, exist_ok=True)
    return DIR_GRAFICAS
