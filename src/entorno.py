# -*- coding: utf-8 -*-
"""Entorno simulado ``EntornoVial`` para el control de la siniestralidad vial."""
import numpy as np

# Etiquetas globales compartidas por el resto de módulos.
NOMBRES_FRANJA = ["madrugada", "mañana", "tarde", "noche"]
ACTORES = ["Automovil", "Moto", "Peaton", "Ciclista"]


def franja(hora: int) -> int:
    """Clasifica una hora del día en su franja (0=madrugada, 1=mañana, 2=tarde, 3=noche)."""
    if 0 <= hora <= 5:
        return 0
    if 6 <= hora <= 11:
        return 1
    if 12 <= hora <= 17:
        return 2
    return 3


def build_eficacia(diag: float) -> np.ndarray:
    """Matriz de eficacia respetando la correspondencia acción -> actor objetivo.

    - acción 0 = control motos      -> actor Moto       (col 1)
    - acción 1 = control vehículos  -> actor Automovil  (col 0)
    - acción 2 = protección peatón  -> actor Peaton     (col 2)
    - acción 3 = infra. ciclista    -> actor Ciclista   (col 3)
    """
    E = np.full((4, 4), 0.25)
    E[0, 1] = diag   # control motos -> Moto
    E[1, 0] = diag   # control vehículos -> Automovil
    E[2, 2] = diag   # protección peatón -> Peaton
    E[3, 3] = diag   # infra. ciclista -> Ciclista
    return E


class EntornoVial:
    """Entorno simulado para el control de siniestralidad vial.

    Estado  : (franja_horaria, tipo_dia, actor_prioritario) -> índice 0..31
    Acción  : 0..3 (control motos / vehículos / peatón / ciclista)
    Recompensa: negativa, proporcional al costo de lesionados no mitigados.
    """

    N_FRANJAS = 4
    N_DIAS = 2      # 0 = entre semana, 1 = fin de semana
    N_ACTORES = 4

    # EFICACIA[fila = acción][columna = actor], orden ACTORES = [Automovil, Moto, Peaton, Ciclista].
    EFICACIA = np.array([
        [0.25, 1.00, 0.25, 0.25],   # 0 control motos
        [1.00, 0.25, 0.25, 0.25],   # 1 control vehículos
        [0.25, 0.25, 1.00, 0.25],   # 2 protección peatón
        [0.25, 0.25, 0.25, 1.00],   # 3 infraestructura ciclista
    ])

    # Peso de severidad relativa por actor (Automovil, Moto, Peaton, Ciclista).
    PESOS = np.array([1.0, 1.2, 1.6, 1.5])

    def __init__(self, tasa_por_franja, p_franja, p_dia, p_actor,
                 horizonte=50, ruido=0.02, seed=0, eficacia=None):
        self.tasa = tasa_por_franja          # dict[franja][actor]
        self.H = horizonte
        self.ruido = ruido
        self.rng = np.random.default_rng(seed)
        self.p_franja = np.asarray(p_franja, dtype=float)
        self.p_dia = np.asarray(p_dia, dtype=float)
        self.p_actor = np.asarray(p_actor, dtype=float)
        if eficacia is not None:
            self.EFICACIA = np.asarray(eficacia, dtype=float)
        self._reset_interno()

    # --- Codificación de estados ---------------------------------
    def _estado_a_indice(self, franja, dia, actor):
        return int(franja * self.N_DIAS * self.N_ACTORES + dia * self.N_ACTORES + actor)

    def _indice_a_estado(self, idx):
        actor = idx % self.N_ACTORES
        dia = (idx // self.N_ACTORES) % self.N_DIAS
        franja = idx // (self.N_DIAS * self.N_ACTORES)
        return franja, dia, actor

    def _muestrear_estado(self):
        franja = int(self.rng.choice(self.N_FRANJAS, p=self.p_franja))
        dia = int(self.rng.choice(self.N_DIAS, p=self.p_dia))
        actor = int(self.rng.choice(self.N_ACTORES, p=self.p_actor))
        return self._estado_a_indice(franja, dia, actor)

    def _reset_interno(self):
        self.t = 0
        self.estado = self._muestrear_estado()

    # --- Interfaz de entorno --------------------------------------
    def reset(self):
        self._reset_interno()
        return self.estado

    def recompenza_esperada(self, estado, accion):
        franja, dia, actor = self._indice_a_estado(estado)
        # Costo centrado en el actor víctima del evento actual:
        #   costo = tasa[actor | franja] * peso[actor] * (1 - eficacia[accion, actor])
        t_actor = self.tasa[franja][ACTORES[actor]]
        mitigacion = 1.0 - self.EFICACIA[accion, actor]
        costo = float(t_actor * mitigacion * self.PESOS[actor])
        return -costo

    def step(self, accion):
        r_media = self.recompenza_esperada(self.estado, accion)
        r = r_media + self.rng.normal(0.0, self.ruido)
        # Transición independiente (el dataset no aporta transiciones entre estados).
        self.estado = self._muestrear_estado()
        self.t += 1
        done = self.t >= self.H
        info = {"recompensa_media": r_media}
        return self.estado, float(r), done, info

    # --- Propiedades ----------------------------------------------
    @property
    def n_estados(self):
        return self.N_FRANJAS * self.N_DIAS * self.N_ACTORES

    @property
    def n_acciones(self):
        return 4
