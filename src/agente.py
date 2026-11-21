# -*- coding: utf-8 -*-
"""Agente tabular Q-Learning con política ε-greedy y decaimiento de ε."""
import numpy as np


class AgenteQLearning:
    """Agente Q-Learning tabular (Watkins, 1992)."""

    def __init__(self, n_estados, n_acciones, alpha=0.1, gamma=0.9,
                 eps=1.0, eps_min=0.01, eps_decay=0.995, seed=0):
        self.n_estados = n_estados
        self.n_acciones = n_acciones
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_estados, n_acciones))

    def elegir_accion(self, estado):
        """Política ε-greedy."""
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_acciones))
        return int(np.argmax(self.Q[estado]))

    def actualizar(self, s, a, r, s_next, done):
        """Regla de actualización de Bellman."""
        objetivo = r + (0.0 if done else self.gamma * np.max(self.Q[s_next]))
        self.Q[s, a] += self.alpha * (objetivo - self.Q[s, a])

    def decaer(self):
        """Reduce ε multiplicativamente hasta ε_min."""
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def politica_greedy(self):
        """Política determinista argmax sobre Q."""
        return np.argmax(self.Q, axis=1)
