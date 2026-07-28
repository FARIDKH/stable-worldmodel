"""
MOVER - Roboter/Agent-Klasse aus dem Original-Skript
(auf die vom Env genutzten Methoden reduziert).
"""

import math
import numpy as np


class Mover:
    """
    Repräsentiert einen einzelnen Roboter/Mover in der Simulation.

    Was ein Mover macht:
    - Verwaltet eigene Position und Ziel
    - Liefert Distanz-/Winkel-Features für die Observation
    - Wählt Constraint-Actions (Abstand, Kollisionsvermeidung)
    - Setzt Joint-Geschwindigkeiten in MuJoCo
    """

    def __init__(
        self,
        env,
        mu_index,
        mu_start,
        mu_joint,
        mu_start_move,
        follow,
        max_dist,
        vel,
        cable_connect,
        cable_start_mu,
    ):
        """
        Initialisiert einen Mover.

        Args:
            env: Referenz zum Environment
            mu_index: Body-ID in MuJoCo (91, 99, 105, 110, 115)
            mu_start: Startposition [x, y] in Metern
            mu_joint: Joint-Name Prefix (z.B. "slide_joint1")
            mu_start_move: Initiale Bewegungsrichtung [x, y]
            follow: True wenn dieser Mover Mover 0 folgen soll
            max_dist: Maximaler erlaubter Abstand zu anderen
            vel: Geschwindigkeitsfaktor
            cable_connect: Liste der verbundenen Kabel-IDs
            cable_start_mu: Body-IDs der Kabel-Startpunkte
        """
        # ========== REWARD TRACKING ==========
        self.reward_total = 0
        self.mean_reward = 0
        self.reward_sum = 0
        self.reward = 0
        self.done = False
        self.reward_list = []

        # ========== KOORDINATEN TRACKING ==========
        self.coords_x = []
        self.coords_y = []

        # ========== ENVIRONMENT REFERENZ ==========
        self.env = env

        # ========== MOVER EIGENSCHAFTEN ==========
        self.mu_index = mu_index
        self.mu_start = mu_start
        self.x = mu_start[0]
        self.y = mu_start[1]

        # ========== JOINT KONTROLLE ==========
        mu_joint_x = mu_joint + 'x'
        self.joint_x = mu_joint_x
        mu_joint_y = mu_joint + 'y'
        self.joint_y = mu_joint_y

        # ========== BEWEGUNGSPARAMETER ==========
        self.vel = vel
        self.start_move = mu_start_move
        self.follow = follow
        self.max_dist = max_dist

        # ========== KABEL VERBINDUNGEN ==========
        self.cable_connect = cable_connect
        self.cable_start_mu = cable_start_mu

        # ========== LOKALE COLLISION MAPS ==========
        # Vom Env befüllt (WireHarnessEnv._update_local_maps), hier nur der
        # Speicherplatz; fließen in Observation und Grid-Penalties ein.
        self.mu_collision_map = np.zeros((5, 5))
        self.mu_cable_collision_map = np.zeros((7, 7))

        # ========== ZIEL KOORDINATEN ==========
        self.x_t = 0
        self.y_t = 0

    def update_pos(self):
        """
        Aktualisiert Position aus MuJoCo-Daten.
        Wird jeden Simulationsschritt aufgerufen.
        """
        self.x = self.env.data.xpos[self.mu_index][0]
        self.y = self.env.data.xpos[self.mu_index][1]

    def get_distance(self, x, y, dist_norm=0):
        """
        Berechnet Distanz zu einem Punkt.

        Was hier berechnet wird:
        - Euklidische Distanz mit Pythagoras
        - Optional: Normalisierung auf [-1, 1]

        Args:
            x, y: Zielpunkt
            dist_norm: Normalisierungsfaktor (0 = keine Normalisierung)

        Returns:
            Distanz oder normalisierte Distanz
        """
        dist = math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

        if dist_norm > 0:
            return (dist - dist_norm / 2) / (dist_norm / 2)
        else:
            return dist

    def get_distance_x(self, x):
        """X-Distanz (mit Vorzeichen)"""
        return self.x - x

    def get_distance_y(self, y):
        """Y-Distanz (mit Vorzeichen)"""
        return self.y - y

    def get_distance_target(self, norm=True):
        """
        Distanz zum Ziel.

        Args:
            norm: True für Normalisierung auf [-1, 1]
        """
        dist = math.sqrt((self.x - self.x_t) ** 2 + (self.y - self.y_t) ** 2)

        if norm:
            return (dist - 5 / 2) / (5 / 2)
        else:
            return dist

    def get_angle_target(self, norm=True):
        """Winkel zum Ziel (atan2, vollständiger Winkelbereich [-π, π])"""
        angle = math.atan2((self.y - self.y_t), (self.x - self.x_t))

        if norm:
            return angle / 3.142
        else:
            return angle

    def make_move(self, action):
        """
        Setzt die Geschwindigkeit der Joints.

        Was hier passiert:
        - Multipliziert Action mit Geschwindigkeitsfaktor
        - Setzt Joint-Velocities in MuJoCo

        Args:
            action: [x, y] Bewegungsrichtung (normalisiert)
        """
        self.env.data.joint(self.joint_x).qvel[0] = self.vel * action[0]
        self.env.data.joint(self.joint_y).qvel[0] = self.vel * action[1]

    def set_target(self, x_t, y_t):
        """Setzt neue Zielkoordinaten"""
        self.x_t = x_t
        self.y_t = y_t

    def choose_constraint_action(self, step, dist):
        """
        Wählt Aktion basierend auf Constraints.

        Prioritäten-Reihenfolge:
        1. Follow-Constraint: Abstand zu Mover 0 einhalten
        2. Abstand-Constraint: Max-Abstand zu anderen
        3. Kollisions-Vermeidung: Lokale Hindernisse

        Returns:
            [x, y] Action oder [0, 0] wenn keine Constraint-Action
        """
        # ========== CONSTRAINT 1: FOLLOW ==========
        if self.follow and dist > self.max_dist:
            x_dist = self.get_distance_x(self.env.movers[0].x)
            y_dist = self.get_distance_y(self.env.movers[0].y)
            action = [-x_dist / dist, -y_dist / dist]
            return action

        # ========== CONSTRAINT 2: ABSTAND ==========
        if not self.follow:
            for i in range(self.env.num_agents - 1):
                dist1 = self.get_distance(
                    self.env.movers[i + 1].x, self.env.movers[i + 1].y
                )
                if dist1 > self.env.movers[i + 1].max_dist:
                    x_dist = self.get_distance_x(self.env.movers[i + 1].x)
                    y_dist = self.get_distance_y(self.env.movers[i + 1].y)
                    action = [-x_dist / dist1, -y_dist / dist1]
                    return action

        # ========== CONSTRAINT 3: KOLLISION ==========
        if np.sum(self.mu_collision_map) > 1:
            action = self.collision_avoidance()
            return action

        # Keine Constraint-Action
        return [0, 0]

    def collision_avoidance(self):
        """
        Einfache Kollisionsvermeidung.

        Strategie:
        - Bewege dich weg von der Seite mit mehr Hindernissen
        - Links vs Rechts und Oben vs Unten
        """
        x_dir = (
            0.5
            if np.sum(self.mu_collision_map[:, :1])
            > np.sum(self.mu_collision_map[:, 2:])
            else -0.5
        )
        y_dir = (
            0.5
            if np.sum(self.mu_collision_map[:1, :])
            > np.sum(self.mu_collision_map[2:, :])
            else -0.5
        )

        return [x_dir, y_dir]

    def deterministic_move_t(self):
        """
        Direkte Bewegung zum Ziel.

        Was hier passiert:
        1. Berechnet Richtungsvektor zum Ziel
        2. Normalisiert auf Manhattan-Distanz 0.5
        """
        x_dist = self.get_distance_x(self.x_t)
        y_dist = self.get_distance_y(self.y_t)

        norm = math.sqrt(x_dist**2 + y_dist**2)
        x_dir = -x_dist / norm
        y_dir = -y_dist / norm

        scaling = 0.5 / (abs(x_dir) + abs(y_dir))
        x_scaled = x_dir * scaling
        y_scaled = y_dir * scaling

        return [x_scaled, y_scaled]
