"""Kleine, reine Hilfsfunktionen, die sowohl vom quasi-statischen Runner
(runner_ui.py) als auch vom experimentellen Dynamic Runner
(dynamic_runner_ui.py) verwendet werden, um Windzustaende schrittweise in
Richtung eines Zielwerts zu bewegen (Rampen).
"""

import numpy as np


def _angle_delta_deg(current, target):
    return (target - current + 180.0) % 360.0 - 180.0


def _move_towards(current, target, max_delta):
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + np.sign(delta) * max_delta


def _move_angle_towards(current, target, max_delta):
    delta = _angle_delta_deg(current, target)
    if abs(delta) <= max_delta:
        return target % 360.0
    return (current + np.sign(delta) * max_delta) % 360.0
