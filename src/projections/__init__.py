"""Projection layers between the rate models and the team rollup.

Station B (playing time) lives here: rate models say *how often* a hitter does
something per plate appearance, this package says *how many plate appearances*
he gets. Everything downstream — team run environment, WAR, contracts —
multiplies one by the other.
"""
from src.projections.playing_time import (
    METHODS,
    project_playing_time,
    realized_pa,
    score_projection,
    team_pa_per_game,
    walk_forward_scores,
    window_pa,
)

__all__ = [
    "METHODS", "project_playing_time", "realized_pa", "score_projection",
    "team_pa_per_game", "walk_forward_scores", "window_pa",
]
