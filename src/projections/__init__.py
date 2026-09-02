"""Projection layers between the rate models and the team rollup.

Station B (playing time) lives here: rate models say *how often* a hitter does
something per plate appearance, this package says *how many plate appearances*
he gets. Everything downstream — team run environment, WAR, contracts —
multiplies one by the other.

`ros` multiplies the two together for the site: the rest-of-season projection
is Marcel fed the partial current season (station A's harness winner) times
station B's projected PA.
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
from src.projections.ros import (
    build_ros_projections,
    league_rates,
    marcel_rates,
    ros_counting_line,
)

__all__ = [
    "METHODS", "project_playing_time", "realized_pa", "score_projection",
    "team_pa_per_game", "walk_forward_scores", "window_pa",
    "build_ros_projections", "league_rates", "marcel_rates", "ros_counting_line",
]
