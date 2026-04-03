"""
Configuration file for the visual navigation system.

This file defines all hyperparameters used across the project, including
localization thresholds, planner behavior, control timing (turn/forward frames),
and goal check-in conditions.

Tuning these values directly affects navigation accuracy, speed, and stability.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class VisNavConfig:
    cache_dir: str = "cache"
    data_dir: str = "data/exploration_data"

    window_w = 1000
    window_h = 700
    camera_scale = 1.5
    target_window_scale = 1.0
    center_display = False
    hud_font_scale_top = 0.38
    hud_font_scale_bottom = 0.36

    # VLAD / graph
    # Tuned for better localization stability on the final maze.
    n_clusters: int = 48
    subsample_rate: int = 1
    top_k_shortcuts: int = 16
    temporal_weight: float = 1.0
    visual_weight_base: float = 2.0
    visual_weight_scale: float = 3.0
    min_shortcut_gap: int = 40

    # Localization
    top_candidates: int = 5
    local_prior_window: int = 28
    global_override_margin: float = 0.18
    sim_ema_alpha: float = 0.35
    node_history: int = 11
    topk_consensus: int = 5
    candidate_prior_penalty: float = 0.003

    # Confidence / planner trust
    low_conf_sim: float = 0.10
    med_conf_sim: float = 0.14
    high_conf_sim: float = 0.18
    min_gap_med: float = 0.006
    min_gap_high: float = 0.015
    conf_hysteresis_margin: float = 0.012
    planner_trust_sim: float = 0.20
    planner_trust_gap: float = 0.015

    # Auto control
    # Semi-auto fast preset:
    # shorter turn commits, longer forward commits
    turn_frames: int = 5
    forward_burst_frames: int = 14
    recovery_turn_frames: int = 10
    recovery_forward_frames: int = 12
    escape_turn_frames: int = 14
    escape_forward_frames: int = 14
    action_stick_frames: int = 4

    # Low-confidence search-forward behavior
    search_turn_frames: int = 7
    search_forward_frames: int = 8
    search_flip_after_failures: int = 1

    # Anti-oscillation rules
    turn_switch_cooldown_frames: int = 18
    min_forward_after_turn_frames: int = 10
    wall_turn_cooldown_frames: int = 14

    # Front-collision guard / wall escape
    wall_escape_turn_frames: int = 18
    wall_stuck_cooldown_frames: int = 16
    front_clear_streak_needed: int = 3
    front_block_brightness: float = 155.0
    front_block_edge_ratio: float = 0.02

    # Loop / revisit suppression
    loop_window: int = 50
    loop_unique_node_threshold: int = 7
    loop_goal_improve_margin: float = 0.010
    loop_escape_turn_frames: int = 18
    loop_escape_forward_frames: int = 14

    # Stagnation / dead-end detection
    stagnation_window: int = 20
    min_progress_nodes: int = 2
    low_improve_margin: float = 0.015
    front_wall_ratio_trigger: float = 0.55

    # Goal / check-in
    max_near_goal_hops: int = 4
    checkin_confirm_frames: int = 2
    checkin_sim: float = 0.10
    goal_node_sim: float = 0.10

    # Wall / openness heuristics
    center_dark_thresh: float = 148.0
    center_edge_thresh: float = 0.10
    openness_margin: float = 3.0

    # Manual goal-assist
    manual_mode_default: bool = True
    assist_goal_best_sim: float = 0.10
    assist_goal_front_sim: float = 0.10
    assist_goal_hold_frames: int = 2
    assist_near_goal_hops: int = 999

    # HUD
    top_banner_h: int = 54
    bottom_banner_h: int = 54
    hud_bg: tuple[int, int, int] = (0, 0, 0)
    hud_text: tuple[int, int, int] = (235, 235, 235)
    hud_gray: tuple[int, int, int] = (185, 185, 185)
    hud_green: tuple[int, int, int] = (80, 255, 80)
    hud_yellow: tuple[int, int, int] = (0, 220, 255)
    hud_red: tuple[int, int, int] = (70, 70, 255)
