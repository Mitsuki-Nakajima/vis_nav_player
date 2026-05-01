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
<<<<<<< HEAD

    # VLAD / graph
    # Tuned for better localization stability on the final maze.
    n_clusters: int = 48
    subsample_rate: int = 1
    top_k_shortcuts: int = 16
=======
    game_speed_multiplier: float = 1.6

    # VLAD / graph
    # Tuned for better localization stability on the final maze.
    n_clusters: int = 64
    subsample_rate: int = 1
    top_k_shortcuts: int = 32
>>>>>>> 38b10aa (Update)
    temporal_weight: float = 1.0
    visual_weight_base: float = 2.0
    visual_weight_scale: float = 3.0
    min_shortcut_gap: int = 40

    # Localization
    top_candidates: int = 5
<<<<<<< HEAD
    local_prior_window: int = 28
    global_override_margin: float = 0.18
    sim_ema_alpha: float = 0.35
    node_history: int = 11
    topk_consensus: int = 5
    candidate_prior_penalty: float = 0.003
=======
    local_prior_window: int = 60
    global_override_margin: float = 0.08
    sim_ema_alpha: float = 0.35
    node_history: int = 11
    topk_consensus: int = 5
    candidate_prior_penalty: float = 0.006

    # v10 localization jump suppression
    # Huge mazes have many repeated wall/poster views. These settings prevent
    # LOW/MED-confidence visual matches from teleporting the estimated node far
    # away, which caused hops to jump from ~100 to 9000.
    max_node_jump_low: int = 80
    max_node_jump_med: int = 220
    max_node_jump_high: int = 900
    jump_accept_sim: float = 0.24
    jump_accept_gap: float = 0.030
    jump_search_radius: int = 260
    lost_global_after_frames: int = 18
>>>>>>> 38b10aa (Update)

    # Confidence / planner trust
    low_conf_sim: float = 0.10
    med_conf_sim: float = 0.14
    high_conf_sim: float = 0.18
    min_gap_med: float = 0.006
    min_gap_high: float = 0.015
    conf_hysteresis_margin: float = 0.012
<<<<<<< HEAD
    planner_trust_sim: float = 0.20
    planner_trust_gap: float = 0.015
=======
    planner_trust_sim: float = 0.16
    planner_trust_gap: float = 0.008
>>>>>>> 38b10aa (Update)

    # Auto control
    # Semi-auto fast preset:
    # shorter turn commits, longer forward commits
<<<<<<< HEAD
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
=======
    turn_frames: int = 3
    forward_burst_frames: int = 10
    recovery_turn_frames: int = 10
    recovery_forward_frames: int = 12
    escape_turn_frames: int = 14
    escape_forward_frames: int = 8
    action_stick_frames: int = 4

    # Low-confidence search-forward behavior
    search_turn_frames: int = 8
    search_forward_frames: int = 5
>>>>>>> 38b10aa (Update)
    search_flip_after_failures: int = 1

    # Anti-oscillation rules
    turn_switch_cooldown_frames: int = 18
<<<<<<< HEAD
    min_forward_after_turn_frames: int = 10
=======
    min_forward_after_turn_frames: int = 9
>>>>>>> 38b10aa (Update)
    wall_turn_cooldown_frames: int = 14

    # Front-collision guard / wall escape
    wall_escape_turn_frames: int = 18
    wall_stuck_cooldown_frames: int = 16
    front_clear_streak_needed: int = 3
    front_block_brightness: float = 155.0
    front_block_edge_ratio: float = 0.02
<<<<<<< HEAD
=======
    front_dark_block_brightness: float = 55.0
    side_low_open_trigger: float = 58.0


    # Dead-end / hard-stuck escape
    # When wall escape cannot clear the view, back up first, then do a longer U-turn.
    deadend_backward_frames: int = 8
    deadend_uturn_frames: int = 30
    deadend_forward_frames: int = 9
    hard_stuck_path_window: int = 10

    # v14 physical contact escape: KLT stuck even when wall detector says wall=0.
    contact_escape_stuck_frames: int = 2
    contact_escape_backward_frames: int = 7
    contact_escape_turn_frames: int = 12
    contact_escape_forward_frames: int = 14


    # v8 corner / wall-follow tuning
    # Instead of one huge turn, use small turn + committed forward motion.
    corner_turn_frames: int = 3
    corner_forward_frames: int = 11
    corner_cooldown_frames: int = 12

    # v15 anti-spin / fast goal-distance lookup
    # Trigger earlier and break oscillation with BACKWARD -> one committed turn -> FORWARD.
    spin_break_trace_window: int = 12
    spin_break_turn_threshold: int = 4
    spin_break_flip_threshold: int = 2
    spin_break_max_forward_inside: int = 2
    spin_break_backward_frames: int = 6
    spin_break_turn_frames: int = 16
    spin_break_extra_turn_frames: int = 6
    spin_break_forward_frames: int = 18
    spin_break_cooldown_frames: int = 28
    spin_break_floor_margin: float = 1.0
    max_path_preview_nodes: int = 6
    floor_front_block_score: float = 7.0
    floor_side_open_score: float = 11.0
    floor_deadend_side_score: float = 5.0
>>>>>>> 38b10aa (Update)

    # Loop / revisit suppression
    loop_window: int = 50
    loop_unique_node_threshold: int = 7
    loop_goal_improve_margin: float = 0.010
<<<<<<< HEAD
    loop_escape_turn_frames: int = 18
    loop_escape_forward_frames: int = 14
=======
    loop_escape_turn_frames: int = 22
    loop_escape_forward_frames: int = 8
>>>>>>> 38b10aa (Update)

    # Stagnation / dead-end detection
    stagnation_window: int = 20
    min_progress_nodes: int = 2
    low_improve_margin: float = 0.015
    front_wall_ratio_trigger: float = 0.55

    # Goal / check-in
    max_near_goal_hops: int = 4
<<<<<<< HEAD
    checkin_confirm_frames: int = 2
    checkin_sim: float = 0.10
    goal_node_sim: float = 0.10
=======
    checkin_confirm_frames: int = 3
    checkin_sim: float = 0.12
    goal_node_sim: float = 0.12
>>>>>>> 38b10aa (Update)

    # Wall / openness heuristics
    center_dark_thresh: float = 148.0
    center_edge_thresh: float = 0.10
    openness_margin: float = 3.0

    # Manual goal-assist
<<<<<<< HEAD
    manual_mode_default: bool = True
    assist_goal_best_sim: float = 0.10
    assist_goal_front_sim: float = 0.10
    assist_goal_hold_frames: int = 2
    assist_near_goal_hops: int = 999

=======
    manual_mode_default: bool = False
    assist_goal_best_sim: float = 0.12
    assist_goal_front_sim: float = 0.12
    assist_goal_hold_frames: int = 3
    assist_near_goal_hops: int = 999

    # Shi-Tomasi / KLT motion feedback
    # v17 relies on KLT more strongly than v16. VLAD/planner still provides
    # goal direction, but KLT now acts as the movement judge: if FORWARD is
    # physically moving, keep committing forward; if FORWARD produces almost
    # no image motion, immediately back up and re-approach.
    use_motion_feedback: bool = True
    klt_frame_skip: int = 1        # v17: check every frame, but on a small image
    klt_downscale: float = 0.45    # slightly smaller image keeps every-frame KLT fast
    klt_max_corners: int = 70
    klt_quality_level: float = 0.01
    klt_min_distance: int = 9
    klt_block_size: int = 5
    klt_win_size: int = 13
    klt_max_level: int = 1
    klt_max_iter: int = 10
    klt_min_points: int = 10
    klt_stuck_flow_px: float = 0.65   # measured on downscaled image
    klt_good_flow_px: float = 1.25    # if FORWARD has this much flow, trust motion more than LOW-confidence planner turns
    klt_stuck_frames: int = 2
    klt_ignore_after_turn_frames: int = 2
    klt_forward_commit_frames: int = 8
    klt_planner_override_conf: str = "LOW"

>>>>>>> 38b10aa (Update)
    # HUD
    top_banner_h: int = 54
    bottom_banner_h: int = 54
    hud_bg: tuple[int, int, int] = (0, 0, 0)
    hud_text: tuple[int, int, int] = (235, 235, 235)
    hud_gray: tuple[int, int, int] = (185, 185, 185)
    hud_green: tuple[int, int, int] = (80, 255, 80)
    hud_yellow: tuple[int, int, int] = (0, 220, 255)
    hud_red: tuple[int, int, int] = (70, 70, 255)
