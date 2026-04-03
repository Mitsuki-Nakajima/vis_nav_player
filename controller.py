"""
Navigation controller.

This module decides the robot's next action (forward, left, right, etc.)
based on localization results, planner output, and environment conditions.

It includes logic for:
- following the planned path
- handling low-confidence situations
- avoiding walls and getting unstuck
- detecting when to check in at the goal
"""

from collections import deque
from dataclasses import dataclass
from vis_nav_game import Action

from config import VisNavConfig
from localizer import LocalizationState

@dataclass
class ControlState:
    auto_action: Action = Action.IDLE
    status_text: str = "Navigation not started"
    turn_lock_dir: Action = Action.IDLE
    turn_lock_left: int = 0
    forward_burst_left: int = 0
    post_turn_forward_left: int = 0
    checkin_ready_count: int = 0
    low_conf_streak: int = 0
    failed_recovery_count: int = 0
    escape_mode: bool = False
    escape_dir: Action = Action.LEFT
    escape_turn_left: int = 0
    escape_forward_left: int = 0
    search_mode: bool = False
    search_dir: Action = Action.LEFT
    search_turn_left: int = 0
    search_forward_left: int = 0
    search_fail_count: int = 0
    loop_escape_mode: bool = False
    loop_escape_dir: Action = Action.LEFT
    loop_escape_turn_left: int = 0
    loop_escape_forward_left: int = 0
    loop_escape_flip_used: bool = False
    last_turn_dir: Action = Action.IDLE
    last_auto_action: Action = Action.IDLE
    last_wall_turn: Action = Action.LEFT
    turn_switch_cooldown_left: int = 0
    wall_stuck_cooldown: int = 0
    wall_escape_mode: bool = False
    wall_escape_dir: Action = Action.LEFT
    wall_escape_frames_left: int = 0
    front_clear_streak: int = 0
    wall_escape_flip_used: bool = False
    forward_preference_left: int = 0
    goal_seen_streak: int = 0
    forward_progress_streak: int = 0
    reroute_mode: bool = False
    reroute_dir: Action = Action.LEFT
    reroute_turn_left: int = 0
    reroute_forward_left: int = 0


class AutoController:
    def __init__(self, cfg: VisNavConfig):
        self.cfg = cfg
        self.state = ControlState()
        self.path_len_hist = deque(maxlen=cfg.stagnation_window)
        self.recent_nodes = deque(maxlen=cfg.loop_window)
        self.goal_trace = deque(maxlen=cfg.loop_window)
        self.sim_trace = deque(maxlen=max(8, cfg.stagnation_window))
        self.path_trace = deque(maxlen=max(8, cfg.stagnation_window))

    def reset(self):
        self.state = ControlState()
        self.path_len_hist.clear()
        self.recent_nodes.clear()
        self.goal_trace.clear()
        self.sim_trace.clear()
        self.path_trace.clear()

    @staticmethod
    def _is_turn(a: Action) -> bool:
        return a in (Action.LEFT, Action.RIGHT)

    @staticmethod
    def _is_opposite_turn(a: Action, b: Action) -> bool:
        return (a == Action.LEFT and b == Action.RIGHT) or (a == Action.RIGHT and b == Action.LEFT)

    def _top_gap(self, loc: LocalizationState) -> float:
        scores = loc.top_match_scores or []
        if len(scores) >= 2:
            return float(scores[0] - scores[1])
        return 0.0

    def _planner_trusted(self, loc: LocalizationState) -> bool:
        sim = 0.0 if loc.current_sim is None else float(loc.current_sim)
        gap = self._top_gap(loc)
        return (
            loc.conf_text == "HIGH"
            and sim >= self.cfg.planner_trust_sim
            and gap >= self.cfg.planner_trust_gap
        )

    def _blocked_ahead(self, loc: LocalizationState) -> bool:
        if loc.front_wall:
            return True
        if loc.gray_mean >= self.cfg.front_block_brightness and loc.edge_ratio <= self.cfg.front_block_edge_ratio:
            return True
        return False

    def _goal_visible(self, loc: LocalizationState) -> bool:
        return (
            loc.goal_best_sim >= min(self.cfg.checkin_sim * 0.8, 0.12)
            or loc.goal_front_sim >= min(self.cfg.goal_node_sim * 0.8, 0.12)
        )

    def _goal_strong(self, loc: LocalizationState) -> bool:
        return (
            loc.goal_best_sim >= self.cfg.checkin_sim
            and loc.goal_front_sim >= self.cfg.goal_node_sim
        )

    def choose_recovery_turn(
        self,
        loc: LocalizationState,
        plan_action: Action = Action.IDLE,
        keep_dir: Action = Action.LEFT,
    ) -> Action:
        s = self.state
        if plan_action == Action.LEFT:
            return Action.LEFT
        if plan_action == Action.RIGHT:
            return Action.RIGHT

        side_margin = self.cfg.openness_margin
        if loc.left_open > loc.right_open + side_margin:
            return Action.LEFT
        if loc.right_open > loc.left_open + side_margin:
            return Action.RIGHT

        if keep_dir != Action.IDLE:
            return keep_dir
        if s.last_turn_dir != Action.IDLE:
            return s.last_turn_dir
        return Action.LEFT

    def _choose_turn_away_from_wall(self, loc: LocalizationState, seeded: Action = Action.IDLE) -> Action:
        s = self.state
        keep = s.last_wall_turn if s.last_wall_turn != Action.IDLE else s.last_turn_dir
        base = self.choose_recovery_turn(loc, seeded, keep if keep != Action.IDLE else Action.LEFT)
        return self._guard_turn_flip(base, loc)

    def _guard_turn_flip(self, desired: Action, loc: LocalizationState) -> Action:
        s = self.state
        if not self._is_turn(desired):
            return desired
        if s.last_turn_dir == Action.IDLE:
            return desired
        if not self._is_opposite_turn(desired, s.last_turn_dir):
            return desired
        if s.turn_switch_cooldown_left <= 0:
            return desired
        if loc.front_wall:
            return desired
        # When the planner/localizer is noisy, avoid immediate left-right-left-right ping-pong.
        return s.last_turn_dir

    def _start_turn(
        self,
        direction: Action,
        status_text: str,
        lock_frames: int | None = None,
        post_forward_frames: int | None = None,
        cooldown_frames: int | None = None,
    ) -> Action:
        s = self.state
        lock = self.cfg.turn_frames if lock_frames is None else max(1, lock_frames)
        post = self.cfg.min_forward_after_turn_frames if post_forward_frames is None else max(0, post_forward_frames)
        cooldown = self.cfg.turn_switch_cooldown_frames if cooldown_frames is None else max(0, cooldown_frames)
        s.turn_lock_dir = direction
        s.turn_lock_left = max(0, lock - 1)
        s.post_turn_forward_left = max(s.post_turn_forward_left, post)
        s.last_turn_dir = direction
        s.turn_switch_cooldown_left = cooldown
        s.status_text = status_text
        s.auto_action = direction
        s.last_auto_action = direction
        return direction

    def _tick_cooldown(self):
        s = self.state
        if s.turn_switch_cooldown_left > 0:
            s.turn_switch_cooldown_left -= 1
        if s.wall_stuck_cooldown > 0:
            s.wall_stuck_cooldown -= 1
        if s.forward_preference_left > 0:
            s.forward_preference_left -= 1

    def _clear_search(self):
        s = self.state
        s.search_mode = False
        s.search_turn_left = 0
        s.search_forward_left = 0

    def _cancel_forward_modes(self):
        s = self.state
        s.forward_burst_left = 0
        s.post_turn_forward_left = 0
        s.escape_forward_left = 0
        s.search_forward_left = 0
        s.loop_escape_forward_left = 0

    def _record_visit(self, loc: LocalizationState, path_len: int) -> None:
        if loc.current_node is not None:
            self.recent_nodes.append(int(loc.current_node))
        self.goal_trace.append(float(loc.goal_front_sim))
        self.sim_trace.append(0.0 if loc.current_sim is None else float(loc.current_sim))
        self.path_trace.append(path_len)

    def _is_looping(self) -> bool:
        if len(self.recent_nodes) < max(18, self.cfg.loop_window // 2):
            return False
        uniq = len(set(self.recent_nodes))
        if uniq > self.cfg.loop_unique_node_threshold:
            return False
        if len(self.goal_trace) >= 8:
            gain = max(self.goal_trace) - min(self.goal_trace)
            if gain > self.cfg.loop_goal_improve_margin:
                return False
        return True

    def _preferred_loop_dir(self, loc: LocalizationState) -> Action:
        s = self.state
        base = self.choose_recovery_turn(
            loc,
            Action.IDLE,
            s.loop_escape_dir if s.loop_escape_dir != Action.IDLE else (
                s.last_turn_dir if s.last_turn_dir != Action.IDLE else Action.LEFT
            ),
        )
        if not s.loop_escape_flip_used and len(self.recent_nodes) >= 10:
            if s.last_turn_dir == base and len(set(self.recent_nodes)) <= max(3, self.cfg.loop_unique_node_threshold - 2):
                base = Action.RIGHT if base == Action.LEFT else Action.LEFT
        return self._guard_turn_flip(base, loc)

    def _enter_loop_escape(self, loc: LocalizationState) -> None:
        s = self.state
        self._cancel_forward_modes()
        self._clear_search()
        s.escape_mode = False
        s.loop_escape_mode = True
        s.loop_escape_dir = self._preferred_loop_dir(loc)
        s.loop_escape_turn_left = self.cfg.loop_escape_turn_frames
        s.loop_escape_forward_left = self.cfg.loop_escape_forward_frames
        s.loop_escape_flip_used = False

    def _run_loop_escape(self, loc: LocalizationState) -> Action:
        s = self.state
        if self._blocked_ahead(loc):
            self._enter_wall_escape(loc, s.loop_escape_dir)
            return self._run_wall_escape(loc)
        if s.loop_escape_turn_left > 0:
            s.loop_escape_turn_left -= 1
            return self._start_turn(
                s.loop_escape_dir,
                f"LOOP-ESCAPE TURN {s.loop_escape_dir.name}",
                lock_frames=1,
                post_forward_frames=0,
                cooldown_frames=self.cfg.wall_turn_cooldown_frames,
            )
        if s.loop_escape_forward_left > 0:
            s.loop_escape_forward_left -= 1
            s.status_text = "LOOP-ESCAPE FORWARD"
            s.auto_action = Action.FORWARD
            s.last_auto_action = Action.FORWARD
            return s.auto_action
        if not s.loop_escape_flip_used:
            s.loop_escape_flip_used = True
            s.loop_escape_dir = Action.RIGHT if s.loop_escape_dir == Action.LEFT else Action.LEFT
            s.loop_escape_turn_left = max(1, self.cfg.loop_escape_turn_frames // 2)
            s.loop_escape_forward_left = max(1, self.cfg.loop_escape_forward_frames // 2)
            return self._start_turn(
                s.loop_escape_dir,
                f"LOOP-ESCAPE FLIP {s.loop_escape_dir.name}",
                lock_frames=1,
                post_forward_frames=0,
                cooldown_frames=self.cfg.wall_turn_cooldown_frames,
            )
        s.loop_escape_mode = False
        for _ in range(min(len(self.recent_nodes), self.cfg.loop_window // 3)):
            self.recent_nodes.popleft()
        return self._start_turn(
            s.loop_escape_dir,
            f"LOOP-ESCAPE EXIT {s.loop_escape_dir.name}",
            lock_frames=self.cfg.turn_frames,
            post_forward_frames=self.cfg.min_forward_after_turn_frames,
            cooldown_frames=self.cfg.wall_turn_cooldown_frames,
        )

    def _enter_wall_escape(self, loc: LocalizationState, seeded_action: Action = Action.IDLE) -> None:
        s = self.state
        self._cancel_forward_modes()
        self._clear_search()
        s.escape_mode = False
        s.loop_escape_mode = False
        s.wall_escape_mode = True
        s.wall_escape_dir = self._choose_turn_away_from_wall(loc, seeded_action)
        s.last_wall_turn = s.wall_escape_dir
        s.wall_escape_frames_left = self.cfg.wall_escape_turn_frames
        s.front_clear_streak = 0
        s.wall_escape_flip_used = False
        s.wall_stuck_cooldown = self.cfg.wall_stuck_cooldown_frames

    def _run_wall_escape(self, loc: LocalizationState) -> Action:
        s = self.state
        blocked = self._blocked_ahead(loc)
        if blocked:
            s.front_clear_streak = 0
        else:
            s.front_clear_streak += 1

        if s.front_clear_streak >= self.cfg.front_clear_streak_needed:
            s.wall_escape_mode = False
            s.wall_escape_frames_left = 0
            s.wall_stuck_cooldown = max(s.wall_stuck_cooldown, self.cfg.wall_stuck_cooldown_frames // 2)
            s.post_turn_forward_left = max(s.post_turn_forward_left, self.cfg.min_forward_after_turn_frames)
            s.forward_preference_left = max(s.forward_preference_left, self.cfg.min_forward_after_turn_frames)
            s.status_text = "WALL-ESCAPE CLEAR -> FORWARD"
            s.auto_action = Action.FORWARD
            s.last_auto_action = Action.FORWARD
            return s.auto_action

        if s.wall_escape_frames_left > 0:
            s.wall_escape_frames_left -= 1
            return self._start_turn(
                s.wall_escape_dir,
                f"WALL-ESCAPE TURN {s.wall_escape_dir.name}",
                lock_frames=1,
                post_forward_frames=0,
                cooldown_frames=self.cfg.wall_turn_cooldown_frames,
            )

        if not s.wall_escape_flip_used:
            s.wall_escape_flip_used = True
            s.wall_escape_dir = Action.RIGHT if s.wall_escape_dir == Action.LEFT else Action.LEFT
            s.last_wall_turn = s.wall_escape_dir
            s.wall_escape_frames_left = max(1, self.cfg.wall_escape_turn_frames // 2)
            s.front_clear_streak = 0
            return self._start_turn(
                s.wall_escape_dir,
                f"WALL-ESCAPE FLIP {s.wall_escape_dir.name}",
                lock_frames=1,
                post_forward_frames=0,
                cooldown_frames=self.cfg.wall_turn_cooldown_frames,
            )

        s.wall_escape_frames_left = max(1, self.cfg.wall_escape_turn_frames // 2)
        s.front_clear_streak = 0
        return self._start_turn(
            s.wall_escape_dir,
            f"WALL-ESCAPE RETRY {s.wall_escape_dir.name}",
            lock_frames=1,
            post_forward_frames=0,
            cooldown_frames=self.cfg.wall_turn_cooldown_frames,
        )

    def _begin_search(self, loc: LocalizationState, seeded_action: Action = Action.IDLE):
        s = self.state
        keep = s.search_dir if s.search_dir != Action.IDLE else (
            s.last_turn_dir if s.last_turn_dir != Action.IDLE else Action.LEFT
        )
        chosen = self.choose_recovery_turn(loc, seeded_action, keep)
        if s.search_fail_count > self.cfg.search_flip_after_failures:
            chosen = Action.RIGHT if chosen == Action.LEFT else Action.LEFT
            s.search_fail_count = 0
        chosen = self._guard_turn_flip(chosen, loc)
        s.search_mode = True
        s.search_dir = chosen
        s.search_turn_left = self.cfg.search_turn_frames
        s.search_forward_left = self.cfg.search_forward_frames

    def _run_search_cycle(self, loc: LocalizationState, status_prefix: str) -> Action:
        s = self.state
        if s.search_turn_left > 0:
            s.search_turn_left -= 1
            return self._start_turn(
                s.search_dir,
                f"{status_prefix} TURN {s.search_dir.name}",
                lock_frames=1,
                post_forward_frames=0,
            )
        if self._blocked_ahead(loc):
            self._enter_wall_escape(loc, s.search_dir)
            return self._run_wall_escape(loc)
        if s.search_forward_left > 0:
            s.search_forward_left -= 1
            s.status_text = f"{status_prefix} FORWARD"
            s.auto_action = Action.FORWARD
            s.last_auto_action = Action.FORWARD
            return s.auto_action
        s.search_fail_count += 1
        self._clear_search()
        if s.search_fail_count >= 2:
            self._enter_reroute(loc, s.search_dir)
            return self._run_reroute(loc)
        self._begin_search(loc, Action.IDLE)
        s.status_text = f"{status_prefix} RETRY {s.search_dir.name}"
        return self._start_turn(s.search_dir, s.status_text, lock_frames=1, post_forward_frames=0)

    def _enter_reroute(self, loc: LocalizationState, seeded_action: Action = Action.IDLE) -> None:
        s = self.state
        self._cancel_forward_modes()
        self._clear_search()
        s.escape_mode = False
        s.loop_escape_mode = False
        s.wall_escape_mode = False
        s.reroute_mode = True
        keep = s.last_wall_turn if s.last_wall_turn != Action.IDLE else s.last_turn_dir
        chosen = self.choose_recovery_turn(loc, seeded_action, keep if keep != Action.IDLE else Action.LEFT)
        if s.last_turn_dir == chosen and s.search_fail_count >= 2:
            chosen = Action.RIGHT if chosen == Action.LEFT else Action.LEFT
        s.reroute_dir = self._guard_turn_flip(chosen, loc)
        s.reroute_turn_left = max(self.cfg.escape_turn_frames, self.cfg.search_turn_frames + 6)
        s.reroute_forward_left = max(self.cfg.escape_forward_frames, self.cfg.search_forward_frames + 8)
        s.front_clear_streak = 0

    def _run_reroute(self, loc: LocalizationState) -> Action:
        s = self.state
        if s.reroute_turn_left > 0:
            s.reroute_turn_left -= 1
            return self._start_turn(
                s.reroute_dir,
                f"REROUTE TURN {s.reroute_dir.name}",
                lock_frames=1,
                post_forward_frames=0,
                cooldown_frames=self.cfg.wall_turn_cooldown_frames,
            )
        if self._blocked_ahead(loc):
            self._enter_wall_escape(loc, s.reroute_dir)
            return self._run_wall_escape(loc)
        if s.reroute_forward_left > 0:
            s.reroute_forward_left -= 1
            s.status_text = "REROUTE FORWARD"
            s.auto_action = Action.FORWARD
            s.last_auto_action = Action.FORWARD
            return s.auto_action
        s.reroute_mode = False
        s.search_fail_count = 0
        s.forward_preference_left = max(s.forward_preference_left, self.cfg.forward_burst_frames)
        s.status_text = "REROUTE DONE"
        s.auto_action = Action.FORWARD
        s.last_auto_action = Action.FORWARD
        return s.auto_action

    def _progress_stalled(self) -> bool:
        if len(self.path_trace) < 6 or len(self.sim_trace) < 6:
            return False
        recent_paths = list(self.path_trace)[-6:]
        recent_sims = list(self.sim_trace)[-6:]
        path_best = min(recent_paths)
        path_now = recent_paths[-1]
        sim_gain = max(recent_sims) - min(recent_sims)
        return path_now >= path_best and sim_gain < 0.01

    def _recent_progress_good(self) -> bool:
        if len(self.goal_trace) < 4:
            return False
        goal_gain = self.goal_trace[-1] - min(list(self.goal_trace)[-4:])
        if goal_gain > 0.01:
            return True
        if len(self.path_trace) >= 4 and self.path_trace[-1] <= min(list(self.path_trace)[-4:]):
            return True
        return False

    def _choose_greedy_action(self, loc: LocalizationState, plan_action: Action, planner_trusted: bool) -> Action:
        s = self.state
        blocked_ahead = self._blocked_ahead(loc)
        goal_visible = self._goal_visible(loc)
        sim = 0.0 if loc.current_sim is None else float(loc.current_sim)

        # Strong direct goal evidence: keep advancing unless the front is blocked.
        if goal_visible and not blocked_ahead:
            s.goal_seen_streak += 1
        else:
            s.goal_seen_streak = 0

        if goal_visible and not blocked_ahead:
            if self._recent_progress_good() or s.goal_seen_streak >= 2:
                s.forward_preference_left = max(s.forward_preference_left, self.cfg.forward_burst_frames)
                s.status_text = "GREEDY GOAL FORWARD"
                s.auto_action = Action.FORWARD
                s.last_auto_action = Action.FORWARD
                return s.auto_action

        # If confidence is only medium/low, prefer continuing forward for a bit rather than spinning.
        if not blocked_ahead and s.forward_preference_left > 0:
            s.status_text = "MOMENTUM FORWARD"
            s.auto_action = Action.FORWARD
            s.last_auto_action = Action.FORWARD
            return s.auto_action

        # Trusted planner: follow it, but suppress sudden flip-flops.
        if planner_trusted:
            if plan_action in (Action.LEFT, Action.RIGHT):
                guarded = self._guard_turn_flip(plan_action, loc)
                if guarded != plan_action and not blocked_ahead:
                    s.forward_preference_left = max(s.forward_preference_left, self.cfg.min_forward_after_turn_frames // 2)
                    s.status_text = "ANTI-FLIP FORWARD"
                    s.auto_action = Action.FORWARD
                    s.last_auto_action = Action.FORWARD
                    return s.auto_action
                return self._start_turn(
                    guarded,
                    f"PLAN TURN {guarded.name}",
                    lock_frames=self.cfg.turn_frames,
                    post_forward_frames=self.cfg.min_forward_after_turn_frames,
                    cooldown_frames=max(self.cfg.turn_switch_cooldown_frames, self.cfg.min_forward_after_turn_frames),
                )
            if plan_action == Action.FORWARD and not blocked_ahead:
                s.forward_preference_left = max(s.forward_preference_left, self.cfg.forward_burst_frames)
                s.status_text = "PLAN FORWARD"
                s.auto_action = Action.FORWARD
                s.last_auto_action = Action.FORWARD
                return s.auto_action

        # use side openness and last good turn, but do not backtrack aggressively.
        if blocked_ahead:
            self._enter_wall_escape(loc, plan_action if plan_action in (Action.LEFT, Action.RIGHT) else Action.IDLE)
            return self._run_wall_escape(loc)

        # If both sides are similar and we are not blocked, go forward to maximize exploration progress.
        if abs(loc.left_open - loc.right_open) <= self.cfg.openness_margin + 2.0:
            s.forward_preference_left = max(s.forward_preference_left, self.cfg.forward_burst_frames // 2)
            s.status_text = "OPEN-CORRIDOR FORWARD"
            s.auto_action = Action.FORWARD
            s.last_auto_action = Action.FORWARD
            return s.auto_action

        turn_dir = self.choose_recovery_turn(
            loc,
            plan_action if plan_action in (Action.LEFT, Action.RIGHT) else Action.IDLE,
            s.last_turn_dir if s.last_turn_dir != Action.IDLE else s.last_wall_turn,
        )
        return self._start_turn(
            turn_dir,
            f"GREEDY TURN {turn_dir.name}",
            lock_frames=max(1, self.cfg.turn_frames - 2),
            post_forward_frames=self.cfg.min_forward_after_turn_frames,
            cooldown_frames=self.cfg.turn_switch_cooldown_frames,
        )

    def compute_action(self, loc: LocalizationState, plan_action: Action, near_goal: bool, stagnating: bool) -> Action:
        s = self.state
        self._tick_cooldown()

        path_len = len(loc.current_path) - 1 if loc.current_path else 999
        self.path_len_hist.append(path_len)
        self._record_visit(loc, path_len)

        planner_trusted = self._planner_trusted(loc)
        blocked_ahead = self._blocked_ahead(loc)
        looping = self._is_looping()

        if loc.conf_text == "LOW":
            s.low_conf_streak += 1
        else:
            s.low_conf_streak = 0
            s.failed_recovery_count = 0

        # Fully-auto needs direct visual check-in too, not only trusted planner state.
        if self._goal_strong(loc) and (near_goal or self._goal_visible(loc)):
            s.checkin_ready_count += 1
        else:
            s.checkin_ready_count = 0
        if s.checkin_ready_count >= self.cfg.checkin_confirm_frames:
            s.status_text = "CHECKIN"
            s.auto_action = Action.CHECKIN
            s.last_auto_action = Action.CHECKIN
            return s.auto_action

        if s.wall_escape_mode:
            return self._run_wall_escape(loc)

        if s.reroute_mode:
            return self._run_reroute(loc)

        if blocked_ahead and (
            s.post_turn_forward_left > 0
            or s.forward_burst_left > 0
            or s.escape_forward_left > 0
            or s.search_forward_left > 0
            or s.loop_escape_forward_left > 0
        ):
            self._enter_wall_escape(loc, plan_action if plan_action in (Action.LEFT, Action.RIGHT) else Action.IDLE)
            return self._run_wall_escape(loc)

        if s.wall_stuck_cooldown > 0 and blocked_ahead and not s.turn_lock_left:
            self._enter_wall_escape(loc, s.last_wall_turn)
            return self._run_wall_escape(loc)

        if s.loop_escape_mode:
            return self._run_loop_escape(loc)

        if looping and not planner_trusted and not self._goal_visible(loc):
            if s.search_fail_count >= 2 or s.low_conf_streak >= max(10, self.cfg.stagnation_window // 2):
                self._enter_reroute(loc, plan_action if plan_action in (Action.LEFT, Action.RIGHT) else Action.IDLE)
                return self._run_reroute(loc)
            self._enter_loop_escape(loc)
            return self._run_loop_escape(loc)

        if (stagnating or self._progress_stalled()) and not s.escape_mode and not self._goal_visible(loc):
            s.escape_mode = True
            s.escape_dir = self.choose_recovery_turn(
                loc,
                plan_action if plan_action in (Action.LEFT, Action.RIGHT) else Action.IDLE,
                s.last_turn_dir if s.last_turn_dir != Action.IDLE else s.last_wall_turn,
            )
            s.escape_turn_left = self.cfg.escape_turn_frames
            s.escape_forward_left = self.cfg.escape_forward_frames
            self._clear_search()

        if s.escape_mode:
            if s.escape_turn_left > 0:
                s.escape_turn_left -= 1
                return self._start_turn(
                    s.escape_dir,
                    f"ESCAPE TURN {s.escape_dir.name}",
                    lock_frames=1,
                    post_forward_frames=0,
                    cooldown_frames=self.cfg.wall_turn_cooldown_frames,
                )
            if blocked_ahead:
                self._enter_wall_escape(loc, s.escape_dir)
                return self._run_wall_escape(loc)
            if s.escape_forward_left > 0:
                s.escape_forward_left -= 1
                s.status_text = "ESCAPE FORWARD"
                s.auto_action = Action.FORWARD
                s.last_auto_action = Action.FORWARD
                return s.auto_action
            s.escape_mode = False

        if s.turn_lock_left > 0:
            s.turn_lock_left -= 1
            s.status_text = f"TURN-LOCK {s.turn_lock_dir.name}"
            s.auto_action = s.turn_lock_dir
            s.last_auto_action = s.turn_lock_dir
            return s.auto_action

        if s.post_turn_forward_left > 0:
            if blocked_ahead:
                self._enter_wall_escape(loc, s.last_turn_dir)
                return self._run_wall_escape(loc)
            s.post_turn_forward_left -= 1
            s.status_text = "POST-TURN FORWARD"
            s.auto_action = Action.FORWARD
            s.last_auto_action = Action.FORWARD
            return s.auto_action

        if s.forward_burst_left > 0:
            if blocked_ahead:
                self._enter_wall_escape(loc, s.last_turn_dir)
                return self._run_wall_escape(loc)
            s.forward_burst_left -= 1
            s.status_text = "BURST FORWARD"
            s.auto_action = Action.FORWARD
            s.last_auto_action = Action.FORWARD
            return s.auto_action

        if blocked_ahead:
            self._enter_wall_escape(loc, plan_action if plan_action in (Action.LEFT, Action.RIGHT) else Action.IDLE)
            return self._run_wall_escape(loc)

        # If the planner wants to go backward, do not obey literally. Search or greedy-forward instead.
        if plan_action == Action.BACKWARD:
            plan_action = Action.IDLE

        # Search mode is now a fallback only after prolonged low confidence.
        if not planner_trusted and s.low_conf_streak >= max(8, self.cfg.stagnation_window // 2) and not self._goal_visible(loc):
            if s.search_fail_count >= 2 or self._progress_stalled():
                self._enter_reroute(loc, plan_action if plan_action in (Action.LEFT, Action.RIGHT) else Action.IDLE)
                return self._run_reroute(loc)
            if not s.search_mode:
                self._begin_search(loc, plan_action if plan_action in (Action.LEFT, Action.RIGHT) else Action.IDLE)
            return self._run_search_cycle(loc, "SEARCH")

        self._clear_search()
        return self._choose_greedy_action(loc, plan_action, planner_trusted)
