"""
Visual localization module.

Given the current camera frame, this module compares it against stored
feature descriptors to estimate the robot's position in the map.

<<<<<<< HEAD
It outputs candidate locations, similarity scores, and confidence levels,
which are used by the planner and controller.
=======
v6 also adds light-weight floor/corner geometry cues. These cues do NOT
replace VLAD localization; they only help the controller decide whether the
view looks like a front block, side opening, corner, or dead end.
>>>>>>> 38b10aa (Update)
"""

from collections import Counter, deque
from dataclasses import dataclass
import cv2
import numpy as np

from config import VisNavConfig

@dataclass
class LocalizationState:
    current_node: int | None = None
    current_sim: float | None = None
    conf_text: str = "LOW"
    top_match_indices: list[int] | None = None
    top_match_scores: list[float] | None = None
    current_path: list[int] | None = None
    goal_best_view: int | None = None
    goal_best_sim: float = 0.0
    goal_front_sim: float = 0.0
    front_wall: bool = False
    left_open: float = 0.0
    right_open: float = 0.0
    gray_mean: float = 255.0
    edge_ratio: float = 0.0
<<<<<<< HEAD
=======
    floor_center: float = 0.0
    floor_left: float = 0.0
    floor_right: float = 0.0
    corner_hint: str = "NONE"  # NONE / LEFT / RIGHT / DEADEND
>>>>>>> 38b10aa (Update)


class SmoothedLocalizer:
    def __init__(self, cfg: VisNavConfig, database: np.ndarray, goal_vecs: list[np.ndarray] | None = None):
        self.cfg = cfg
        self.database = database
        self.goal_vecs = goal_vecs
        self.goal_front_vec = goal_vecs[0] if goal_vecs else None
        self.prev_node = None
        self.prev_conf = "LOW"
        self.ema_sims = None
        self.node_history = deque(maxlen=cfg.node_history)
        self.sim_history = deque(maxlen=cfg.stagnation_window)
        self.node_trace = deque(maxlen=cfg.stagnation_window)
        self.wall_trace = deque(maxlen=cfg.stagnation_window)
<<<<<<< HEAD
=======
        self.lost_jump_streak = 0
>>>>>>> 38b10aa (Update)
        self.state = LocalizationState(top_match_indices=[], top_match_scores=[], current_path=[])

    def _center_wall_score(self, img: np.ndarray):
        h, w = img.shape[:2]
        y0, y1 = int(0.22 * h), int(0.82 * h)
        x0, x1 = int(0.35 * w), int(0.65 * w)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            return 255.0, 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        return float(np.mean(gray)), float(np.mean(edges > 0))

    def _side_openness(self, img: np.ndarray):
        h, w = img.shape[:2]
        y0, y1 = int(0.35 * h), int(0.92 * h)
        left = img[y0:y1, : w // 3]
        right = img[y0:y1, 2 * w // 3:]

        def score(roi: np.ndarray) -> float:
            if roi.size == 0:
                return 0.0
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2].astype(np.float32)
            bright = float(np.mean(v))
            edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 60, 140)
            edge_penalty = 35.0 * float(np.mean(edges > 0))
            return bright - edge_penalty

        return score(left), score(right)

<<<<<<< HEAD
=======
    def _floor_geometry(self, img: np.ndarray):
        """Estimate walkable floor from color continuity + low texture.

        The game floor is usually a smooth blue/white plane. Instead of trying
        to detect all possible walls, this detects where the smooth floor-like
        region continues in the lower/middle image. This is more useful for
        corner/dead-end handling than brightness alone.
        """
        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140) > 0

        # Reference floor color from the bottom-center area nearest the robot.
        by0, by1 = int(0.78 * h), int(0.96 * h)
        bx0, bx1 = int(0.38 * w), int(0.62 * w)
        ref_patch = lab[by0:by1, bx0:bx1]
        if ref_patch.size == 0:
            return 0.0, 0.0, 0.0, "NONE"
        ref = np.median(ref_patch.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(lab - ref, axis=2)

        # Floor candidates: similar to bottom floor color OR bright/low-saturation,
        # and not too edge-heavy. Restrict to lower 65% to avoid ceiling/sky.
        similar = dist < 48.0
        bright_flat = (hsv[:, :, 2] > 120) & (hsv[:, :, 1] < 95)
        mask = (similar | bright_flat) & (~edges)
        mask[: int(0.34 * h), :] = False

        def region_score(x0f, x1f, y0f=0.46, y1f=0.92):
            x0, x1 = int(x0f * w), int(x1f * w)
            y0, y1 = int(y0f * h), int(y1f * h)
            roi = mask[y0:y1, x0:x1]
            if roi.size == 0:
                return 0.0
            edge_roi = edges[y0:y1, x0:x1]
            return max(0.0, 100.0 * float(np.mean(roi)) - 15.0 * float(np.mean(edge_roi)))

        center = region_score(0.40, 0.60, 0.50, 0.88)
        left = region_score(0.10, 0.38, 0.44, 0.90)
        right = region_score(0.62, 0.90, 0.44, 0.90)

        hint = "NONE"
        front_bad = center < self.cfg.floor_front_block_score
        left_good = left >= self.cfg.floor_side_open_score
        right_good = right >= self.cfg.floor_side_open_score
        left_dead = left < self.cfg.floor_deadend_side_score
        right_dead = right < self.cfg.floor_deadend_side_score
        if front_bad and left_dead and right_dead:
            hint = "DEADEND"
        elif front_bad and left_good and left > right + 4.0:
            hint = "LEFT"
        elif front_bad and right_good and right > left + 4.0:
            hint = "RIGHT"
        return center, left, right, hint

>>>>>>> 38b10aa (Update)
    def _goal_view_decision(self, feat: np.ndarray):
        if self.goal_vecs is None:
            return None, 0.0
        sims = [float(g @ feat) for g in self.goal_vecs]
        best = int(np.argmax(sims))
        return best, sims[best]

    def _consensus_pick(self, smoothed: np.ndarray) -> tuple[int, float, list[int], list[float]]:
        order = np.argsort(-smoothed)
        top_idx = [int(i) for i in order[: self.cfg.top_candidates]]
        top_scores = [float(smoothed[i]) for i in top_idx]

        candidates = top_idx[: self.cfg.topk_consensus]
        best_score = -1e9
        best_node = top_idx[0]
        for idx in candidates:
            score = float(smoothed[idx])
            if self.prev_node is not None:
                score -= self.cfg.candidate_prior_penalty * abs(idx - self.prev_node)
            if score > best_score:
                best_score = score
                best_node = int(idx)

        if self.prev_node is not None:
            lo = max(0, self.prev_node - self.cfg.local_prior_window)
            hi = min(len(smoothed), self.prev_node + self.cfg.local_prior_window + 1)
            local_best_rel = int(np.argmax(smoothed[lo:hi]))
            local_best = lo + local_best_rel
            local_sim = float(smoothed[local_best])
            global_sim = float(smoothed[best_node])
            if global_sim <= local_sim + self.cfg.global_override_margin:
                best_node = int(local_best)

        self.node_history.append(best_node)
        majority = Counter(self.node_history).most_common(1)[0][0]
        if abs(majority - best_node) <= 6:
            best_node = int(majority)

        return best_node, float(smoothed[best_node]), top_idx, top_scores

    def _confidence_from_scores(self, current_sim: float | None, top_scores: list[float]) -> str:
        gap = 0.0
        if len(top_scores) >= 2:
            gap = top_scores[0] - top_scores[1]
        if current_sim is None:
            return "LOW"

        raw = "LOW"
        if current_sim >= self.cfg.high_conf_sim and gap >= self.cfg.min_gap_high:
            raw = "HIGH"
        elif current_sim >= self.cfg.med_conf_sim and gap >= self.cfg.min_gap_med:
            raw = "MED"

        m = self.cfg.conf_hysteresis_margin
        prev = self.prev_conf

        if prev == "HIGH":
            if current_sim >= self.cfg.high_conf_sim - m and gap >= self.cfg.min_gap_med:
                return "HIGH"
            if current_sim >= self.cfg.med_conf_sim - m and gap >= self.cfg.min_gap_med * 0.75:
                return "MED"
            return "LOW"

        if prev == "MED":
            if current_sim >= self.cfg.high_conf_sim + m and gap >= self.cfg.min_gap_high:
                return "HIGH"
            if current_sim >= self.cfg.med_conf_sim - m and gap >= self.cfg.min_gap_med * 0.75:
                return "MED"
            return "LOW"

<<<<<<< HEAD
        # prev LOW
=======
>>>>>>> 38b10aa (Update)
        if current_sim >= self.cfg.high_conf_sim + m and gap >= self.cfg.min_gap_high:
            return "HIGH"
        if current_sim >= self.cfg.med_conf_sim + m and gap >= self.cfg.min_gap_med:
            return "MED"
        return raw

<<<<<<< HEAD
=======
    def _stabilize_node_jump(self, node: int, sim: float, top_scores: list[float], smoothed: np.ndarray) -> tuple[int, float]:
        """Reject sudden far-away localization jumps unless evidence is very strong.

        In the 51x51 maze, repeated wall images can make VLAD briefly match a
        visually similar but physically unrelated corridor. If we accept that
        immediately, the planner hops jump to a huge number. This method keeps
        the estimate near the previous node during LOW/MED evidence and only
        accepts global jumps when the match is clearly strong.
        """
        if self.prev_node is None:
            self.lost_jump_streak = 0
            return node, sim

        gap = 0.0
        if len(top_scores) >= 2:
            gap = float(top_scores[0] - top_scores[1])

        jump = abs(int(node) - int(self.prev_node))
        if sim >= self.cfg.high_conf_sim and gap >= self.cfg.min_gap_high:
            max_jump = self.cfg.max_node_jump_high
        elif sim >= self.cfg.med_conf_sim and gap >= self.cfg.min_gap_med:
            max_jump = self.cfg.max_node_jump_med
        else:
            max_jump = self.cfg.max_node_jump_low

        strong_global = sim >= self.cfg.jump_accept_sim and gap >= self.cfg.jump_accept_gap
        if jump <= max_jump or strong_global:
            self.lost_jump_streak = 0
            return node, sim

        # Prefer the best nearby candidate instead of teleporting across the map.
        radius = max(self.cfg.local_prior_window, self.cfg.jump_search_radius)
        lo = max(0, int(self.prev_node) - radius)
        hi = min(len(smoothed), int(self.prev_node) + radius + 1)
        if hi <= lo:
            return self.prev_node, float(smoothed[int(self.prev_node)])

        local_best = lo + int(np.argmax(smoothed[lo:hi]))
        local_sim = float(smoothed[local_best])
        self.lost_jump_streak += 1

        # If the same global evidence stays strong for many frames, allow it;
        # otherwise the robot would never recover after being truly lost.
        if self.lost_jump_streak >= self.cfg.lost_global_after_frames and sim >= self.cfg.med_conf_sim:
            self.lost_jump_streak = 0
            return node, sim

        return int(local_best), local_sim

>>>>>>> 38b10aa (Update)
    def update(self, feat: np.ndarray, fpv: np.ndarray) -> LocalizationState:
        raw_sims = self.database @ feat
        if self.ema_sims is None:
            smoothed = raw_sims.copy()
        else:
            smoothed = self.cfg.sim_ema_alpha * raw_sims + (1.0 - self.cfg.sim_ema_alpha) * self.ema_sims
        self.ema_sims = smoothed

        node, sim, top_idx, top_scores = self._consensus_pick(smoothed)
<<<<<<< HEAD
        self.prev_node = node

        gray_mean, edge_ratio = self._center_wall_score(fpv)
        front_wall = gray_mean < self.cfg.center_dark_thresh and edge_ratio > self.cfg.center_edge_thresh
=======
        node, sim = self._stabilize_node_jump(node, sim, top_scores, smoothed)
        self.prev_node = node

        gray_mean, edge_ratio = self._center_wall_score(fpv)
        floor_center, floor_left, floor_right, corner_hint = self._floor_geometry(fpv)
        front_wall = (
            (gray_mean < self.cfg.center_dark_thresh and edge_ratio > self.cfg.center_edge_thresh)
            or gray_mean <= self.cfg.front_dark_block_brightness
            or (floor_center < self.cfg.floor_front_block_score and corner_hint in ("LEFT", "RIGHT", "DEADEND"))
        )
>>>>>>> 38b10aa (Update)
        left_open, right_open = self._side_openness(fpv)
        goal_best_view, goal_best_sim = self._goal_view_decision(feat)
        goal_front_sim = float(self.goal_front_vec @ feat) if self.goal_front_vec is not None else 0.0
        conf_text = self._confidence_from_scores(sim, top_scores)
        self.prev_conf = conf_text

        self.sim_history.append(sim)
        self.node_trace.append(node)
        self.wall_trace.append(front_wall)

        self.state = LocalizationState(
            current_node=node,
            current_sim=sim,
            conf_text=conf_text,
            top_match_indices=top_idx,
            top_match_scores=top_scores,
            current_path=[],
            goal_best_view=goal_best_view,
            goal_best_sim=goal_best_sim,
            goal_front_sim=goal_front_sim,
            front_wall=front_wall,
            left_open=left_open,
            right_open=right_open,
            gray_mean=gray_mean,
            edge_ratio=edge_ratio,
<<<<<<< HEAD
=======
            floor_center=floor_center,
            floor_left=floor_left,
            floor_right=floor_right,
            corner_hint=corner_hint,
>>>>>>> 38b10aa (Update)
        )
        return self.state

    def is_stagnating(self) -> bool:
        if len(self.node_trace) < self.cfg.stagnation_window:
            return False
        node_span = max(self.node_trace) - min(self.node_trace)
        wall_ratio = float(np.mean(self.wall_trace)) if self.wall_trace else 0.0
        sim_gain = float(max(self.sim_history) - min(self.sim_history)) if self.sim_history else 0.0
        return (
            node_span <= self.cfg.min_progress_nodes
            and sim_gain <= self.cfg.low_improve_margin
            and wall_ratio >= self.cfg.front_wall_ratio_trigger
        )
