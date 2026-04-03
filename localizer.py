"""
Visual localization module.

Given the current camera frame, this module compares it against stored
feature descriptors to estimate the robot's position in the map.

It outputs candidate locations, similarity scores, and confidence levels,
which are used by the planner and controller.
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

        # prev LOW
        if current_sim >= self.cfg.high_conf_sim + m and gap >= self.cfg.min_gap_high:
            return "HIGH"
        if current_sim >= self.cfg.med_conf_sim + m and gap >= self.cfg.min_gap_med:
            return "MED"
        return raw

    def update(self, feat: np.ndarray, fpv: np.ndarray) -> LocalizationState:
        raw_sims = self.database @ feat
        if self.ema_sims is None:
            smoothed = raw_sims.copy()
        else:
            smoothed = self.cfg.sim_ema_alpha * raw_sims + (1.0 - self.cfg.sim_ema_alpha) * self.ema_sims
        self.ema_sims = smoothed

        node, sim, top_idx, top_scores = self._consensus_pick(smoothed)
        self.prev_node = node

        gray_mean, edge_ratio = self._center_wall_score(fpv)
        front_wall = gray_mean < self.cfg.center_dark_thresh and edge_ratio > self.cfg.center_edge_thresh
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
