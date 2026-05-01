"""
KLT-heavy motion feedback (v17).

This module still does NOT replace visual localization.  Instead, it gives the
controller a stronger real-time estimate of whether the robot is physically
moving, stuck against a wall, or just spinning in place.  v17 checks KLT every
frame on a small image and reports both "stuck" and "good forward motion".
"""

from dataclasses import dataclass
import cv2
import numpy as np
from vis_nav_game import Action

from config import VisNavConfig


@dataclass
class MotionFeedbackState:
    median_flow: float = 0.0
    mean_dx: float = 0.0
    mean_dy: float = 0.0
    tracked_points: int = 0
    stuck_streak: int = 0
    stuck: bool = False
    good_forward: bool = False
    turn_flow: bool = False
    cooldown_left: int = 0
    skipped: bool = False


class ShiTomasiKLTFeedback:
    def __init__(self, cfg: VisNavConfig):
        self.cfg = cfg
        self.prev_gray = None
        self.prev_pts = None
        self.frame_count = 0
        self.state = MotionFeedbackState()

        self.feature_params = dict(
            maxCorners=int(getattr(cfg, "klt_max_corners", 70)),
            qualityLevel=float(getattr(cfg, "klt_quality_level", 0.01)),
            minDistance=int(getattr(cfg, "klt_min_distance", 9)),
            blockSize=int(getattr(cfg, "klt_block_size", 5)),
        )
        win = int(getattr(cfg, "klt_win_size", 13))
        self.lk_params = dict(
            winSize=(win, win),
            maxLevel=int(getattr(cfg, "klt_max_level", 1)),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(getattr(cfg, "klt_max_iter", 10)),
                0.03,
            ),
        )

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.frame_count = 0
        self.state = MotionFeedbackState()

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scale = float(getattr(self.cfg, "klt_downscale", 0.45))
        if 0.0 < scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray

    def _detect_points(self, gray: np.ndarray):
        h, w = gray.shape[:2]
        mask = np.zeros_like(gray)
        # Track lower/middle view where wall/floor texture moves strongly.
        mask[int(0.26 * h): int(0.96 * h), int(0.05 * w): int(0.95 * w)] = 255
        return cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)

    def _refresh_reference(self, gray: np.ndarray, clear_stuck: bool = True):
        self.prev_gray = gray
        self.prev_pts = self._detect_points(gray)
        if clear_stuck:
            self.state.stuck = False
            self.state.stuck_streak = 0

    def update(self, frame_bgr: np.ndarray, commanded_action: Action) -> MotionFeedbackState:
        if not self.cfg.use_motion_feedback:
            self.state = MotionFeedbackState()
            return self.state

        self.frame_count += 1
        gray = self._preprocess(frame_bgr)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < self.cfg.klt_min_points:
            self._refresh_reference(gray, clear_stuck=True)
            self.state.skipped = False
            return self.state

        skip = max(1, int(getattr(self.cfg, "klt_frame_skip", 1)))
        if (self.frame_count % skip) != 0:
            self.state.skipped = True
            return self.state

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        if next_pts is None or status is None:
            tracked = 0
            median_flow = 0.0
            mean_dx = 0.0
            mean_dy = 0.0
        else:
            good_new = next_pts[status.flatten() == 1].reshape(-1, 2)
            good_old = self.prev_pts[status.flatten() == 1].reshape(-1, 2)
            tracked = len(good_new)
            if tracked > 0:
                vec = good_new - good_old
                flow = np.linalg.norm(vec, axis=1)
                # Ignore crazy outliers caused by repeated textures.
                if len(flow) >= 6:
                    lo, hi = np.percentile(flow, [10, 90])
                    keep = (flow >= lo) & (flow <= hi)
                    vec = vec[keep]
                    flow = flow[keep]
                median_flow = float(np.median(flow)) if len(flow) else 0.0
                mean_dx = float(np.median(vec[:, 0])) if len(vec) else 0.0
                mean_dy = float(np.median(vec[:, 1])) if len(vec) else 0.0
                tracked = int(len(flow))
            else:
                median_flow = 0.0
                mean_dx = 0.0
                mean_dy = 0.0

        self.state.tracked_points = int(tracked)
        self.state.median_flow = float(median_flow)
        self.state.mean_dx = float(mean_dx)
        self.state.mean_dy = float(mean_dy)
        self.state.skipped = False

        enough = tracked >= self.cfg.klt_min_points
        low_motion = enough and median_flow < self.cfg.klt_stuck_flow_px
        good_motion = enough and median_flow >= self.cfg.klt_good_flow_px

        if commanded_action == Action.FORWARD:
            if self.state.cooldown_left > 0:
                self.state.cooldown_left -= 1
                self.state.stuck = False
                self.state.stuck_streak = 0
            else:
                if low_motion:
                    self.state.stuck_streak += 1
                else:
                    self.state.stuck_streak = 0
                self.state.stuck = self.state.stuck_streak >= self.cfg.klt_stuck_frames
            self.state.good_forward = good_motion and not self.state.stuck
            self.state.turn_flow = False
        elif commanded_action in (Action.LEFT, Action.RIGHT):
            # Turning should create horizontal flow. This is mainly for HUD/debug
            # and for clearing false stuck state while rotating.
            self.state.cooldown_left = self.cfg.klt_ignore_after_turn_frames
            self.state.stuck = False
            self.state.stuck_streak = 0
            self.state.good_forward = False
            self.state.turn_flow = good_motion or abs(mean_dx) >= self.cfg.klt_good_flow_px * 0.6
        else:
            self.state.cooldown_left = max(0, self.state.cooldown_left - 1)
            self.state.stuck = False
            self.state.stuck_streak = 0
            self.state.good_forward = False
            self.state.turn_flow = False

        self.prev_gray = gray
        self.prev_pts = self._detect_points(gray)
        return self.state
