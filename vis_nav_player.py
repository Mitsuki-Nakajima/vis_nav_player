"""
Main entry point for the visual navigation system.

This script initializes all components (config, features, localizer,
planner, controller) and runs the navigation loop.

It also handles user input (manual control), rendering the UI,
and displaying navigation status and goal detection.
"""

import argparse
import os
import sys

import cv2
import pygame
from vis_nav_game import Action, Phase, Player

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config import VisNavConfig
from controller import AutoController
from features import VLADExtractor
from localizer import SmoothedLocalizer
from planner import TrajectoryGraphPlanner


class FinalManualAssistPlayer(Player):
    def __init__(self, cfg: VisNavConfig | None = None):
        self.cfg = cfg or VisNavConfig()
        self.fpv = None
        self.screen = None
        self.last_act = Action.IDLE
        self.keymap = None
        self.auto_enabled = not self.cfg.manual_mode_default
        self.assist_hold = 0
        self.last_assist_text = "NOT READY"
        self._last_overlay = None
        self._fonts_ready = False
        self._font_small = None
        self._font_big = None

        self.extractor = VLADExtractor(self.cfg)
        self.planner = TrajectoryGraphPlanner(self.cfg)
        self.controller = AutoController(self.cfg)
        self.database = None
        self.localizer = None

        super().__init__()

    def reset(self):
        self.fpv = None
        self.screen = None
        self.last_act = Action.IDLE
        self.auto_enabled = not self.cfg.manual_mode_default
        self.assist_hold = 0
        self.last_assist_text = "NOT READY"
        self._last_overlay = None
        self._fonts_ready = False
        self._font_small = None
        self._font_big = None
        self.controller.reset()
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def _ensure_fonts(self):
        if not self._fonts_ready:
            pygame.font.init()
            self._font_small = pygame.font.SysFont("Arial", 22)
            self._font_big = pygame.font.SysFont("Arial", 34, bold=True)
            self._fonts_ready = True

    def pre_navigation(self):
        if self.database is None:
            self.extractor.load_sift_cache(self.planner.file_list)
            self.extractor.build_vocabulary(self.planner.file_list)
            self.database = self.extractor.extract_batch(self.planner.file_list)
            print(f"Database shape: {self.database.shape}")
        self.planner.build_graph(self.database)
        self.planner.setup_goal(self.get_target_images(), self.extractor, self.database)
        self.localizer = SmoothedLocalizer(self.cfg, self.database, self.planner.goal_vecs)

    def _compute_auto_action(self):
        feat = self.extractor.extract(self.fpv)
        loc = self.localizer.update(feat, self.fpv)
        loc.current_path = self.planner.path_from_node(loc.current_node)
        _, action_name = self.planner.first_executable_step(loc.current_path)
        plan_action = {
            "FORWARD": Action.FORWARD,
            "BACKWARD": Action.BACKWARD,
            "LEFT": Action.LEFT,
            "RIGHT": Action.RIGHT,
        }.get(action_name, Action.IDLE)
        near_goal = (
            loc.current_node == self.planner.goal_node or
            len(loc.current_path) - 1 <= self.cfg.max_near_goal_hops
        )
        stagnating = self.localizer.is_stagnating()
        auto_action = self.controller.compute_action(loc, plan_action, near_goal, stagnating)
        self.controller.state.auto_action = auto_action
        return auto_action, loc, plan_action, stagnating

    def _update_goal_assist(self, loc):
        path_hops = len(loc.current_path) - 1 if loc.current_path else 999
        near_goal_by_path = path_hops <= self.cfg.assist_near_goal_hops

        # More forgiving than the original: do not require only the front target view.
        # This helps when the goal has two usable target photos and the best match is not view 0.
        strong_any_view = loc.goal_best_sim >= max(0.24, self.cfg.assist_goal_best_sim - 0.08)
        front_ok = loc.goal_front_sim >= max(0.18, self.cfg.assist_goal_front_sim - 0.06)
        very_strong_any = loc.goal_best_sim >= max(0.30, self.cfg.assist_goal_best_sim)

        likely_now = (
            (strong_any_view and front_ok) or
            (near_goal_by_path and strong_any_view) or
            very_strong_any
        )

        if likely_now:
            self.assist_hold += 1
        else:
            self.assist_hold = 0

        ready_frames = max(2, self.cfg.assist_goal_hold_frames - 1)
        if self.assist_hold >= ready_frames:
            self.last_assist_text = "CHECK-IN LIKELY"
        elif likely_now:
            self.last_assist_text = "ALMOST"
        else:
            self.last_assist_text = "NOT READY"

    def _overlay_info(self, loc, plan_action: Action, stagnating: bool):
        mode_text = "MANUAL" if not self.auto_enabled else "AUTO"
        line1 = f"MODE {mode_text} (A toggle) | PLAN {plan_action.name} | CONF {loc.conf_text}"
        line2 = (
            f"node {loc.current_node} sim {0.0 if loc.current_sim is None else loc.current_sim:.3f} | "
            f"goal best view {loc.goal_best_view} sim {loc.goal_best_sim:.3f} front {loc.goal_front_sim:.3f}"
        )
        path_hops = len(loc.current_path) - 1 if loc.current_path else 999
        line3 = (
            f"{self.last_assist_text} | hops {path_hops} | wall {int(loc.front_wall)} "
            f"L {loc.left_open:.1f} R {loc.right_open:.1f}"
        )
        if stagnating:
            line3 += " | STUCK"
        return line1, line2, line3

    @staticmethod
    def _np_to_surface(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], "RGB")

    def _draw_text(self, text, x, y, color, big=False):
        font = self._font_big if big else self._font_small
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv
        self._ensure_fonts()

        window_size = (self.cfg.window_w, self.cfg.window_h)
        if self.screen is None or self.screen.get_size() != window_size:
            self.screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)

        overlay = None
        if self._state and self._state[1] == Phase.NAVIGATION and self.localizer is not None:
            auto_action, loc, plan_action, stagnating = self._compute_auto_action()
            self._update_goal_assist(loc)
            overlay = self._overlay_info(loc, plan_action, stagnating)
        self._last_overlay = overlay

        screen_w, screen_h = self.screen.get_size()
        img_h, img_w = fpv.shape[:2]

        # Leave margins for HUD so text stays crisp instead of being scaled with the image.
        top_margin = 70 if overlay is not None else 10
        bottom_margin = 70 if overlay is not None else 10
        avail_w = max(100, screen_w - 20)
        avail_h = max(100, screen_h - top_margin - bottom_margin)

        scale = min(avail_w / img_w, avail_h / img_h)
        scale *= min(1.0, self.cfg.camera_scale)
        scale = max(scale, 0.1)

        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(fpv, (new_w, new_h), interpolation=interp)
        surface = self._np_to_surface(scaled)

        self.screen.fill((0, 0, 0))
        x = max(0, (screen_w - new_w) // 2)
        y = max(top_margin, top_margin + (avail_h - new_h) // 2)

        pygame.display.set_caption("FinalManualAssistPlayer")
        self.screen.blit(surface, (x, y))

        if overlay is not None:
            line1, line2, line3 = overlay
            mode_color = self.cfg.hud_yellow if not self.auto_enabled else self.cfg.hud_green
            if self.last_assist_text == "CHECK-IN LIKELY":
                assist_color = self.cfg.hud_green
            elif self.last_assist_text == "ALMOST":
                assist_color = self.cfg.hud_yellow
            else:
                assist_color = self.cfg.hud_gray

            pygame.draw.rect(self.screen, self.cfg.hud_bg, pygame.Rect(0, 0, screen_w, top_margin - 8))
            pygame.draw.rect(self.screen, self.cfg.hud_bg, pygame.Rect(0, screen_h - bottom_margin + 8, screen_w, bottom_margin))
            self._draw_text(line1, 14, 12, mode_color)
            self._draw_text(line2, 14, 38, self.cfg.hud_text)
            self._draw_text(line3, 14, screen_h - bottom_margin + 24, assist_color)

            if self.last_assist_text == "CHECK-IN LIKELY":
                msg = "PRESS SPACE NOW!"
                msg_surf = self._font_big.render(msg, True, self.cfg.hud_green)
                msg_x = max(10, (screen_w - msg_surf.get_width()) // 2)
                msg_y = max(top_margin + 4, y - 42)
                self.screen.blit(msg_surf, (msg_x, msg_y))

        pygame.display.flip()

    def act(self):
        manual = Action.IDLE
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.auto_enabled = not self.auto_enabled
                elif event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP and event.key in self.keymap:
                self.last_act ^= self.keymap[event.key]

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            manual |= Action.LEFT
        if keys[pygame.K_RIGHT]:
            manual |= Action.RIGHT
        if keys[pygame.K_UP]:
            manual |= Action.FORWARD
        if keys[pygame.K_DOWN]:
            manual |= Action.BACKWARD
        if keys[pygame.K_SPACE]:
            manual |= Action.CHECKIN
        if keys[pygame.K_ESCAPE]:
            return Action.QUIT

        if manual != Action.IDLE:
            return manual
        if self.auto_enabled:
            return self.controller.state.auto_action
        return Action.IDLE

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])
        h, w = concat_img.shape[:2]
        color = (0, 0, 0)
        concat_img = cv2.line(concat_img, (w // 2, 0), (w // 2, h), color, 2)
        concat_img = cv2.line(concat_img, (0, h // 2), (w, h // 2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(concat_img, "Front View", (12, 34), font, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(concat_img, "Left View", (w // 2 + 12, 34), font, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(concat_img, "Back View", (12, h // 2 + 34), font, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(concat_img, "Right View", (w // 2 + 12, h // 2 + 34), font, 1.0, color, 2, cv2.LINE_AA)
        if self.cfg.target_window_scale != 1.0:
            concat_img = cv2.resize(
                concat_img,
                None,
                fx=self.cfg.target_window_scale,
                fy=self.cfg.target_window_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        cv2.imshow("Target Images", concat_img)
        cv2.waitKey(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--n-clusters", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_kwargs = {}
    if args.subsample is not None:
        cfg_kwargs["subsample_rate"] = args.subsample
    if args.n_clusters is not None:
        cfg_kwargs["n_clusters"] = args.n_clusters
    if args.top_k is not None:
        cfg_kwargs["top_k_shortcuts"] = args.top_k
    cfg = VisNavConfig(**cfg_kwargs)
    import vis_nav_game as vng
    vng.play(the_player=FinalManualAssistPlayer(cfg))
