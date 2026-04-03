"""
Path planning module.

This module builds a graph from exploration data, where nodes represent
locations and edges represent transitions between them.

It computes the shortest path (in hops) from the current estimated location
to the goal, providing high-level navigation guidance.
"""

import json
import os
import networkx as nx
import numpy as np

from config import VisNavConfig

class TrajectoryGraphPlanner:
    def __init__(self, cfg: VisNavConfig):
        self.cfg = cfg
        self.motion_frames: list[dict] = []
        self.file_list: list[str] = []
        self.traj_boundaries: list[tuple[int, int]] = []
        self.G = None
        self.goal_node = None
        self.goal_vecs = None
        self.goal_front_vec = None
        self._load_exploration_data()

    def _load_exploration_data(self):
        pure = {"FORWARD", "LEFT", "RIGHT", "BACKWARD"}
        traj_dirs = sorted([
            d for d in os.listdir(self.cfg.data_dir)
            if d.startswith("traj_") and os.path.isdir(os.path.join(self.cfg.data_dir, d))
        ])

        all_motion = []
        if traj_dirs:
            for traj_dir_name in traj_dirs:
                traj_path = os.path.join(self.cfg.data_dir, traj_dir_name)
                info_path = os.path.join(traj_path, "data_info.json")
                if not os.path.exists(info_path):
                    continue
                with open(info_path) as f:
                    raw = json.load(f)
                traj_motion = [
                    {
                        "step": d["step"],
                        "image": d["image"],
                        "action": d["action"][0],
                        "traj_id": traj_dir_name,
                        "image_path": os.path.join(traj_path, d["image"]),
                    }
                    for d in raw
                    if len(d["action"]) == 1 and d["action"][0] in pure
                ]
                all_motion.extend(traj_motion)
        else:
            legacy_info = os.path.join(self.cfg.data_dir, "data_info.json")
            legacy_img_dir = os.path.join(self.cfg.data_dir, "images")
            with open(legacy_info) as f:
                raw = json.load(f)
            all_motion = [
                {
                    "step": d["step"],
                    "image": d["image"],
                    "action": d["action"][0],
                    "traj_id": "traj_0",
                    "image_path": os.path.join(legacy_img_dir, d["image"]),
                }
                for d in raw
                if len(d["action"]) == 1 and d["action"][0] in pure
            ]

        self.motion_frames = all_motion[:: self.cfg.subsample_rate]
        self.file_list = [m["image_path"] for m in self.motion_frames]

        self.traj_boundaries = []
        prev_traj = None
        for idx, m in enumerate(self.motion_frames):
            if m["traj_id"] != prev_traj:
                if prev_traj is not None:
                    self.traj_boundaries[-1] = (self.traj_boundaries[-1][0], idx)
                self.traj_boundaries.append((idx, len(self.motion_frames)))
                prev_traj = m["traj_id"]
        if self.traj_boundaries:
            self.traj_boundaries[-1] = (self.traj_boundaries[-1][0], len(self.motion_frames))

        print(f"Loaded {len(self.motion_frames)} motion frames")

    def build_graph(self, database: np.ndarray):
        if self.G is not None:
            return
        n = len(database)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))
        for start, end in self.traj_boundaries:
            for i in range(start, end - 1):
                self.G.add_edge(i, i + 1, weight=self.cfg.temporal_weight, edge_type="temporal")

        if self.cfg.top_k_shortcuts <= 0:
            return

        print("Computing visual shortcuts...")
        sim = database @ database.T
        np.fill_diagonal(sim, -2)
        for i in range(n):
            lo = max(0, i - self.cfg.min_shortcut_gap)
            hi = min(n, i + self.cfg.min_shortcut_gap + 1)
            sim[i, lo:hi] = -2
        sim[~np.triu(np.ones((n, n), dtype=bool), k=1)] = -2

        flat = sim.ravel()
        k = min(self.cfg.top_k_shortcuts, max(1, len(flat) - 1))
        top_idx = np.argpartition(flat, -k)[-k:]
        top_idx = top_idx[np.argsort(-flat[top_idx])]
        for fi in top_idx:
            i, j = divmod(int(fi), n)
            s = float(flat[fi])
            d = float(np.sqrt(max(0.0, 2.0 - 2.0 * s)))
            self.G.add_edge(
                i,
                j,
                weight=self.cfg.visual_weight_base + self.cfg.visual_weight_scale * d,
                edge_type="visual",
            )

    def setup_goal(self, target_images: list[np.ndarray], extractor, database: np.ndarray):
        if not target_images:
            return
        self.goal_vecs = [extractor.extract(img) for img in target_images]
        self.goal_front_vec = self.goal_vecs[0]
        sims = database @ self.goal_front_vec
        self.goal_node = int(np.argmax(sims))
        print(f"Goal node = {self.goal_node}, sim={float(sims[self.goal_node]):.4f}")

    def path_from_node(self, start: int) -> list[int]:
        try:
            return nx.shortest_path(self.G, start, self.goal_node, weight="weight")
        except nx.NetworkXNoPath:
            return [start]

    def edge_action(self, a: int, b: int) -> str:
        reverse = {"FORWARD": "BACKWARD", "BACKWARD": "FORWARD", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        if b == a + 1 and a < len(self.motion_frames):
            return self.motion_frames[a]["action"]
        if b == a - 1 and b < len(self.motion_frames):
            return reverse.get(self.motion_frames[b]["action"], "?")
        return "?"

    def first_executable_step(self, path: list[int]):
        if len(path) < 2:
            return None, None
        nxt = path[1]
        if self.G[path[0]][nxt].get("edge_type") == "temporal":
            return nxt, self.edge_action(path[0], nxt)
        cur = path[0]
        candidates = []
        for nb in self.G.neighbors(cur):
            if self.G[cur][nb].get("edge_type") != "temporal":
                continue
            try:
                dist = nx.shortest_path_length(self.G, nb, self.goal_node, weight="weight")
            except nx.NetworkXNoPath:
                dist = 1e9
            candidates.append((dist, nb, self.edge_action(cur, nb)))
        if not candidates:
            return None, None
        candidates.sort(key=lambda x: x[0])
        _, nb, action = candidates[0]
        return nb, action
