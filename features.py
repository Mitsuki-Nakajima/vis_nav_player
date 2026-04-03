"""
Feature extraction and visual descriptor construction.

This module processes images from the exploration data and computes
feature representations (e.g., SIFT + VLAD). These descriptors are used
to compare the current camera view with stored map images.

It also handles caching of computed features for faster runtime.
"""

import os
import pickle
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from config import VisNavConfig


class VLADExtractor:
    def __init__(self, cfg: VisNavConfig):
        self.cfg = cfg
        self.n_clusters = cfg.n_clusters
        self.sift = cv2.SIFT_create(nfeatures=1200)
        self.codebook = None
        self._sift_cache: dict[str, np.ndarray] = {}
        os.makedirs(self.cfg.cache_dir, exist_ok=True)

    @property
    def dim(self) -> int:
        return self.n_clusters * 128

    @staticmethod
    def _root_sift(des: np.ndarray) -> np.ndarray:
        des = des.astype(np.float32)
        des /= (np.sum(des, axis=1, keepdims=True) + 1e-12)
        return np.sqrt(des)

    def _des_to_vlad(self, des: np.ndarray) -> np.ndarray:
        labels = self.codebook.predict(des)
        centers = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        vlad = np.zeros((k, des.shape[1]), dtype=np.float32)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                vlad[i] = np.sum(des[mask] - centers[i], axis=0)
                norm = np.linalg.norm(vlad[i])
                if norm > 0:
                    vlad[i] /= norm
        vlad = vlad.ravel()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm
        return vlad.astype(np.float32)

    def load_sift_cache(self, file_list: list[str]):
        cache_file = os.path.join(self.cfg.cache_dir, f"sift_ss{self.cfg.subsample_rate}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self._sift_cache = pickle.load(f)
            if all(fname in self._sift_cache for fname in file_list):
                print(f"Loaded SIFT cache: {cache_file}")
                return

        print(f"Extracting SIFT for {len(file_list)} frames...")
        self._sift_cache = {}
        for fname in tqdm(file_list, desc="SIFT"):
            img = cv2.imread(fname)
            if img is None:
                continue
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None and len(des) > 0:
                self._sift_cache[fname] = self._root_sift(des)
        with open(cache_file, "wb") as f:
            pickle.dump(self._sift_cache, f)
        print(f"Saved SIFT cache: {cache_file}")

    def build_vocabulary(self, file_list: list[str]):
        cache_file = os.path.join(self.cfg.cache_dir, f"codebook_k{self.n_clusters}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.codebook = pickle.load(f)
            print(f"Loaded codebook: {cache_file}")
            return

        all_des = [self._sift_cache[f] for f in file_list if f in self._sift_cache]
        if not all_des:
            raise RuntimeError("No descriptors found for codebook construction.")
        all_des = np.vstack(all_des)
        print(f"Training MiniBatchKMeans with {len(all_des)} descriptors...")
        self.codebook = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=4096,
            n_init=3,
            max_iter=200,
            verbose=0,
        ).fit(all_des)
        with open(cache_file, "wb") as f:
            pickle.dump(self.codebook, f)
        print(f"Saved codebook: {cache_file}")

    def extract(self, img: np.ndarray) -> np.ndarray:
        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return self._des_to_vlad(self._root_sift(des))

    def extract_batch(self, file_list: list[str]) -> np.ndarray:
        vectors = []
        for fname in tqdm(file_list, desc="VLAD"):
            if fname in self._sift_cache and len(self._sift_cache[fname]) > 0:
                vectors.append(self._des_to_vlad(self._sift_cache[fname]))
            else:
                vectors.append(np.zeros(self.dim, dtype=np.float32))
        return np.asarray(vectors, dtype=np.float32)
