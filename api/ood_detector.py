import json
import numpy as np
from PIL import Image
from fastembed import ImageEmbedding


class OODDetector:
    """
    Out-of-Distribution detector:
      - CLIP ViT-B/32 vision encoder (via fastembed / ONNX runtime,
        model "Qdrant/clip-ViT-B-32-vision") -> raw image embedding
      - Dimensionality reduction via PCA (~40 components)
      - Mahalanobis distance against per-class in-distribution centers,
        using a shared covariance estimated with Ledoit-Wolf (sigma_inv)

    Expected files inside `ood_dir`:
      - ood_config.json         -> metadata (threshold, model_name, etc.)
      - ood_pca_mean.npy        -> (D_raw,) mean used to center raw embedding before PCA
      - ood_pca_components.npy  -> (n_components, D_raw) PCA projection matrix
      - ood_centers.npy         -> (K, n_components) per-class means (PCA space)
      - ood_sigma_inv.npy       -> (n_components, n_components) shared inverse covariance
    """

    def __init__(self, ood_dir="models/ood", model_name=None):
        with open(f"{ood_dir}/ood_config.json") as f:
            self.config = json.load(f)

        self.threshold = (
            self.config.get("threshold")
            or self.config.get("mahalanobis_threshold")
            or self.config.get("ood_threshold")
        )
        if self.threshold is None:
            raise ValueError(
                "No threshold found in ood_config.json. "
                f"Available keys: {list(self.config.keys())}"
            )

        self.model_name = (
            model_name
            or self.config.get("model_name")
            or "Qdrant/clip-ViT-B-32-vision"
        )

        self.pca_mean = np.load(f"{ood_dir}/ood_pca_mean.npy")
        self.pca_components = np.load(f"{ood_dir}/ood_pca_components.npy")
        self.centers = np.load(f"{ood_dir}/ood_centers.npy")
        self.sigma_inv = np.load(f"{ood_dir}/ood_sigma_inv.npy")

        if self.centers.ndim == 1:
            self.centers = self.centers[None, :]

        print(
            f"[OOD] model={self.model_name} "
            f"pca_mean={self.pca_mean.shape} "
            f"pca_components={self.pca_components.shape} "
            f"centers={self.centers.shape} sigma_inv={self.sigma_inv.shape} "
            f"threshold={self.threshold}"
        )

        # fastembed downloads/caches the ONNX model on first use
        self.embedder = ImageEmbedding(model_name=self.model_name)

    def _extract_features(self, crop: Image.Image) -> np.ndarray:
        crop = crop.convert("RGB")
        try:
            # Newer fastembed versions accept PIL Images directly
            emb = next(iter(self.embedder.embed([crop])))
        except Exception:
            # Fallback for versions that only accept file paths
            import os
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                crop.save(tmp.name)
                tmp_path = tmp.name
            try:
                emb = next(iter(self.embedder.embed([tmp_path])))
            finally:
                os.unlink(tmp_path)
        return np.asarray(emb, dtype=np.float32)

    def _mahalanobis(self, z: np.ndarray) -> float:
        diffs = self.centers - z[None, :]  # (K, n_components)
        dists = np.einsum("ij,jk,ik->i", diffs, self.sigma_inv, diffs)
        return float(np.sqrt(np.maximum(dists.min(), 0.0)))

    def score(self, crop: Image.Image) -> dict:
        feats = self._extract_features(crop)          # (D_raw,)
        z = (feats - self.pca_mean) @ self.pca_components.T  # (n_components,)
        distance = self._mahalanobis(z)
        return {
            "ood_distance": distance,
            "ood_threshold": self.threshold,
            "is_ood": distance > self.threshold,
        }