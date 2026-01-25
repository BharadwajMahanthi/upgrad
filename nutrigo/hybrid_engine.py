# hybrid_engine.py
import os
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import config

# --------------------------- FILE PATHS --------------------------- #
TFIDF_VECTORIZER_PATH = config.TFIDF_VECTORIZER_PATH
TEXT_SVD_PATH = config.TEXT_SVD_PATH
SCALER_PATH = config.SCALER_PATH
SVD_MODEL_PATH = config.SVD_MODEL_PATH
LE_USER_PATH = config.LE_USER_PATH
LE_RECIPE_PATH = config.LE_RECIPE_PATH

RECIPE_MASTER_PATH = config.RECIPE_MASTER_CACHE_PATH
RECIPE_EMBEDDINGS_PATH = config.RECIPE_EMBEDDINGS_PATH
NEIGHBORS_INDEX_PATH = config.NEIGHBORS_INDEX_PATH

TARGET_NUTRIENTS = config.TARGET_NUTRIENTS

# Embedding config
TFIDF_MAX_FEATURES = config.TFIDF_MAX_FEATURES
TEXT_EMBED_DIM = config.TEXT_EMBED_DIM
IMAGE_EMBED_DIM = 0  # NOT using images in Flask version by default (optional)
CANDIDATES_FROM_CONTENT = config.CANDIDATES_FROM_CONTENT

# --------------------------- TEXT CLEANING --------------------------- #
_TEXT_CLEAN_RE = re.compile(r"[^a-zA-Z\s]+")

def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = _TEXT_CLEAN_RE.sub(" ", text)
    return " ".join(text.split())

def tokenize_ingredients(clean_text: str):
    if not isinstance(clean_text, str) or not clean_text.strip():
        return set()
    return set(clean_text.split())


# --------------------------- Hybrid Recommender --------------------------- #
class HybridRecommender:
    """
    Hybrid recommender engine for Flask:
      - SVD predictions (collaborative filtering)
      - Content similarity using TF-IDF + TruncatedSVD (fast embeddings)
      - Constraints + Explanation
    """

    def __init__(self, recipe_master_df, embeddings, neighbors, tfidf, text_svd, scaler, svd_model=None):
        self.recipe_master = recipe_master_df.reset_index(drop=True)
        self.emb = embeddings.astype(np.float32)  # normalized
        self.nn = neighbors
        self.tfidf = tfidf
        self.text_svd = text_svd
        self.scaler = scaler
        self.svd = svd_model

        self.rid_to_index = dict(zip(self.recipe_master["recipe_id"].astype(str), self.recipe_master.index.values))
        self.index_to_rid = dict(zip(self.recipe_master.index.values, self.recipe_master["recipe_id"].astype(str)))

    @staticmethod
    def _safe_float(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    def _apply_constraints_mask(self, df: pd.DataFrame, constraints: dict | None):
        if not constraints:
            return df
        out = df.copy()
        if constraints.get("max_calories") is not None:
            out = out[out["calories"] <= float(constraints["max_calories"])]
        if constraints.get("min_protein") is not None:
            out = out[out["protein"] >= float(constraints["min_protein"])]
        if constraints.get("max_fat") is not None:
            out = out[out["fat"] <= float(constraints["max_fat"])]
        if constraints.get("max_carbohydrates") is not None:
            out = out[out["carbohydrates"] <= float(constraints["max_carbohydrates"])]
        if constraints.get("min_fiber") is not None:
            out = out[out["fiber"] >= float(constraints["min_fiber"])]
        return out

    def _content_candidates(self, query_vec: np.ndarray, top_k: int = CANDIDATES_FROM_CONTENT):
        top_k = min(top_k, len(self.recipe_master))
        distances, indices = self.nn.kneighbors(query_vec.reshape(1, -1), n_neighbors=top_k)
        indices = indices.flatten()
        sims = 1.0 - distances.flatten()
        return indices, sims

    def recommend_hybrid(
        self,
        user_id: str,
        svd_model,
        top_k=10,
        alpha=0.70,
        constraints=None,
        exclude_seen=True,
    ):
        """
        Hybrid score:
          final = alpha * normalized_svd + (1-alpha) * content_similarity
        """
        # Note: we use passed svd_model if available, else self.svd
        model = svd_model if svd_model else self.svd
        
        # This implementation matches the original app.py logic
        # We'll use a mean vector for the user as a fallback
        query_vec = self.emb.mean(axis=0)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        cand_idx, cand_sim = self._content_candidates(query_vec, top_k=CANDIDATES_FROM_CONTENT)

        rows = []
        svd_scores = []

        for i, idx in enumerate(cand_idx):
            recipe_id = str(self.recipe_master.loc[idx, "recipe_id"])
            row = self.recipe_master.loc[idx]

            # constraint pre-check
            if constraints:
                if constraints.get("max_calories") is not None and float(row["calories"]) > float(constraints["max_calories"]):
                    continue
                if constraints.get("min_protein") is not None and float(row["protein"]) < float(constraints["min_protein"]):
                    continue
                if constraints.get("max_fat") is not None and float(row["fat"]) > float(constraints["max_fat"]):
                    continue
                if constraints.get("max_carbohydrates") is not None and float(row["carbohydrates"]) > float(constraints["max_carbohydrates"]):
                    continue
                if constraints.get("min_fiber") is not None and float(row["fiber"]) < float(constraints["min_fiber"]):
                    continue

            try:
                s = float(model.predict(str(user_id), str(recipe_id)).est) if model else np.nan
            except Exception:
                s = np.nan

            if not np.isnan(s):
                svd_scores.append(s)

            rows.append({
                "recipe_id": recipe_id,
                "recipe_name": row.get("recipe_name", ""),
                "content_similarity": float(cand_sim[i]),
                "svd_score": float(s) if not np.isnan(s) else np.nan,
                "calories": self._safe_float(row.get("calories", np.nan)),
                "protein": self._safe_float(row.get("protein", np.nan)),
                "fat": self._safe_float(row.get("fat", np.nan)),
                "carbohydrates": self._safe_float(row.get("carbohydrates", np.nan)),
                "fiber": self._safe_float(row.get("fiber", np.nan)),
            })

        recs = pd.DataFrame(rows)
        recs = self._apply_constraints_mask(recs, constraints)

        if recs.empty:
            return pd.DataFrame()

        # Normalize SVD scores
        svd_valid = recs["svd_score"].dropna().values
        if len(svd_valid) > 0:
            s_min, s_max = float(np.min(svd_valid)), float(np.max(svd_valid))
            if s_max > s_min:
                recs["svd_norm"] = (recs["svd_score"] - s_min) / (s_max - s_min)
            else:
                recs["svd_norm"] = 0.5
        else:
            recs["svd_norm"] = 0.0

        recs["final_score"] = alpha * recs["svd_norm"] + (1.0 - alpha) * recs["content_similarity"]
        recs = recs.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)
        return recs

    def recommend_from_keywords(self, keywords: str, top_k=10, constraints=None):
        """
        Cold-start recommendations from keyword query
        """
        clean = preprocess_text(keywords)
        vec = self.tfidf.transform([clean])
        text_emb = self.text_svd.transform(vec).astype(np.float32).reshape(-1)

        # Need same dimension as embeddings: [text_emb + nutrition_scaled(5)]
        nutr_neutral = np.zeros((len(TARGET_NUTRIENTS),), dtype=np.float32)
        query = np.hstack([text_emb, nutr_neutral]).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-9)

        cand_idx, cand_sim = self._content_candidates(query, top_k=CANDIDATES_FROM_CONTENT)

        rows = []
        for i, idx in enumerate(cand_idx):
            r = self.recipe_master.loc[idx]
            rows.append({
                "recipe_id": str(r["recipe_id"]),
                "recipe_name": r.get("recipe_name", ""),
                "content_similarity": float(cand_sim[i]),
                "svd_score": np.nan,
                "final_score": float(cand_sim[i]),
                "calories": self._safe_float(r.get("calories", np.nan)),
                "protein": self._safe_float(r.get("protein", np.nan)),
                "fat": self._safe_float(r.get("fat", np.nan)),
                "carbohydrates": self._safe_float(r.get("carbohydrates", np.nan)),
                "fiber": self._safe_float(r.get("fiber", np.nan)),
            })

        recs = pd.DataFrame(rows)
        recs = self._apply_constraints_mask(recs, constraints)
        if recs.empty:
            return pd.DataFrame()
        return recs.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)

    def explain(self, user_id: str, recipe_id: str, svd_model, top_similar: int = 5):
        """
        Explains recommendation
        """
        if recipe_id not in self.rid_to_index:
            raise ValueError("recipe_id not found in recipe_master")

        idx = self.rid_to_index[recipe_id]
        recipe_row = self.recipe_master.iloc[idx]
        model = svd_model if svd_model else self.svd

        # SVD score
        svd_score = None
        try:
            svd_score = float(model.predict(str(user_id), str(recipe_id)).est) if model else None
        except Exception:
            svd_score = None

        # nearest neighbors
        distances, indices = self.nn.kneighbors(self.emb[idx].reshape(1, -1), n_neighbors=min(top_similar + 1, len(self.recipe_master)))
        sim_indices = indices.flatten().tolist()
        sim_distances = distances.flatten().tolist()

        similar = []
        for j, ridx in enumerate(sim_indices):
            if ridx == idx: continue
            r = self.recipe_master.iloc[ridx]
            similar.append({
                "recipe_id": str(r["recipe_id"]),
                "recipe_name": r["recipe_name"],
                "similarity": float(1.0 - sim_distances[j])
            })
            if len(similar) >= top_similar: break

        # ingredient overlap
        tokens_target = tokenize_ingredients(recipe_row.get("ingredients_clean", ""))
        overlap_samples = []
        if similar:
            s0_idx = self.rid_to_index.get(similar[0]["recipe_id"])
            if s0_idx is not None:
                s0 = self.recipe_master.iloc[s0_idx]
                tokens_sim = tokenize_ingredients(s0.get("ingredients_clean", ""))
                overlap_samples = sorted(list(tokens_target & tokens_sim))[:25]

        return {
            "recipe": {
                "recipe_id": str(recipe_row["recipe_id"]),
                "recipe_name": recipe_row["recipe_name"],
                "nutrients": {k: float(recipe_row.get(k, np.nan)) for k in TARGET_NUTRIENTS},
            },
            "svd_score_for_user": svd_score,
            "top_similar_recipes": similar,
            "ingredient_overlap_sample": overlap_samples,
        }


# --------------------------- LOADER --------------------------- #
def load_hybrid_engine():
    """
    Loads the cached hybrid recommender artifacts.
    Must exist already from training pipeline.
    """
    required = [
        RECIPE_MASTER_PATH,
        RECIPE_EMBEDDINGS_PATH,
        NEIGHBORS_INDEX_PATH,
        SVD_MODEL_PATH,
        LE_USER_PATH,
        LE_RECIPE_PATH,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing hybrid file: {p}")

    recipe_master = pd.read_parquet(RECIPE_MASTER_PATH)
    recipe_embeddings = np.load(RECIPE_EMBEDDINGS_PATH).astype(np.float32)
    neighbors = joblib.load(NEIGHBORS_INDEX_PATH)

    svd_model = joblib.load(SVD_MODEL_PATH)
    le_user = joblib.load(LE_USER_PATH)
    le_recipe = joblib.load(LE_RECIPE_PATH)

    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)

    return HybridRecommender(
        recipe_master_df=recipe_master,
        embeddings=recipe_embeddings,
        neighbors=neighbors,
        tfidf=joblib.load(TFIDF_VECTORIZER_PATH),
        text_svd=joblib.load(TEXT_SVD_PATH),
        scaler=scaler,
        svd_model=svd_model,
    )
