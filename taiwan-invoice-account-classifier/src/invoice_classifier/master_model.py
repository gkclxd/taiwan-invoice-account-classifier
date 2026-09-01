# -*- coding: utf-8 -*-
"""
invoice_classifier.master_model
=================================
母版模型（Master Model）：整合自 M2-1 embedder.py / train_master_model.py。

設計：
    母版模型不是傳統意義上的「分類器權重」，而是「科目原型向量庫」：
        1. 使用向量化器（TF-IDF+SVD 或 Sentence Transformer）將訓練集內
           每筆發票摘要轉為向量。
        2. 依「歷史正確科目」分組，計算各科目的平均向量（原型向量）。
        3. 推論時，將新摘要向量化後，與各科目原型向量計算 cosine similarity，
           取最相似（最近原型）者作為母版模型分數來源。
    此設計可解釋（能說明「與哪個科目的歷史案例最相似」）、增量友善
    （新增訓練資料只需重算受影響科目的原型向量，不需重新訓練分類器）。

⚠️ 本檔案僅包含「訓練 / 評估 / 原型向量計算」邏輯本身，不包含已訓練完成
    的模型檔案（.pkl）。請執行 scripts 或本模組的 train_from_csv() /
    __main__ 區塊，於本機使用你自己的訓練資料重建模型。

向量化後端：
    預設 "tfidf_svd"（jieba 中文分詞 + TF-IDF + TruncatedSVD 降維），
    完全離線運算，不依賴外部網路。同時保留 "sentence_transformer" 後端
    介面（需可連線 huggingface.co 下載模型權重），供未來網路環境開通後，
    以 backend="auto" 自動優先嘗試、失敗則退回離線方案。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np


# ============================================================================
# 向量化後端（獨立於 preprocessor.InvoiceVectorizer，供母版模型訓練管線
# 直接對「原始摘要」做端對端向量化與斷詞，介面與 preprocessor.py 一致，
# 以保持與現有訓練腳本相容）
# ============================================================================

import re

_PUNCT_RE = re.compile(r"[\s\-\(\)\*\.,，。、\(\)（）／/\\]+")

try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JIEBA_AVAILABLE = False


def _tokenize_zh(text: str) -> str:
    """中文分詞（jieba），回傳以空白分隔的詞彙字串供 TfidfVectorizer 使用。"""
    text = _PUNCT_RE.sub(" ", text)
    if _JIEBA_AVAILABLE:
        tokens = [t for t in jieba.cut(text) if t.strip()]
    else:  # pragma: no cover
        tokens = list(text.replace(" ", ""))
    return " ".join(tokens)


def _split_on_space(s: str) -> list:
    return s.split(" ")


class TfidfSvdEmbedder:
    """離線可用的向量化後端：jieba 分詞 + TF-IDF（詞級+字元 n-gram）+ SVD 降維。"""

    BACKEND_NAME = "tfidf_svd"

    def __init__(self, embedding_dim: int = 384, random_state: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.embedding_dim = embedding_dim
        self.random_state = random_state

        self.word_vectorizer = TfidfVectorizer(
            tokenizer=_split_on_space,
            preprocessor=_tokenize_zh,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3),
            min_df=1,
            sublinear_tf=True,
        )
        self.svd: Optional[object] = None
        self._fitted = False

    def fit(self, texts: List[str]) -> "TfidfSvdEmbedder":
        from scipy.sparse import hstack
        from sklearn.decomposition import TruncatedSVD

        word_matrix = self.word_vectorizer.fit_transform(texts)
        char_matrix = self.char_vectorizer.fit_transform(texts)
        combined = hstack([word_matrix, char_matrix]).tocsr()

        n_features = combined.shape[1]
        n_components = min(self.embedding_dim, max(2, n_features - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        self.svd.fit(combined)
        self._fitted = True
        return self

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfSvdEmbedder 尚未 fit()，請先呼叫 fit() 或載入已訓練的向量化器。")
        from scipy.sparse import hstack

        word_matrix = self.word_vectorizer.transform(texts)
        char_matrix = self.char_vectorizer.transform(texts)
        combined = hstack([word_matrix, char_matrix]).tocsr()
        vecs = self.svd.transform(combined)
        return vecs.astype(np.float32)

    def save(self, path) -> None:
        joblib.dump({
            "word_vectorizer": self.word_vectorizer,
            "char_vectorizer": self.char_vectorizer,
            "svd": self.svd,
            "embedding_dim": self.embedding_dim,
            "random_state": self.random_state,
        }, path, compress=3)

    @classmethod
    def load(cls, path) -> "TfidfSvdEmbedder":
        bundle = joblib.load(path)
        obj = cls(embedding_dim=bundle["embedding_dim"], random_state=bundle["random_state"])
        obj.word_vectorizer = bundle["word_vectorizer"]
        obj.char_vectorizer = bundle["char_vectorizer"]
        obj.svd = bundle["svd"]
        obj._fitted = True
        return obj


class SentenceTransformerEmbedder:
    """可選方案：Sentence Transformer（需可連線 huggingface.co 下載權重）。"""

    BACKEND_NAME = "sentence_transformer"

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"無法載入 Sentence Transformer 模型 '{model_name}'（可能是網路無法連線 "
                f"huggingface.co）。請改用 embedding_backend='tfidf_svd'，"
                f"或確認網路環境已開通 huggingface.co 存取。原始錯誤：{exc}"
            ) from exc

    def encode(self, texts: List[str], batch_size: int = 64, show_progress_bar: bool = False,
               convert_to_numpy: bool = True) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
        ).astype(np.float32)


def build_embedder(backend: str = "tfidf_svd", **kwargs):
    """
    工廠函式：依 backend 名稱建立對應的 embedder。

    backend:
        "tfidf_svd"            -> TfidfSvdEmbedder（離線，預設）
        "sentence_transformer" -> SentenceTransformerEmbedder（需外部網路）
        "auto"                 -> 優先嘗試 sentence_transformer，失敗則自動改用 tfidf_svd
    """
    if backend == "tfidf_svd":
        return TfidfSvdEmbedder(**kwargs)
    if backend == "sentence_transformer":
        return SentenceTransformerEmbedder(**kwargs)
    if backend == "auto":
        try:
            model_name = kwargs.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
            return SentenceTransformerEmbedder(model_name=model_name)
        except RuntimeError as exc:
            print(f"[master_model] {exc}\n[master_model] 自動改用離線 tfidf_svd 後端。")
            dim = kwargs.get("embedding_dim", 384)
            return TfidfSvdEmbedder(embedding_dim=dim)
    raise ValueError(f"未知的 embedding backend: {backend}")


# ============================================================================
# 母版模型：科目原型向量庫
# ============================================================================

@dataclass
class MasterModel:
    """母版模型：科目代碼 -> 平均原型向量（訓練完成後由 fusion_engine 讀取使用）。"""

    account_codes: List[str]
    account_vectors: Dict[str, np.ndarray]   # account_code -> vector（已正規化）
    account_names: Dict[str, str] = field(default_factory=dict)
    embedding_backend: str = "tfidf_svd"
    embedding_dim: int = 384
    embedder: Optional[Any] = None           # 已 fit 的 embedder 實例（推論時用於向量化新摘要）
    trained_at: Optional[str] = None
    n_training_samples: int = 0
    cv_metrics: Dict[str, float] = field(default_factory=dict)

    def encode(self, text: str) -> np.ndarray:
        """將摘要文字轉為向量。若無已 fit 的 embedder，拋出明確錯誤（提示需先訓練/載入模型）。"""
        if self.embedder is None:
            raise RuntimeError(
                "MasterModel 尚未附帶已訓練的向量化器（embedder）。"
                "請先呼叫 train_from_csv() 訓練，或使用 MasterModel.load() 載入已訓練模型。"
            )
        vec = self.embedder.encode([text])
        return _normalize(np.asarray(vec[0], dtype=float))

    def save(self, path) -> None:
        """
        儲存母版模型（原型向量 + 中繼資料）。

        注意：本函式僅供使用者於自己的環境訓練/保存自己的模型使用；
        本 repo 不會提交任何已訓練完成的模型二進位檔（見 .gitignore）。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "embedding_backend": self.embedding_backend,
            "embedding_dim": self.embedding_dim,
            "account_vectors": self.account_vectors,
            "account_codes": self.account_codes,
            "account_names": self.account_names,
            "trained_at": self.trained_at,
            "n_training_samples": self.n_training_samples,
            "cv_metrics": self.cv_metrics,
        }
        joblib.dump(bundle, path, compress=3)

        if self.embedding_backend == "tfidf_svd" and self.embedder is not None:
            vectorizer_path = path.parent / f"{path.stem}_vectorizer.pkl"
            self.embedder.save(vectorizer_path)

    @classmethod
    def load(cls, model_path, vectorizer_path=None) -> "MasterModel":
        """
        載入已訓練完成的母版模型。

        model_path : master_model.pkl 的路徑（使用者自行執行 train_from_csv() 產生）
        vectorizer_path : tfidf_svd 後端需額外提供向量化器路徑
                          （預設猜測為 "{model_path 檔名}_vectorizer.pkl"）
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到母版模型檔案：{model_path}。"
                f"本 repo 不包含已訓練模型，請先執行訓練腳本"
                f"（可參考 docs/ 或 examples/ 內的說明）產生此檔案。"
            )
        bundle = joblib.load(model_path)
        backend = bundle.get("embedding_backend", "tfidf_svd")

        embedder = None
        if backend == "tfidf_svd":
            vpath = Path(vectorizer_path) if vectorizer_path else model_path.parent / f"{model_path.stem}_vectorizer.pkl"
            if vpath.exists():
                embedder = TfidfSvdEmbedder.load(vpath)
        elif backend == "sentence_transformer":
            model_name = bundle.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
            try:
                embedder = SentenceTransformerEmbedder(model_name=model_name)
            except RuntimeError:
                embedder = None

        account_vectors = {
            code: np.asarray(vec, dtype=float)
            for code, vec in zip(bundle["account_codes"], bundle["account_vectors"])
        } if isinstance(bundle["account_vectors"], np.ndarray) else bundle["account_vectors"]

        account_names = bundle.get("account_names", {})
        if isinstance(account_names, list):
            account_names = dict(zip(bundle["account_codes"], account_names))

        return cls(
            account_codes=list(bundle["account_codes"]),
            account_vectors=account_vectors,
            account_names=account_names,
            embedding_backend=backend,
            embedding_dim=bundle.get("embedding_dim", 384),
            embedder=embedder,
            trained_at=bundle.get("trained_at"),
            n_training_samples=bundle.get("n_training_samples", 0),
            cv_metrics=bundle.get("cv_metrics", {}),
        )


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def build_prototype_vectors(
    embeddings: np.ndarray, labels: np.ndarray
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """依標籤群組計算每個會計科目的平均向量（原型向量），回傳正規化後的字典與代碼清單。"""
    codes = sorted(set(labels))
    vectors: Dict[str, np.ndarray] = {}
    for code in codes:
        mask = labels == code
        mean_vec = embeddings[mask].mean(axis=0)
        vectors[code] = _normalize(mean_vec.astype(np.float64))
    return vectors, codes


def cosine_predict(
    query_vecs: np.ndarray, proto_vecs: np.ndarray, proto_codes: List[str], top_k: int = 3
):
    """以餘弦相似度做最近原型分類，回傳 top-1 預測與 top-k 候選清單。"""
    q_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
    p_norm = proto_vecs / (np.linalg.norm(proto_vecs, axis=1, keepdims=True) + 1e-8)
    sims = q_norm @ p_norm.T
    top1_idx = np.argmax(sims, axis=1)
    top1_pred = [proto_codes[i] for i in top1_idx]

    topk_idx = np.argsort(-sims, axis=1)[:, :top_k]
    topk_pred = [[proto_codes[j] for j in row] for row in topk_idx]
    return top1_pred, topk_pred, sims


def run_cross_validation(
    texts: np.ndarray, labels: np.ndarray, embedder_backend: str = "tfidf_svd",
    embedding_dim: int = 384, n_splits: int = 5, random_state: int = 42,
) -> Dict[str, Any]:
    """
    5-fold Stratified K-Fold 交叉驗證。向量化器（TF-IDF/SVD 或 Sentence
    Transformer）僅在每個 fold 的訓練集上 fit，測試集僅做 transform，
    避免資料洩漏。
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accuracies, fold_f1s, fold_top3s = [], [], []
    all_true, all_pred = [], []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(texts, labels), start=1):
        train_texts, train_lbl = texts[train_idx], labels[train_idx]
        test_texts, test_lbl = texts[test_idx], labels[test_idx]

        fold_embedder = build_embedder(embedder_backend, embedding_dim=embedding_dim)
        if hasattr(fold_embedder, "fit"):
            fold_embedder.fit(list(train_texts))
        train_emb = fold_embedder.encode(list(train_texts))
        test_emb = fold_embedder.encode(list(test_texts))

        proto_dict, proto_codes = build_prototype_vectors(train_emb, train_lbl)
        proto_vecs = np.array([proto_dict[c] for c in proto_codes])
        top1_pred, topk_pred, _ = cosine_predict(test_emb, proto_vecs, proto_codes, top_k=3)

        acc = accuracy_score(test_lbl, top1_pred)
        f1 = f1_score(test_lbl, top1_pred, average="macro", zero_division=0)
        top3_hits = sum(1 for true, cands in zip(test_lbl, topk_pred) if true in cands)
        top3_acc = top3_hits / len(test_lbl)

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        fold_top3s.append(top3_acc)
        all_true.extend(test_lbl.tolist())
        all_pred.extend(top1_pred)

        print(f"  Fold {fold_i}/{n_splits}: accuracy={acc:.4f}  macro-F1={f1:.4f}  top3_acc={top3_acc:.4f}")

    codes_sorted = sorted(set(labels))
    cm = confusion_matrix(all_true, all_pred, labels=codes_sorted)
    report_txt = classification_report(all_true, all_pred, labels=codes_sorted, zero_division=0)

    return {
        "fold_accuracies": fold_accuracies,
        "fold_f1s": fold_f1s,
        "fold_top3s": fold_top3s,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "mean_f1": float(np.mean(fold_f1s)),
        "mean_top3": float(np.mean(fold_top3s)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "confusion_matrix": cm,
        "codes_sorted": codes_sorted,
        "classification_report_txt": report_txt,
    }


def train_from_csv(
    csv_path,
    account_subjects_path=None,
    embedding_backend: str = "tfidf_svd",
    embedding_dim: int = 384,
    run_cv: bool = True,
) -> MasterModel:
    """
    從 CSV 訓練資料訓練母版模型。

    csv_path 需含欄位：invoice_id, summary, seller_ban, buyer_ban,
                        account_code, account_name

    重要：本 repo 不提供任何歷史發票資料（historical_invoices.csv 因涉及
    真實客戶資料已排除），使用者需自行準備符合上述格式的訓練資料
    （可參考 examples/sample_invoices.csv 之格式，但該檔案筆數過少，
    僅供介面示範，正式訓練請使用具代表性的自有資料）。
    """
    import pandas as pd

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype={"seller_ban": str, "buyer_ban": str, "account_code": str})
    required_cols = {"summary", "account_code"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"訓練資料缺少必要欄位：{missing}")
    df = df.dropna(subset=["summary", "account_code"]).reset_index(drop=True)
    df["account_code"] = df["account_code"].astype(str).str.zfill(4)

    texts = df["summary"].values
    labels = df["account_code"].values

    cv_results = None
    if run_cv and len(set(labels)) > 1 and len(df) >= 10:
        print(f"執行 {5}-fold 交叉驗證...")
        cv_results = run_cross_validation(texts, labels, embedding_backend, embedding_dim)
        print(f"平均 Accuracy: {cv_results['mean_accuracy']:.4f}")

    embedder = build_embedder(embedding_backend, embedding_dim=embedding_dim)
    if hasattr(embedder, "fit"):
        embedder.fit(list(texts))
    embeddings = embedder.encode(list(texts))
    actual_backend = getattr(embedder, "BACKEND_NAME", embedding_backend)

    account_vectors, account_codes = build_prototype_vectors(embeddings, labels)

    account_names: Dict[str, str] = {}
    if account_subjects_path and Path(account_subjects_path).exists():
        subj_df = pd.read_csv(account_subjects_path, dtype={"account_code": str})
        subj_df["account_code"] = subj_df["account_code"].astype(str).str.zfill(4)
        account_names = dict(zip(subj_df["account_code"], subj_df["account_name"]))
    for code, name in zip(df["account_code"], df.get("account_name", [""] * len(df))):
        account_names.setdefault(code, name)

    import datetime as _dt

    model = MasterModel(
        account_codes=account_codes,
        account_vectors=account_vectors,
        account_names=account_names,
        embedding_backend=actual_backend,
        embedding_dim=embeddings.shape[1],
        embedder=embedder,
        trained_at=_dt.datetime.now().isoformat(),
        n_training_samples=len(df),
        cv_metrics={
            "mean_accuracy": cv_results["mean_accuracy"],
            "mean_f1": cv_results["mean_f1"],
            "mean_top3": cv_results["mean_top3"],
        } if cv_results else {},
    )
    return model


__all__ = [
    "MasterModel",
    "build_embedder",
    "TfidfSvdEmbedder",
    "SentenceTransformerEmbedder",
    "build_prototype_vectors",
    "cosine_predict",
    "run_cross_validation",
    "train_from_csv",
]
