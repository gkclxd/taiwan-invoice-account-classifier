# -*- coding: utf-8 -*-
"""
invoice_classifier.preprocessor
=================================
發票摘要（繁體中文）前處理與向量化管線（整合自 M2-2 preprocessor.py + vectorizer.py）。

    原始摘要 (str) ──► 斷詞 (jieba + 自訂辭典) ──► 停用詞過濾 ──► 向量化 ──► np.ndarray

設計原則：
    1. 使用 jieba 斷詞，並載入 assets/custom_dict.txt（會計科目 / 稅務 /
       貿易 / 行業專有名詞），避免「進項稅額」「報關費」等關鍵詞被拆散成
       不具語意的單字，這些詞彙是後續稅法合規規則比對（關鍵字觸發）與
       向量分類的關鍵特徵。
    2. 停用詞過濾採「精簡白名單式排除」：只移除的、了、在等虛詞，
       數字、金額、百分比（如「5%」「NT$1,200」）一律保留，
       因為金額與稅率數字可能是稅法規則觸發的依據。
    3. 向量化預設使用 TF-IDF + SVD 離線方案（tfidf_svd），不依賴外部網路；
       亦保留 sentence_transformer 後端介面，供未來網路環境開通後切換
       （見 build_vectorizer 的 backend="auto"）。

輸入驗證：
    InvoicePreprocessor.preprocess() 對空字串 / 非字串輸入採保守處理
    （回傳空字串，不拋例外），避免單筆髒資料中斷整批處理；
    向量化層則對未 fit 的向量器呼叫 encode() 明確拋出 RuntimeError。
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

try:
    import jieba
    _JIEBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JIEBA_AVAILABLE = False

import joblib


_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CUSTOM_DICT_PATH = _THIS_DIR / "assets" / "custom_dict.txt"
DEFAULT_STOPWORDS_PATH = _THIS_DIR / "assets" / "stopwords.txt"

# 千分位逗號（如「1,200」）需先保護起來，避免被標點清除規則拆散成
# "1" "200" 兩個獨立數字，導致金額語意流失。
_THOUSANDS_SEP_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

# 會被清成空白的符號：括號、連字號、標點、全形空白等「純分隔用」符號
_PUNCT_RE = re.compile(r"[\s\-\(\)\*（）\[\]【】「」『』、,，。;；:：!！?？/\\|~＿_]+")

# 保留的金額 / 百分比模式
_NUMBER_LIKE_RE = re.compile(r"^[\d,，.]+%?$|^NT\$?[\d,，.]+$|^\$[\d,，.]+$", re.IGNORECASE)


def _load_stopwords(path: Path) -> Set[str]:
    words: Set[str] = set()
    if not path.exists():
        return words
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.add(line)
    return words


def _looks_like_number(token: str) -> bool:
    """判斷 token 是否為數字/金額/百分比（這類 token 一律保留，不進停用詞過濾）。"""
    return bool(_NUMBER_LIKE_RE.match(token))


class InvoicePreprocessor:
    """
    發票摘要前處理器：斷詞 + 停用詞過濾。

    Parameters
    ----------
    custom_dict_path : 自定義辭典路徑（預設使用 assets/custom_dict.txt）
    stopwords_path : 停用詞表路徑（預設使用 assets/stopwords.txt）
    extra_custom_words : 額外要動態加入 jieba 辭典的詞（例如個別公司特有用語）
    keep_numbers : 是否保留數字/金額類 token（預設 True，稅法規則常需比對金額與稅率）
    """

    def __init__(
        self,
        custom_dict_path: Optional[Path] = DEFAULT_CUSTOM_DICT_PATH,
        stopwords_path: Optional[Path] = DEFAULT_STOPWORDS_PATH,
        extra_custom_words: Optional[List[str]] = None,
        keep_numbers: bool = True,
    ):
        self.keep_numbers = keep_numbers
        self._stopwords = _load_stopwords(Path(stopwords_path)) if stopwords_path else set()

        if _JIEBA_AVAILABLE:
            # 使用獨立的 jieba.Tokenizer 實例，避免污染全域 jieba 狀態
            # （多個公司/多個 preprocessor 實例可各自載入不同的自定義辭典）
            self._tokenizer = jieba.Tokenizer()
            if custom_dict_path and Path(custom_dict_path).exists():
                self._tokenizer.load_userdict(str(custom_dict_path))
            if extra_custom_words:
                for w in extra_custom_words:
                    self._tokenizer.add_word(w, freq=100000)
            self._tokenizer.initialize()
        else:  # pragma: no cover
            self._tokenizer = None

    def _tokenize(self, text: str) -> List[str]:
        if self._tokenizer is not None:
            return [t for t in self._tokenizer.cut(text) if t.strip()]
        # jieba 不可用時的退化方案：以空白切分
        return [t for t in re.split(r"\s+", text) if t.strip()]

    def preprocess(self, summary: Optional[str]) -> str:
        """
        對單筆發票摘要進行斷詞與停用詞過濾。

        對空字串 / None / 非字串輸入採保守處理（回傳空字串），
        不拋出例外，避免單筆髒資料中斷整批處理。
        """
        if not summary or not isinstance(summary, str) or not summary.strip():
            return ""

        text = _THOUSANDS_SEP_RE.sub("", summary.strip())  # 保護千分位逗號：1,200 -> 1200
        text = _PUNCT_RE.sub(" ", text)
        tokens = self._tokenize(text)

        kept: List[str] = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if self.keep_numbers and _looks_like_number(tok):
                kept.append(tok)
                continue
            if tok in self._stopwords:
                continue
            kept.append(tok)

        return " ".join(kept)

    def preprocess_batch(self, summaries: List[str]) -> List[str]:
        """批次處理多筆摘要，回傳對應的預處理後字串清單。"""
        return [self.preprocess(s) for s in summaries]


_default_preprocessor: Optional[InvoicePreprocessor] = None


def _get_default_preprocessor() -> InvoicePreprocessor:
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = InvoicePreprocessor()
    return _default_preprocessor


def preprocess_summary(summary: str) -> str:
    """函式介面版本：輸入原始摘要 str，輸出預處理後摘要 str（內部使用模組單例）。"""
    return _get_default_preprocessor().preprocess(summary)


# ============================================================================
# 向量化（整合自 M2-2 vectorizer.py）
# ============================================================================

def _split_on_space(s: str) -> list:
    """模組層級函式（可被 pickle），取代 lambda 作為 TfidfVectorizer 的 tokenizer。"""
    return s.split(" ")


def _identity_preprocessor(s: str) -> str:
    """輸入已是 InvoicePreprocessor 處理過的『空白分隔字串』，此處不再重複斷詞。"""
    return s


class TfidfSvdVectorizer:
    """
    離線可用的向量化後端：TF-IDF（詞級 + 字元 n-gram）+ SVD 降維。

    輸入須為「已完成斷詞、以空白分隔的字串」（即 InvoicePreprocessor.preprocess()
    的輸出），本類別本身不再進行分詞，只負責向量化，職責與 preprocessor 分離。
    """

    BACKEND_NAME = "tfidf_svd"

    def __init__(self, embedding_dim: int = 384, random_state: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.embedding_dim = embedding_dim
        self.random_state = random_state

        self.word_vectorizer = TfidfVectorizer(
            tokenizer=_split_on_space,
            preprocessor=_identity_preprocessor,
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

    def fit(self, texts: List[str]) -> "TfidfSvdVectorizer":
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

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "TfidfSvdVectorizer 尚未 fit()。請先呼叫 .fit(訓練用摘要清單)，"
                "或載入已訓練好的向量化器（.load()）。"
            )
        from scipy.sparse import hstack

        word_matrix = self.word_vectorizer.transform(texts)
        char_matrix = self.char_vectorizer.transform(texts)
        combined = hstack([word_matrix, char_matrix]).tocsr()
        vecs = self.svd.transform(combined)

        if vecs.shape[1] < self.embedding_dim:
            pad = np.zeros((vecs.shape[0], self.embedding_dim - vecs.shape[1]), dtype=np.float32)
            vecs = np.hstack([vecs, pad])
        return vecs.astype(np.float32)

    def save(self, path) -> None:
        joblib.dump({
            "word_vectorizer": self.word_vectorizer,
            "char_vectorizer": self.char_vectorizer,
            "svd": self.svd,
            "embedding_dim": self.embedding_dim,
            "random_state": self.random_state,
        }, Path(path), compress=3)

    @classmethod
    def load(cls, path) -> "TfidfSvdVectorizer":
        bundle = joblib.load(Path(path))
        obj = cls(embedding_dim=bundle["embedding_dim"], random_state=bundle["random_state"])
        obj.word_vectorizer = bundle["word_vectorizer"]
        obj.char_vectorizer = bundle["char_vectorizer"]
        obj.svd = bundle["svd"]
        obj._fitted = True
        return obj


class SentenceTransformerVectorizer:
    """
    可選方案：Sentence Transformer。需可連線 huggingface.co 下載模型權重；
    若目前環境無法連線，__init__ 會拋出附帶說明的 RuntimeError，
    供上層（build_vectorizer）捕捉並自動改用 TfidfSvdVectorizer。
    """

    BACKEND_NAME = "sentence_transformer"

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"無法載入 Sentence Transformer 模型 '{model_name}'。"
                f"最常見原因是目前網路環境無法存取 huggingface.co。"
                f"請改用 backend='tfidf_svd'，或於網路環境開通後重試。原始錯誤：{exc}"
            ) from exc

    def fit(self, texts: List[str]) -> "SentenceTransformerVectorizer":
        return self  # 預訓練模型，不需 fit，保留介面一致性

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True,
        ).astype(np.float32)

    def save(self, path) -> None:
        joblib.dump({"model_name": self.model_name}, Path(path))

    @classmethod
    def load(cls, path) -> "SentenceTransformerVectorizer":
        bundle = joblib.load(Path(path))
        return cls(model_name=bundle["model_name"])


def build_vectorizer(backend: str = "auto", embedding_dim: int = 384, **kwargs):
    """
    工廠函式：依 backend 名稱建立對應的向量化器。

    backend:
        "sentence_transformer" -> SentenceTransformerVectorizer（需外部網路）
        "tfidf_svd"             -> TfidfSvdVectorizer（離線，預設可用方案）
        "auto"                  -> 優先嘗試 sentence_transformer，失敗則自動改用 tfidf_svd
    """
    if backend == "sentence_transformer":
        return SentenceTransformerVectorizer(**kwargs)
    if backend == "tfidf_svd":
        return TfidfSvdVectorizer(embedding_dim=embedding_dim, **kwargs)
    if backend == "auto":
        model_name = kwargs.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
        try:
            return SentenceTransformerVectorizer(model_name=model_name)
        except RuntimeError as exc:
            print(f"[preprocessor] {exc}\n[preprocessor] 自動改用離線 tfidf_svd 後端。")
            return TfidfSvdVectorizer(embedding_dim=embedding_dim)
    raise ValueError(f"未知的向量化 backend: {backend}")


class InvoiceVectorizer:
    """
    對外統一介面：包裝底層 backend，提供
    「輸入預處理後字串 -> 輸出固定維度 numpy array」介面，
    並記錄計時供效能驗證。
    """

    def __init__(self, backend: str = "tfidf_svd", embedding_dim: int = 384,
                 model_path: Optional[Path] = None, **kwargs):
        self.embedding_dim = embedding_dim
        if model_path and Path(model_path).exists():
            self._impl = self._load_impl(model_path, backend)
        else:
            self._impl = build_vectorizer(backend=backend, embedding_dim=embedding_dim, **kwargs)
        self.backend_name = getattr(self._impl, "BACKEND_NAME", backend)

    @staticmethod
    def _load_impl(model_path, backend: str):
        if backend == "sentence_transformer":
            return SentenceTransformerVectorizer.load(model_path)
        return TfidfSvdVectorizer.load(model_path)

    def fit(self, texts: List[str]) -> "InvoiceVectorizer":
        """僅 tfidf_svd 後端需要；sentence_transformer 為 no-op。"""
        self._impl.fit(texts)
        return self

    def vectorize(self, preprocessed_summary: str) -> np.ndarray:
        """單筆向量化，回傳 shape = (embedding_dim,) 的 float32 向量。"""
        vec = self._impl.encode([preprocessed_summary])
        return vec[0]

    def vectorize_batch(self, preprocessed_summaries: List[str]) -> np.ndarray:
        """批次向量化，回傳 shape = (n, embedding_dim) 的 float32 矩陣。"""
        return self._impl.encode(preprocessed_summaries)

    def encode(self, text) -> np.ndarray:
        """相容別名：接受單一字串或字串清單。"""
        if isinstance(text, str):
            return self.vectorize(text)
        return self.vectorize_batch(list(text))

    def save(self, path) -> None:
        self._impl.save(Path(path))


def benchmark_latency(vectorizer: InvoiceVectorizer, sample_text: str, n_runs: int = 20) -> dict:
    """量測單筆向量化的平均耗時（秒），用於驗證效能約束。"""
    vectorizer.vectorize(sample_text)  # 暖機一次
    t0 = time.perf_counter()
    for _ in range(n_runs):
        vectorizer.vectorize(sample_text)
    elapsed = time.perf_counter() - t0
    return {"avg_seconds_per_call": elapsed / n_runs, "n_runs": n_runs}


__all__ = [
    "InvoicePreprocessor",
    "preprocess_summary",
    "InvoiceVectorizer",
    "TfidfSvdVectorizer",
    "SentenceTransformerVectorizer",
    "build_vectorizer",
    "benchmark_latency",
]
