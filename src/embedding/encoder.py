from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional
import pickle
import os


class EmbeddingEncoder:
    """Embedding 编码器 - 使用 Sentence-Transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        初始化编码器

        Args:
            model_name: Sentence-Transformers 模型名
            cache_dir: embedding 缓存目录
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or "data/embeddings"
        self.model = SentenceTransformer(model_name)
        os.makedirs(self.cache_dir, exist_ok=True)

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """将文本列表转为向量"""
        return self.model.encode(texts, show_progress_bar=show_progress)

    def encode_single(self, text: str) -> np.ndarray:
        """将单个文本转为向量"""
        return self.model.encode([text])[0]

    def get_text_for_entry(self, title: str, description: Optional[str] = None) -> str:
        """拼接 title 和 description 为模型输入"""
        if description:
            return f"{title}: {description[:500]}"  # 截取部分内容避免过长
        return title

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """保存 embeddings 到缓存"""
        filepath = os.path.join(self.cache_dir, filepath)
        with open(filepath, "wb") as f:
            pickle.dump(embeddings, f)

    def load_embeddings(self, filepath: str) -> Optional[np.ndarray]:
        """从缓存加载 embeddings"""
        filepath = os.path.join(self.cache_dir, filepath)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None