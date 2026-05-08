"""Faiss 向量存储模块 — 管理 chunk 级向量的增删查"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import faiss


class FaissChunkStore:
    """
    基于 Faiss IndexFlatIP 的 chunk 向量存储。
    由于 sentence-transformers 默认输出归一化向量，
    内积（IP）等价于 cosine similarity。
    """

    def __init__(self, index_path: str = "data/faiss/chunk_index", dim: int = 384):
        self.index_path = Path(index_path)
        self.dim = dim
        self.index_file = self.index_path / "index.faiss"
        self.meta_file = self.index_path / "chunk_meta.json"
        self.index: Optional[faiss.IndexFlatIP] = None
        self.meta: list[dict] = []  # faiss_id -> {chunk_id, doc_id, text}
        self._load()

    def _load(self):
        """从磁盘加载索引和元数据"""
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        if self.meta_file.exists():
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = []

    def _save(self):
        """保存索引和元数据到磁盘"""
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file))
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def add_chunks(self, embeddings: np.ndarray, metadata: list[dict]) -> list[int]:
        """
        添加 chunks 到索引。

        Args:
            embeddings: (n, dim) 的 numpy 数组，已归一化
            metadata: n 个 dict，每个包含 chunk_id, doc_id, text

        Returns:
            分配的 faiss_id 列表
        """
        if len(embeddings) == 0:
            return []

        # 确保是 float32 且二维
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        start_id = self.index.ntotal
        self.index.add(embeddings)
        faiss_ids = list(range(start_id, start_id + len(metadata)))

        for fid, meta in zip(faiss_ids, metadata):
            self.meta.append({"faiss_id": fid, **meta})

        self._save()
        return faiss_ids

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[dict]:
        """
        向量检索，返回 top-k 结果。

        Args:
            query_vec: (dim,) 的查询向量，已归一化
            k: 返回结果数

        Returns:
            结果列表，每个包含 faiss_id, chunk_id, doc_id, text, score
        """
        if self.index.ntotal == 0:
            return []

        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(k, self.index.ntotal)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            meta = self.meta[idx].copy()
            meta["score"] = float(score)
            results.append(meta)

        return results

    def rebuild(self, embeddings: np.ndarray, metadata: list[dict]):
        """全量重建索引（rebuild 模式用）"""
        self.index = faiss.IndexFlatIP(self.dim)
        self.meta = []
        if len(embeddings) > 0:
            self.add_chunks(embeddings, metadata)
        else:
            self._save()

    def clear(self):
        """清空索引"""
        self.index = faiss.IndexFlatIP(self.dim)
        self.meta = []
        self._save()

    def count(self) -> int:
        return self.index.ntotal
