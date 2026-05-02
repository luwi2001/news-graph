import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass


@dataclass
class GraphConfig:
    """图构建配置"""
    window_days: int = 3  # 向前天数
    similarity_threshold: float = 0.7  # 相似度阈值
    max_edges_per_node: int = 3  # 每篇新闻最多产生的边数


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的 cosine similarity"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_days_diff(date1: datetime, date2: datetime) -> int:
    """计算两个日期之间的有符号天数差 (date1 - date2)"""
    return (date1 - date2).days


def is_same_day(date1: datetime, date2: datetime) -> bool:
    """判断两个日期是否是同一天 (均为 naive UTC datetime)"""
    return date1.date() == date2.date()


def build_edges(
    embeddings: np.ndarray,
    pub_dates: list[datetime],
    config: GraphConfig
) -> list[tuple[int, int, float]]:
    """根据 embeddings 和时间窗口构建边（双向度数限制）"""
    edges = []
    n = len(embeddings)
    degree_count = [0] * n  # 跟踪每个节点的当前度数

    for i in range(n):
        current_date = pub_dates[i]
        current_vec = embeddings[i]
        candidate_edges = []

        for j in range(i + 1, n):
            target_date = pub_dates[j]
            days_diff = compute_days_diff(current_date, target_date)

            # 修复：取绝对值判断时间窗口
            if abs(days_diff) > config.window_days:
                continue

            if is_same_day(current_date, target_date):
                continue

            sim = compute_cosine_similarity(current_vec, embeddings[j])

            if sim >= config.similarity_threshold:
                candidate_edges.append((j, sim))

        candidate_edges.sort(key=lambda x: x[1], reverse=True)
        for j, sim in candidate_edges:
            # 双向度数限制：source 和 target 都不能超过上限
            if degree_count[i] >= config.max_edges_per_node:
                break
            if degree_count[j] >= config.max_edges_per_node:
                continue
            edges.append((i, j, sim))
            degree_count[i] += 1
            degree_count[j] += 1

    return edges


class NewsGraphBuilder:
    """新闻图构建器"""

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()

    def build_graph(
        self,
        embeddings: np.ndarray,
        pub_dates: list[datetime],
        entry_ids: list[int],
        titles: list[str]
    ) -> nx.Graph:
        """
        构建新闻图

        Args:
            embeddings: 所有新闻的 embedding 向量
            pub_dates: 新闻发布时间
            entry_ids: 新闻 ID 列表
            titles: 新闻标题列表

        Returns:
            NetworkX 图
        """
        G = nx.Graph()

        # 添加节点
        for i, (entry_id, title, date) in enumerate(zip(entry_ids, titles, pub_dates)):
            G.add_node(i, id=entry_id, title=title, pub_date=date.isoformat())

        # 构建边
        edges = build_edges(embeddings, pub_dates, self.config)

        # 添加边
        for src, tgt, sim in edges:
            G.add_edge(src, tgt, weight=sim)

        return G


class IncrementalGraphBuilder:
    """增量图构建器 - 为单个节点计算边"""

    def __init__(self, config: GraphConfig):
        self.config = config

    def compute_candidate_edges_for_node(
        self,
        new_vec: np.ndarray,
        new_date: datetime,
        all_embeddings: np.ndarray,
        all_pub_dates: list[datetime]
    ) -> list[tuple[int, float]]:
        """为单个新节点计算候选边"""
        candidate_edges = []

        for j, (existing_vec, existing_date) in enumerate(zip(all_embeddings, all_pub_dates)):
            days_diff = compute_days_diff(new_date, existing_date)

            if abs(days_diff) > self.config.window_days:
                continue
            if days_diff <= 0:  # 新节点不比已有节点晚
                continue

            if is_same_day(new_date, existing_date):
                continue

            sim = compute_cosine_similarity(new_vec, existing_vec)

            if sim >= self.config.similarity_threshold:
                candidate_edges.append((j, sim))

        candidate_edges.sort(key=lambda x: x[1], reverse=True)
        return candidate_edges[:self.config.max_edges_per_node]


def add_incremental_edges(
        self,
        G: nx.Graph,
        new_embeddings: np.ndarray,
        new_pub_dates: list[datetime],
        new_entry_ids: list[int],
        new_titles: list[str],
        all_embeddings: np.ndarray,
        all_pub_dates: list[datetime]
    ) -> nx.Graph:
        """
        增量添加新边 - 为新新闻向前 N 天构建边

        Args:
            G: 现有图
            new_embeddings: 新增新闻的 embeddings
            new_pub_dates: 新增新闻的时间
            new_entry_ids: 新增新闻的 ID
            new_titles: 新增新闻的标题
            all_embeddings: 所有历史 news embeddings (用于计算相似度)
            all_pub_dates: 所有历史新闻时间

        Returns:
            更新后的图
        """
        existing_count = G.number_of_nodes()

        # 添加新节点
        for i, (entry_id, title, date) in enumerate(zip(new_entry_ids, new_titles, new_pub_dates)):
            G.add_node(existing_count + i, id=entry_id, title=title, pub_date=date.isoformat())

        # 新新闻与历史新闻构建边
        all_edges = build_edges(all_embeddings, all_pub_dates, self.config)

        # 只添加涉及新节点的边
        start_idx = len(all_embeddings) - len(new_embeddings)
        for src, tgt, sim in all_edges:
            if src >= start_idx or tgt >= start_idx:
                G.add_edge(src, tgt, weight=sim)

        return G