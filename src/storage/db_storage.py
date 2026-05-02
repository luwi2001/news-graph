"""图数据库存储模块"""

import sqlite3
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

DB_PATH = "data/news_graph.db"


def init_db(conn: sqlite3.Connection):
    """初始化数据库表"""
    schema = Path(__file__).parent.parent / "schema.sql"
    if schema.exists():
        conn.exec_script(schema.read_text())


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """获取数据库连接"""
    import os
    db_path = os.path.expanduser(db_path)
    os.makedirs(Path(db_path).parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def save_node(conn: sqlite3.Connection, node_data: dict):
    """保存节点"""
    conn.execute("""
        INSERT OR REPLACE INTO node (id, graph_id, title, link, pub_date, description, feed_id, in_graph)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        node_data["id"],
        node_data.get("graph_id", 1),
        node_data["title"],
        node_data.get("link"),
        node_data.get("pub_date"),
        node_data.get("description"),
        node_data.get("feed_id"),
        node_data.get("in_graph", 0)
    ))


def save_embedding(conn: sqlite3.Connection, node_id: int, embedding: np.ndarray):
    """保存节点的 embedding 向量"""
    conn.execute("""
        UPDATE node SET embedding = ? WHERE id = ?
    """, (pickle.dumps(embedding), node_id))


def save_edge(conn: sqlite3.Connection, source_id: int, target_id: int, weight: float, graph_id: int = 1):
    """保存边"""
    try:
        conn.execute("""
            INSERT OR IGNORE INTO edge (source_id, target_id, weight, graph_id)
            VALUES (?, ?, ?, ?)
        """, (source_id, target_id, weight, graph_id))
    except sqlite3.IntegrityError:
        pass


def get_max_node_id(conn: sqlite3.Connection, graph_id: int = 1) -> Optional[int]:
    """获取图中最大节点 ID"""
    cursor = conn.execute("""
        SELECT MAX(id) FROM node WHERE graph_id = ? AND in_graph = 1
    """, (graph_id,))
    return cursor.fetchone()[0]


def get_all_embeddings(conn: sqlite3.Connection, graph_id: int = 1) -> list:
    """获取所有节点及其 embeddings"""
    cursor = conn.execute("""
        SELECT id, title, pub_date, embedding 
        FROM node 
        WHERE graph_id = ? AND embedding IS NOT NULL
        ORDER BY pub_date DESC
    """, (graph_id,))
    
    results = []
    for row in cursor.fetchall():
        if row[3]:
            results.append({
                "id": row[0],
                "title": row[1],
                "pub_date": row[2],
                "embedding": pickle.loads(row[3])
            })
    return results


def get_node_count(conn: sqlite3.Connection, in_graph: bool = True, graph_id: int = 1) -> int:
    """获取节点数量"""
    cursor = conn.execute("""
        SELECT COUNT(*) FROM node WHERE graph_id = ? AND in_graph = ?
    """, (graph_id, in_graph))
    return cursor.fetchone()[0]


def get_edge_count(conn: sqlite3.Connection, graph_id: int = 1) -> int:
    """获取边数量"""
    cursor = conn.execute("""
        SELECT COUNT(*) FROM edge WHERE graph_id = ?
    """, (graph_id,))
    return cursor.fetchone()[0]


def set_node_in_graph(conn: sqlite3.Connection, node_id: int, in_graph: bool = True):
    """标记节点已加入图"""
    conn.execute("""
        UPDATE node SET in_graph = ? WHERE id = ?
    """, (in_graph, node_id))


def save_meta(conn: sqlite3.Connection, name: str, window_days: int, threshold: float, max_edges: int):
    """保存图元数据"""
    conn.execute("""
        INSERT OR REPLACE INTO graph_meta (id, name, window_days, threshold, max_edges_per_node, last_updated)
        VALUES (1, ?, ?, ?, ?, ?)
    """, (name, window_days, threshold, max_edges, datetime.now().isoformat()))


def get_meta(conn: sqlite3.Connection) -> Optional[dict]:
    """获取图元数据"""
    cursor = conn.execute("SELECT * FROM graph_meta WHERE id = 1")
    row = cursor.fetchone()
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "window_days": row[2],
            "threshold": row[3],
            "max_edges_per_node": row[4],
            "last_updated": row[5],
            "created_at": row[6]
        }
    return None


def export_graph_json(conn: sqlite3.Connection, output_path: str):
    """导出图为 JSON (供前端使用)"""
    nodes = []
    cursor = conn.execute("""
        SELECT id, title, link, pub_date, feed_id 
        FROM node WHERE in_graph = 1 ORDER BY id
    """)
    for row in cursor:
        nodes.append({
            "id": row[0],
            "title": row[1],
            "link": row[2],
            "pub_date": row[3],
            "feed_id": row[4]
        })

    edges = []
    cursor = conn.execute("SELECT source_id, target_id, weight FROM edge ORDER BY source_id")
    for row in cursor:
        w = row[2]
        # 兼容旧数据：np.float32 曾被错误存为 BLOB
        if isinstance(w, bytes):
            import struct
            w = struct.unpack("f", w)[0]
        edges.append({
            "source": row[0],
            "target": row[1],
            "weight": round(float(w), 4)
        })

    meta = get_meta(conn)
    
    data = {
        "meta": meta,
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "nodes": nodes,
        "edges": edges,
        "exportedAt": datetime.now().isoformat()
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return len(nodes), len(edges)