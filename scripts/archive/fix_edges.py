#!/usr/bin/env python3
"""
修复边构建：利用已有 embeddings 重新构建边，不再重新生成 embeddings
"""
import sqlite3
import pickle
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.builder import build_edges, GraphConfig
from src.utils.db import parse_datetime

DB_PATH = "data/news_graph.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    
    # 读取已有节点和 embeddings
    rows = conn.execute("""
        SELECT id, title, pub_date, embedding 
        FROM node 
        WHERE embedding IS NOT NULL
        ORDER BY pub_date ASC
    """).fetchall()
    
    print(f"[*] 读取 {len(rows)} 个已有节点")
    
    entries = []
    for row in rows:
        eid, title, pub_date, emb_blob = row
        dt = parse_datetime(pub_date)
        emb = pickle.loads(emb_blob)
        entries.append({"id": eid, "date": dt, "emb": emb})
    
    valid = [e for e in entries if e["date"]]
    print(f"[*] 有效节点: {len(valid)}")
    
    # 按时间升序排列
    valid.sort(key=lambda x: x["date"])
    
    embeddings = np.array([e["emb"] for e in valid])
    pub_dates = [e["date"] for e in valid]
    entry_ids = [e["id"] for e in valid]
    
    # 构建边
    config = GraphConfig(window_days=3, similarity_threshold=0.4, max_edges_per_node=3)
    edges = build_edges(embeddings, pub_dates, config)
    
    print(f"[*] 找到 {len(edges)} 条候选边")
    
    # 清空旧边并插入新边
    conn.execute("DELETE FROM edge")
    conn.execute("UPDATE node SET in_graph = 0")
    conn.commit()
    
    for src_idx, tgt_idx, sim in edges:
        conn.execute("""
            INSERT OR IGNORE INTO edge (source_id, target_id, weight, graph_id)
            VALUES (?, ?, ?, 1)
        """, (entry_ids[src_idx], entry_ids[tgt_idx], sim))
    
    for eid in entry_ids:
        conn.execute("UPDATE node SET in_graph = 1 WHERE id = ?", (eid,))
    
    conn.commit()
    
    node_count = conn.execute("SELECT COUNT(*) FROM node WHERE in_graph = 1").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
    
    print(f"[*] 完成: {node_count} 节点, {edge_count} 边")
    
    # 验证 weight 类型
    c = conn.execute("SELECT typeof(weight) FROM edge LIMIT 1").fetchone()
    print(f"[*] weight 类型: {c[0]}")
    
    conn.close()

if __name__ == "__main__":
    main()
