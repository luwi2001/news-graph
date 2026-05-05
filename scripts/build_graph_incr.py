#!/usr/bin/env python3
"""
增量构建新闻图谱到数据库
"""
import argparse
import os
import sys
import pickle
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.db import load_entries, get_connection as get_rss_connection
from src.embedding.encoder import EmbeddingEncoder
from src.graph.builder import NewsGraphBuilder, GraphConfig, IncrementalGraphBuilder
from src.graph.builder import compute_cosine_similarity, compute_days_diff, is_same_day


def init_graph_db(db_path):
    """初始化图数据库"""
    import sqlite3
    os.makedirs(Path(db_path).parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_meta (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            window_days INTEGER DEFAULT 3,
            threshold REAL DEFAULT 0.6,
            max_edges_per_node INTEGER DEFAULT 3,
            last_updated TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node (
            id INTEGER PRIMARY KEY,
            graph_id INTEGER DEFAULT 1,
            title TEXT NOT NULL,
            link TEXT,
            pub_date TEXT,
            description TEXT,
            feed_id INTEGER,
            embedding BLOB,
            in_graph BOOLEAN DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            graph_id INTEGER DEFAULT 1,
            feed_id INTEGER,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            weight REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_id, target_id, feed_id)
        )
    """)
    
    conn.commit()
    return conn


def main():
    parser = argparse.ArgumentParser(description="增量构建新闻图谱到数据库")
    parser.add_argument("--rebuild", action="store_true", help="全量重建（删边、重置 in_graph）")
    parser.add_argument("--skip-embeddings", action="store_true", help="跳过 embedding 生成，从数据库读取已有 embedding（秒级重建边）")
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--max-edges", type=int, default=5)
    args = parser.parse_args()
    
    rss_db = os.path.expanduser("~/services/rsstt/config/db.sqlite3")
    graph_db = "data/news_graph.db"
    
    print(f"[*] RSS 数据库: {rss_db}")
    rss_conn = get_rss_connection(rss_db)
    
    print(f"[*] 图数据库: {graph_db}")
    db_conn = init_graph_db(graph_db)
    
    # 给已有 edge 表添加 feed_id 列（兼容旧数据库）
    try:
        db_conn.execute("ALTER TABLE edge ADD COLUMN feed_id INTEGER")
        db_conn.commit()
        print("    [+] 已升级 edge 表结构（添加 feed_id）")
    except:
        pass
    
    if args.rebuild:
        print("    [+] 全量重建模式")
        db_conn.execute("DELETE FROM edge")
        db_conn.execute("UPDATE node SET in_graph = 0")
        db_conn.commit()
    
    print("[*] 加载 RSS 数据...")
    entries = load_entries(rss_conn)
    entries = [e for e in entries if e.pub_date is not None]
    print(f"    共 {len(entries)} 条有效新闻")
    
    print("[*] 同步节点...")
    existing_ids = set(row[0] for row in db_conn.execute("SELECT id FROM node"))
    
    new_entries = []
    for e in entries:
        if e.id not in existing_ids:
            new_entries.append(e)
            db_conn.execute("""
                INSERT INTO node (id, graph_id, title, link, pub_date, description, feed_id, in_graph)
                VALUES (?, 1, ?, ?, ?, ?, ?, 0)
            """, (e.id, e.title, e.link, e.pub_date.isoformat(), e.description, e.feed_id))
    
    db_conn.commit()
    print(f"    新增 {len(new_entries)} 个节点")
    
    all_entries = entries
    
    if not new_entries and not args.rebuild:
        print("[*] 没有新节点，退出")
        return
    
    config = GraphConfig(
        window_days=args.window_days,
        similarity_threshold=args.threshold,
        max_edges_per_node=args.max_edges
    )
    
    # 统一增量逻辑：从数据库读取已有 embedding，仅对缺失的生成
    print("[*] 加载已有 Embedding...")
    rows = db_conn.execute("""
        SELECT id, embedding FROM node WHERE embedding IS NOT NULL
    """).fetchall()
    
    emb_map = {}
    for row in rows:
        nid, emb_blob = row
        if emb_blob:
            emb_map[nid] = pickle.loads(emb_blob)
    
    # 找出所有缺少 embedding 的节点
    missing_entries = [e for e in all_entries if e.id not in emb_map]
    
    if missing_entries:
        if args.skip_embeddings:
            print(f"[!] {len(missing_entries)} 个节点缺少 embedding，但 --skip-embeddings 已指定，无法生成")
        else:
            print(f"[*] {len(missing_entries)} 个节点需要生成 embedding...")
            encoder = EmbeddingEncoder(cache_dir="data/embeddings")
            texts = [encoder.get_text_for_entry(e.title, e.description) for e in missing_entries]
            new_embeddings = encoder.encode(texts)
            for i, e in enumerate(missing_entries):
                db_conn.execute("UPDATE node SET embedding = ? WHERE id = ?",
                               (pickle.dumps(new_embeddings[i]), e.id))
            db_conn.commit()
            for i, e in enumerate(missing_entries):
                emb_map[e.id] = new_embeddings[i]
            print(f"    生成了 {len(new_embeddings)} 个新向量")
    else:
        print("    所有节点已有 embedding")
    
    # 筛选有 embedding 的条目
    entries_with_emb = []
    embeddings_list = []
    for e in all_entries:
        if e.id in emb_map:
            entries_with_emb.append(e)
            embeddings_list.append(emb_map[e.id])
    
    missing = len(all_entries) - len(entries_with_emb)
    if missing > 0:
        print(f"    警告: {missing} 个节点缺少 embedding，将被跳过")
    
    all_entries = entries_with_emb
    embeddings = np.array(embeddings_list)
    print(f"    共 {len(embeddings)} 个向量可用")
    
    if len(all_entries) == 0:
        print("[!] 没有可用节点，退出")
        return
    
    # 建立 id -> embedding 映射
    id_to_emb = {e.id: embeddings[i] for i, e in enumerate(all_entries)}
    
    print("[*] 按 RSS 源分组构建边...")
    from collections import defaultdict
    from src.graph.builder import build_edges as build_edge_func
    
    # 按 feed_id 分组
    feed_groups = defaultdict(list)
    for e in all_entries:
        feed_groups[e.feed_id].append(e)
    
    total_edges = 0
    
    for feed_id, feed_entries in sorted(feed_groups.items()):
        if len(feed_entries) < 2:
            continue
        
        # 按时间升序排列
        sorted_feed = sorted(feed_entries, key=lambda e: e.pub_date)
        pub_dates = [e.pub_date for e in sorted_feed]
        entry_ids = [e.id for e in sorted_feed]
        embs = np.array([id_to_emb[eid] for eid in entry_ids])
        
        edges = build_edge_func(embs, pub_dates, config)
        total_edges += len(edges)
        
        for src_idx, tgt_idx, sim in edges:
            db_conn.execute("""
                INSERT OR IGNORE INTO edge (source_id, target_id, weight, graph_id, feed_id)
                VALUES (?, ?, ?, 1, ?)
            """, (entry_ids[src_idx], entry_ids[tgt_idx], sim, feed_id))
    
    print(f"    共 {total_edges} 条边（{len(feed_groups)} 个 feed）")
    
    for entry_id in [e.id for e in all_entries]:
        db_conn.execute("UPDATE node SET in_graph = 1 WHERE id = ?", (entry_id,))
    
    db_conn.commit()
    
    node_count = db_conn.execute("SELECT COUNT(*) FROM node WHERE in_graph = 1").fetchone()[0]
    edge_count = db_conn.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
    
    print(f"[*] 统计: {node_count} 节点, {edge_count} 边")
    
    print("[*] 导出 JSON...")
    nodes = []
    cursor = db_conn.execute("SELECT id, title, link, pub_date, feed_id FROM node WHERE in_graph = 1 ORDER BY id")
    for row in cursor:
        nodes.append({"id": row[0], "title": row[1], "link": row[2], "pub_date": row[3], "feed_id": row[4]})
    
    edges = []
    cursor = db_conn.execute("SELECT source_id, target_id, weight, feed_id FROM edge ORDER BY source_id")
    for row in cursor:
        w = row[2]
        # 兼容旧数据：np.float32 曾被错误存为 BLOB
        if isinstance(w, bytes):
            import struct
            w = struct.unpack("f", w)[0]
        edges.append({"source": row[0], "target": row[1], "weight": round(float(w), 4), "feed_id": row[3]})
    
    data = {
        "window_days": args.window_days,
        "threshold": args.threshold,
        "max_edges": args.max_edges,
        "nodeCount": node_count,
        "edgeCount": edge_count,
        "nodes": nodes,
        "edges": edges,
        "exportedAt": datetime.now().isoformat()
    }
    
    with open("data/output/graph_api.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("[+] 完成!")
    
    db_conn.close()
    rss_conn.close()


if __name__ == "__main__":
    main()