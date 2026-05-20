#!/usr/bin/env python3
"""
增量构建新闻图谱到数据库（知识库版）
支持 chunk 切分、Faiss 向量索引、实体关系抽取
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
from src.graph.builder import GraphConfig
from src.chunking.chunker import TextChunker
from src.vectorstore.faiss_store import FaissChunkStore
from src.knowledge.entity_extractor import EntityExtractor


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

    # --- 知识库新增表 ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            faiss_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(doc_id, chunk_index)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            mention_count INTEGER DEFAULT 1,
            UNIQUE(name, type)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_mention (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
            chunk_id INTEGER,
            start_pos INTEGER,
            end_pos INTEGER,
            text_span TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_relation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relation_type TEXT DEFAULT 'cooccurrence',
            doc_id INTEGER NOT NULL,
            chunk_id INTEGER,
            confidence REAL DEFAULT 1.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_entity_id, target_entity_id, doc_id, chunk_id)
        )
    """)
    # --- 知识库新增表结束 ---

    conn.commit()
    return conn


def process_entities(entry, db_conn, extractor):
    """抽取并存储实体和共现关系（如果该文档尚未处理）"""
    existing = db_conn.execute(
        "SELECT COUNT(*) FROM entity_mention WHERE doc_id = ?",
        (entry.id,)
    ).fetchone()[0]
    if existing > 0:
        return  # 已有实体，跳过

    # HTML 清洗（P0）
    title = entry.title.strip() if entry.title else ""
    description = entry.description or ""
    if description:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(description, 'html.parser')
            for tag in soup(['img', 'figure', 'figcaption', 'script', 'style', 'iframe']):
                tag.decompose()
            description = soup.get_text(separator=' ', strip=True)
        except ImportError:
            import re
            description = re.sub(r'<[^>]+>', ' ', description)
            description = re.sub(r'\s+', ' ', description).strip()

    full_text = title
    if description:
        full_text = f"{full_text}: {description}"

    entities, relations = extractor.extract(full_text, entry.id)

    # 存储实体和提及
    for ent in entities:
        db_conn.execute("""
            INSERT OR IGNORE INTO entity (name, type) VALUES (?, ?)
        """, (ent["name"], ent["type"]))

        cursor = db_conn.execute("""
            SELECT id FROM entity WHERE name = ? AND type = ?
        """, (ent["name"], ent["type"]))
        entity_id = cursor.fetchone()[0]

        db_conn.execute("""
            INSERT INTO entity_mention (entity_id, doc_id, start_pos, end_pos, text_span)
            VALUES (?, ?, ?, ?, ?)
        """, (entity_id, entry.id, ent["start"], ent["end"], ent["text_span"]))

    # 存储关系
    for rel in relations:
        src = db_conn.execute(
            "SELECT id FROM entity WHERE name = ? AND type = ?",
            (rel["source"], rel["source_type"])
        ).fetchone()
        tgt = db_conn.execute(
            "SELECT id FROM entity WHERE name = ? AND type = ?",
            (rel["target"], rel["target_type"])
        ).fetchone()
        if src and tgt:
            db_conn.execute("""
                INSERT OR IGNORE INTO entity_relation
                (source_entity_id, target_entity_id, relation_type, doc_id, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (src[0], tgt[0], rel["type"], entry.id, rel["confidence"]))

    db_conn.commit()


def main():
    parser = argparse.ArgumentParser(description="增量构建新闻图谱到数据库（知识库版）")
    parser.add_argument("--rebuild", action="store_true", help="全量重建（删边、重置 in_graph、清空 chunks/entities/Faiss）")
    parser.add_argument("--skip-embeddings", action="store_true", help="跳过 embedding 生成，仅复用已有 chunks/Faiss（秒级重建边）")
    parser.add_argument("--skip-entities", action="store_true", help="跳过实体关系抽取")
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
    
    # 兼容旧数据库：给 edge 表添加 feed_id
    try:
        db_conn.execute("ALTER TABLE edge ADD COLUMN feed_id INTEGER")
        db_conn.commit()
        print("    [+] 已升级 edge 表结构（添加 feed_id）")
    except:
        pass
    
    # 初始化知识库组件
    chunker = TextChunker(chunk_size=512, overlap=64)
    faiss_store = FaissChunkStore(index_path="data/faiss/chunk_index", dim=384)
    extractor = EntityExtractor()
    
    if args.rebuild:
        print("    [+] 全量重建模式")
        db_conn.execute("DELETE FROM edge")
        db_conn.execute("UPDATE node SET in_graph = 0")
        db_conn.execute("DELETE FROM chunk")
        db_conn.execute("DELETE FROM entity_mention")
        db_conn.execute("DELETE FROM entity_relation")
        db_conn.execute("DELETE FROM entity")
        db_conn.commit()
        faiss_store.clear()
    
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
    
    # --- 阶段 1：区分已有 chunks 和需要新 chunks 的文档 ---
    print("[*] 分析文档 chunks 状态...")
    entries_with_chunks = []   # 已有 chunks，可从 Faiss 复用
    entries_need_chunks = []   # 需要生成新 chunks
    
    for e in all_entries:
        rows = db_conn.execute(
            "SELECT text, chunk_index, faiss_id FROM chunk WHERE doc_id = ? ORDER BY chunk_index",
            (e.id,)
        ).fetchall()
        if rows:
            entries_with_chunks.append((e, rows))
        else:
            entries_need_chunks.append(e)
    
    print(f"    {len(entries_with_chunks)} 篇已有 chunks，{len(entries_need_chunks)} 篇需要生成")
    
    if args.skip_embeddings and entries_need_chunks:
        print(f"[!] {len(entries_need_chunks)} 篇缺少 chunks，但 --skip-embeddings 已指定，将跳过")
    
    # --- 阶段 2：批量生成新 chunks 的 embeddings ---
    doc_embeddings_map = {}
    
    if entries_need_chunks and not args.skip_embeddings:
        print("[*] 生成新 chunks...")
        # 先为所有需要处理的文档生成 chunks 文本
        all_new_chunks_texts = []
        doc_chunk_counts = []  # 记录每篇文档有多少个 chunks
        
        for e in entries_need_chunks:
            chunks = chunker.chunk(e.title, e.description)
            doc_chunk_counts.append(len(chunks))
            all_new_chunks_texts.extend(chunks)
        
        print(f"    共 {len(all_new_chunks_texts)} 个新 chunks，开始批量编码...")
        encoder = EmbeddingEncoder(cache_dir="data/embeddings")
        all_new_embeddings = encoder.encode_chunks(all_new_chunks_texts, show_progress=True)
        print(f"    编码完成")
        
        # 按文档切分 embeddings，写入 Faiss 和 chunk 表
        offset = 0
        for e, count in zip(entries_need_chunks, doc_chunk_counts):
            chunk_embeddings = all_new_embeddings[offset:offset + count]
            chunks_texts = all_new_chunks_texts[offset:offset + count]
            offset += count
            
            # 添加到 Faiss
            metadata = [{"doc_id": e.id, "text": t} for t in chunks_texts]
            faiss_ids = faiss_store.add_chunks(chunk_embeddings, metadata)
            
            # 写入 chunk 表
            for idx, (chunk_text, fid) in enumerate(zip(chunks_texts, faiss_ids)):
                db_conn.execute("""
                    INSERT OR REPLACE INTO chunk (doc_id, text, chunk_index, faiss_id)
                    VALUES (?, ?, ?, ?)
                """, (e.id, chunk_text, idx, fid))
            
            # 计算文档代表向量
            doc_emb = EmbeddingEncoder.get_doc_representative(chunk_embeddings)
            doc_embeddings_map[e.id] = doc_emb
        
        db_conn.commit()
        print(f"    已写入 {len(entries_need_chunks)} 篇文档的 chunks")
    
    # --- 阶段 3：从 Faiss 复用已有 chunks 的文档向量 ---
    if entries_with_chunks:
        print(f"[*] 从 Faiss 复用 {len(entries_with_chunks)} 篇已有 chunks...")
        for e, rows in entries_with_chunks:
            chunk_embeddings = []
            for text, idx, fid in rows:
                if fid is not None and 0 <= fid < faiss_store.index.ntotal:
                    vec = faiss_store.index.reconstruct(int(fid))
                    chunk_embeddings.append(vec)
                else:
                    # Faiss ID 无效，重新编码这个 chunk
                    if args.skip_embeddings:
                        break
                    encoder = encoder if 'encoder' in dir() else EmbeddingEncoder(cache_dir="data/embeddings")
                    vec = encoder.encode_chunks([text])[0]
                    chunk_embeddings.append(vec)
            
            if len(chunk_embeddings) > 0:
                doc_emb = EmbeddingEncoder.get_doc_representative(np.array(chunk_embeddings))
                doc_embeddings_map[e.id] = doc_emb
    
    skipped = len(all_entries) - len(doc_embeddings_map)
    if skipped > 0:
        print(f"    警告: {skipped} 个节点缺少 embedding，将被跳过")
    
    print(f"    共 {len(doc_embeddings_map)} 个文档向量可用")
    
    # --- 阶段 4：实体抽取 ---
    if not args.skip_entities:
        print("[*] 抽取实体关系...")
        for i, e in enumerate(all_entries):
            if e.id in doc_embeddings_map:
                process_entities(e, db_conn, extractor)
            if (i + 1) % 200 == 0:
                print(f"    已处理 {i + 1}/{len(all_entries)} 篇")
        
        entity_count = db_conn.execute("SELECT COUNT(*) FROM entity").fetchone()[0]
        relation_count = db_conn.execute("SELECT COUNT(*) FROM entity_relation").fetchone()[0]
        print(f"    共 {entity_count} 个实体, {relation_count} 条关系")
    
    # --- 阶段 5：按 feed 分组构建边 ---
    entries_with_emb = [e for e in all_entries if e.id in doc_embeddings_map]
    if len(entries_with_emb) == 0:
        print("[!] 没有可用节点，退出")
        return
    
    print("[*] 按 RSS 源分组构建边...")
    from collections import defaultdict
    from src.graph.builder import build_edges as build_edge_func
    
    feed_groups = defaultdict(list)
    for e in entries_with_emb:
        feed_groups[e.feed_id].append(e)
    
    total_edges = 0
    
    for feed_id, feed_entries in sorted(feed_groups.items()):
        if len(feed_entries) < 2:
            continue
        
        sorted_feed = sorted(feed_entries, key=lambda e: e.pub_date)
        pub_dates = [e.pub_date for e in sorted_feed]
        entry_ids = [e.id for e in sorted_feed]
        embs = np.array([doc_embeddings_map[eid] for eid in entry_ids])
        
        edges = build_edge_func(embs, pub_dates, config)
        total_edges += len(edges)
        
        for src_idx, tgt_idx, sim in edges:
            db_conn.execute("""
                INSERT OR IGNORE INTO edge (source_id, target_id, weight, graph_id, feed_id)
                VALUES (?, ?, ?, 1, ?)
            """, (entry_ids[src_idx], entry_ids[tgt_idx], sim, feed_id))
    
    print(f"    共 {total_edges} 条边（{len(feed_groups)} 个 feed）")
    
    # 更新向后兼容的 node.embedding BLOB
    for e in entries_with_emb:
        db_conn.execute("""
            UPDATE node SET embedding = ? WHERE id = ?
        """, (pickle.dumps(doc_embeddings_map[e.id]), e.id))
        db_conn.execute("UPDATE node SET in_graph = 1 WHERE id = ?", (e.id,))
    
    db_conn.commit()
    
    node_count = db_conn.execute("SELECT COUNT(*) FROM node WHERE in_graph = 1").fetchone()[0]
    edge_count = db_conn.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
    chunk_count = db_conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    
    print(f"[*] 统计: {node_count} 节点, {edge_count} 边, {chunk_count} chunks")
    
    print("[*] 导出 JSON...")
    nodes = []
    cursor = db_conn.execute("SELECT id, title, link, pub_date, feed_id FROM node WHERE in_graph = 1 ORDER BY id")
    for row in cursor:
        nodes.append({"id": row[0], "title": row[1], "link": row[2], "pub_date": row[3], "feed_id": row[4]})
    
    edges = []
    cursor = db_conn.execute("SELECT source_id, target_id, weight, feed_id FROM edge ORDER BY source_id")
    for row in cursor:
        w = row[2]
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
        "chunkCount": chunk_count,
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
