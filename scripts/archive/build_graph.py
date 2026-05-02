#!/usr/bin/env python3
"""
构建新闻图谱入口脚本
"""
import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.db import load_entries, get_connection
from src.embedding.encoder import EmbeddingEncoder
from src.graph.builder import NewsGraphBuilder, GraphConfig
from src.storage.nx_storage import save_graph, export_to_json, get_graph_info


def main():
    parser = argparse.ArgumentParser(description="构建新闻图谱")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--limit", type=int, default=None, help="限制新闻数量 (测试用)")
    parser.add_argument("--incremental", action="store_true", help="增量模式 (仅新增新闻)")
    parser.add_argument("--window-days", type=int, default=3, help="向前天数")
    parser.add_argument("--threshold", type=float, default=0.7, help="相似度阈值")
    args = parser.parse_args()

    # 1. 加载配置
    db_path = os.path.expanduser("~/services/rsstt/config/db.sqlite3")
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] 连接数据库: {db_path}")
    conn = get_connection(db_path)

    # 2. 加载新闻数据
    print(f"[*] 加载新闻数据 (limit={args.limit})...")
    entries = load_entries(conn, limit=args.limit)
    print(f"    加载了 {len(entries)} 条新闻")

    # 过滤掉 pub_date 为 None 的条目
    entries = [e for e in entries if e.pub_date is not None]
    print(f"    有效新闻: {len(entries)} 条")

    if not entries:
        print("[!] 没有有效新闻数据")
        return

    # 3. 生成 Embedding
    print(f"[*] 生成 Embedding...")
    encoder = EmbeddingEncoder(
        model_name="all-MiniLM-L6-v2",
        cache_dir="data/embeddings"
    )

    texts = [
        encoder.get_text_for_entry(e.title, e.description)
        for e in entries
    ]
    embeddings = encoder.encode(texts)
    print(f"    生成了 {len(embeddings)} 个向量")

    # 4. 构建图
    print(f"[*] 构建图 (window={args.window_days}, threshold={args.threshold})...")
    config = GraphConfig(
        window_days=args.window_days,
        similarity_threshold=args.threshold
    )
    builder = NewsGraphBuilder(config)

    pub_dates = [e.pub_date for e in entries]
    entry_ids = [e.id for e in entries]
    titles = [e.title for e in entries]

    G = builder.build_graph(embeddings, pub_dates, entry_ids, titles)

    # 5. 保存图
    print(f"[*] 保存图...")
    info = get_graph_info(G)
    print(f"    节点: {info['nodes']}, 边: {info['edges']}, 密度: {info['density']:.4f}")

    save_graph(G, f"{output_dir}/news_graph.graphml", format="graphml")
    export_to_json(G, f"{output_dir}/news_graph.json")

    print(f"[+] 完成! 图谱已保存到 {output_dir}/")


if __name__ == "__main__":
    main()