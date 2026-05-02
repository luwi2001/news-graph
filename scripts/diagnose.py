#!/usr/bin/env python3
"""
诊断脚本：分析 embedding 相似度分布和时间窗口逻辑
"""
import sqlite3
import pickle
import numpy as np
from datetime import datetime
import random

db_path = "data/news_graph.db"

conn = sqlite3.connect(db_path)
cursor = conn.execute("SELECT id, title, pub_date, embedding FROM node WHERE embedding IS NOT NULL ORDER BY pub_date ASC")
rows = cursor.fetchall()

print(f"总共 {len(rows)} 个节点\n")

# 检查时间格式
print("=== 时间格式检查 ===")
sample_dates = [r[2] for r in rows[:5]]
for d in sample_dates:
    print(f"  {d!r}")

# 解析时间和 embedding
entries = []
for row in rows:
    eid, title, pub_date, emb_blob = row
    embedding = pickle.loads(emb_blob)
    # 尝试解析日期
    dt = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(pub_date, fmt)
            break
        except:
            pass
    if dt is None:
        print(f"  警告：无法解析日期 {pub_date!r} (id={eid})")
    entries.append({"id": eid, "title": title, "date": dt, "emb": embedding})

# 检查时间范围
valid_entries = [e for e in entries if e["date"]]
if valid_entries:
    dates = [e["date"] for e in valid_entries]
    print(f"\n时间范围: {min(dates)} ~ {max(dates)}")
    print(f"有效节点: {len(valid_entries)}")

# 随机抽样计算相似度分布
print("\n=== 相似度分布抽样 (随机 5000 对) ===")
embeddings = np.array([e["emb"] for e in valid_entries])
dates = [e["date"] for e in valid_entries]
n = len(embeddings)

sims = []
window_sims = []  # 1-3 天时间窗口内的相似度
same_day_sims = []

random.seed(42)
for _ in range(5000):
    i, j = random.sample(range(n), 2)
    a, b = embeddings[i], embeddings[j]
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
    sims.append(sim)
    
    # 计算天数差
    diff = abs((dates[i] - dates[j]).days)
    if 1 <= diff <= 3:
        window_sims.append(sim)
    if diff == 0:
        same_day_sims.append(sim)

sims = np.array(sims)
window_sims = np.array(window_sims) if window_sims else np.array([])
same_day_sims = np.array(same_day_sims) if same_day_sims else np.array([])

print(f"  总体相似度:  min={sims.min():.4f}, max={sims.max():.4f}, mean={sims.mean():.4f}, median={np.median(sims):.4f}")
print(f"  >=0.5 占比: {(sims >= 0.5).sum()} / {len(sims)} = {(sims >= 0.5).mean()*100:.1f}%")
print(f"  >=0.7 占比: {(sims >= 0.7).sum()} / {len(sims)} = {(sims >= 0.7).mean()*100:.1f}%")

if len(window_sims) > 0:
    print(f"\n  时间窗口(1-3天)内相似度:  min={window_sims.min():.4f}, max={window_sims.max():.4f}, mean={window_sims.mean():.4f}, median={np.median(window_sims):.4f}")
    print(f"  >=0.5 占比: {(window_sims >= 0.5).sum()} / {len(window_sims)} = {(window_sims >= 0.5).mean()*100:.1f}%")
    print(f"  >=0.7 占比: {(window_sims >= 0.7).sum()} / {len(window_sims)} = {(window_sims >= 0.7).mean()*100:.1f}%")
else:
    print(f"\n  时间窗口(1-3天)内无样本！")

if len(same_day_sims) > 0:
    print(f"\n  同一天相似度:  min={same_day_sims.min():.4f}, max={same_day_sims.max():.4f}, mean={same_day_sims.mean():.4f}, median={np.median(same_day_sims):.4f}")

# 检查时间差分布
print("\n=== 时间差分布 ===")
diffs = []
for _ in range(5000):
    i, j = random.sample(range(n), 2)
    diff = abs((dates[i] - dates[j]).days)
    diffs.append(diff)

diffs = np.array(diffs)
print(f"  平均时间差: {diffs.mean():.1f} 天")
print(f"  同一天: {(diffs == 0).sum()} / {len(diffs)} = {(diffs == 0).mean()*100:.1f}%")
print(f"  1-3天:  {((diffs >= 1) & (diffs <= 3)).sum()} / {len(diffs)} = {((diffs >= 1) & (diffs <= 3)).mean()*100:.1f}%")
print(f"  >3天:   {(diffs > 3).sum()} / {len(diffs)} = {(diffs > 3).mean()*100:.1f}%")

# 检查 rebuild_db.py 的边构建逻辑
print("\n=== 模拟 rebuild_db.py 边构建逻辑 ===")
window_days = 3
threshold = 0.5
edge_count = 0
for i in range(min(100, n)):  # 只模拟前100个避免太慢
    for j in range(n):
        if i == j:
            continue
        date_i = dates[i].toordinal()
        date_j = dates[j].toordinal()
        diff = abs(date_i - date_j)
        if diff == 0 or diff > window_days:
            continue
        a, b = embeddings[i], embeddings[j]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        if sim >= threshold:
            edge_count += 1

print(f"  前 100 个节点模拟: 产生 {edge_count} 条边")
if edge_count == 0:
    print("  ⚠️  即使在前 100 个节点中也没有边！")

# 检查 builder.py 的边构建逻辑
print("\n=== 模拟 builder.py build_edges 逻辑 ===")
# build_edges 要求按时间升序排列（for j in range(i+1, n) 只往后看）
# 检查 entries 是否已经按时间升序
is_sorted = all(dates[i] <= dates[i+1] for i in range(min(n-1, 100)))
print(f"  前 100 条数据按时间升序: {is_sorted}")

from src.graph.builder import build_edges, GraphConfig
config = GraphConfig(window_days=3, similarity_threshold=0.5, max_edges_per_node=3)
edges = build_edges(embeddings[:100], dates[:100], config)
print(f"  前 100 个节点 build_edges: 产生 {len(edges)} 条边")

# 尝试不同阈值
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    config = GraphConfig(window_days=3, similarity_threshold=thresh, max_edges_per_node=3)
    edges = build_edges(embeddings, dates, config)
    print(f"  全部节点 threshold={thresh}: {len(edges)} 条边")

conn.close()
print("\n诊断完成")
