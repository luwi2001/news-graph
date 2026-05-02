import sqlite3
import pickle
import numpy as np
import os
from datetime import datetime

# Load RSS DB
rss_path = os.path.expanduser("~/services/rsstt/config/db.sqlite3")
rss_conn = sqlite3.connect(rss_path)
entries = []
cursor = rss_conn.execute("""
    SELECT id, feed_id, title, link, pub_date, description
    FROM entries WHERE pub_date IS NOT NULL 
    ORDER BY pub_date ASC LIMIT 300
""")
for row in cursor:
    entries.append({"id": row[0], "feed_id": row[1], "title": row[2], "link": row[3], "pub_date": row[4], "description": row[5]})

print(f"Loaded {len(entries)} entries")

# Create graph DB
graph_db = "data/news_graph.db"
os.makedirs("data", exist_ok=True)
gconn = sqlite3.connect(graph_db)

gconn.execute("DROP TABLE IF EXISTS node")
gconn.execute("DROP TABLE IF EXISTS edge")

gconn.execute("""
    CREATE TABLE node (
        id INTEGER PRIMARY KEY, graph_id INTEGER DEFAULT 1,
        title TEXT, link TEXT, pub_date TEXT, description TEXT,
        feed_id INTEGER, embedding BLOB, in_graph INTEGER DEFAULT 0
    )
""")
gconn.execute("""
    CREATE TABLE edge (
        id INTEGER PRIMARY KEY AUTOINCREMENT, graph_id INTEGER DEFAULT 1,
        source_id INTEGER, target_id INTEGER, weight REAL, 
        UNIQUE(source_id, target_id)
    )
""")

for e in entries:
    gconn.execute("""
        INSERT INTO node (id, graph_id, title, link, pub_date, description, feed_id, in_graph)
        VALUES (?, 1, ?, ?, ?, ?, ?, 0)
    """, (e["id"], e["title"], e["link"], e["pub_date"], e["description"], e["feed_id"]))

gconn.commit()

# Generate embeddings
from sentence_transformers import SentenceTransformer
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [e["title"][:500] for e in entries]
print(f"Encoding...")
embeddings = model.encode(texts)

# Save embeddings
for i, e in enumerate(entries):
    gconn.execute("UPDATE node SET embedding = ? WHERE id = ?", (pickle.dumps(embeddings[i]), e["id"]))

gconn.commit()
print("Embeddings saved")

# Build edges - simplified: just find similar entries within 3 days and add edges
print("Building edges...")
window_days = 3
threshold = 0.5
max_edges = 3

def get_date_days(s):
    # Extract date and return as days from epoch
    # Handles both "2026-02-05 21:17:26" and "Fri, 01 May 2026 00:09:46 -0700"
    try:
        # Try RFC 2822 first
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(s)
        return dt.toordinal()
    except Exception:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
            return dt.toordinal()
        except Exception:
            return 0

edge_count = 0
for i in range(len(embeddings)):
    candidates = []
    
    for j in range(len(embeddings)):
        if i == j:
            continue
            
        date_i = get_date_days(entries[i]["pub_date"])
        date_j = get_date_days(entries[j]["pub_date"])
        
        # Check days difference - allow 1-3 days difference
        diff = abs(date_i - date_j)
        if diff == 0 or diff > window_days:
            continue
            
        # Compute cosine similarity
        norm_i = np.linalg.norm(embeddings[i])
        norm_j = np.linalg.norm(embeddings[j])
        if norm_i > 0 and norm_j > 0:
            sim = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
            if sim >= threshold:
                candidates.append((j, sim))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    for j, sim in candidates[:max_edges]:
        try:
            gconn.execute("INSERT INTO edge VALUES (?, 1, ?, ?, ?)", (None, entries[i]["id"], entries[j]["id"], sim))
            edge_count += 1
        except:
            pass

gconn.commit()
gconn.execute("UPDATE node SET in_graph = 1")
gconn.commit()

print(f"Done: {len(entries)} nodes, {edge_count} edges")
print(f"In DB: {gconn.execute('SELECT COUNT(*) FROM edge').fetchone()[0]} edges")