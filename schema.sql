-- News Graph 数据库

-- 图元数据
CREATE TABLE IF NOT EXISTS graph_meta (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    window_days INTEGER DEFAULT 3,
    threshold REAL DEFAULT 0.6,
    max_edges_per_node INTEGER DEFAULT 3,
    last_updated TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 节点表 (存储新闻)
CREATE TABLE IF NOT EXISTS node (
    id INTEGER PRIMARY KEY,  -- 对应 entry id
    graph_id INTEGER DEFAULT 1,
    title TEXT NOT NULL,
    link TEXT,
    pub_date TEXT,
    description TEXT,
    feed_id INTEGER,
    embedding BLOB,  -- 存储 embedding 向量 (pickle)
    in_graph BOOLEAN DEFAULT 0,  -- 是否已加入图
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_node_graph ON node(graph_id);
CREATE INDEX idx_node_pubdate ON node(pub_date);
CREATE INDEX idx_node_ingraph ON node(in_graph);

-- 边表
CREATE TABLE IF NOT EXISTS edge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_id INTEGER DEFAULT 1,
    feed_id INTEGER,             -- 所属 RSS 源
    source_id INTEGER NOT NULL,  -- 对应 node id
    target_id INTEGER NOT NULL,  -- 对应 node id
    weight REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, target_id, feed_id)
);

CREATE INDEX edge_source ON edge(source_id);
CREATE INDEX edge_target ON edge(target_id);
CREATE INDEX edge_feed ON edge(feed_id);

-- 记录增量更新状态
CREATE TABLE IF NOT EXISTS update_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id INTEGER NOT NULL,  -- 对应的 RSS entry id
    node_id INTEGER,  -- 转换后的 graph node id
    status TEXT DEFAULT 'pending',  -- pending, processed, failed
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);