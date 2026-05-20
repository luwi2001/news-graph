-- News Graph 数据库（知识库版）

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
    id INTEGER PRIMARY KEY,  -- 对应 RSS entry id
    graph_id INTEGER DEFAULT 1,
    title TEXT,
    link TEXT,
    pub_date TEXT,
    description TEXT,
    feed_id INTEGER,
    embedding BLOB,  -- 文档级 representative 向量 (pickle)
    in_graph INTEGER DEFAULT 0,  -- 是否已加入图 (0/1)
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_node_graph ON node(graph_id);
CREATE INDEX IF NOT EXISTS idx_node_pubdate ON node(pub_date);
CREATE INDEX IF NOT EXISTS idx_node_ingraph ON node(in_graph);

-- 边表 (语义相似度关联)
CREATE TABLE IF NOT EXISTS edge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_id INTEGER DEFAULT 1,
    source_id INTEGER NOT NULL,  -- 对应 node id
    target_id INTEGER NOT NULL,  -- 对应 node id
    weight REAL NOT NULL,
    feed_id INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, target_id)
);

CREATE INDEX IF NOT EXISTS edge_source ON edge(source_id);
CREATE INDEX IF NOT EXISTS edge_target ON edge(target_id);
CREATE INDEX IF NOT EXISTS edge_feed ON edge(feed_id);

-- chunk 表 (文本分块，用于 Faiss 向量检索)
CREATE TABLE IF NOT EXISTS chunk (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,     -- 对应 node id
    text TEXT NOT NULL,          -- chunk 文本
    chunk_index INTEGER NOT NULL, -- 文档内 chunk 序号
    faiss_id INTEGER,            -- Faiss 索引中的 id
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doc_id, chunk_index)
);

-- 实体表 (命名实体库)
CREATE TABLE IF NOT EXISTS entity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,          -- PERSON, ORG, GPE, PRODUCT, EVENT, etc.
    mention_count INTEGER DEFAULT 1,
    UNIQUE(name, type)
);

-- 实体提及表 (实体在文档中的出现位置)
CREATE TABLE IF NOT EXISTS entity_mention (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL,  -- 对应 entity id
    doc_id INTEGER NOT NULL,     -- 对应 node id
    chunk_id INTEGER,            -- 对应 chunk id (可选)
    start_pos INTEGER,
    end_pos INTEGER,
    text_span TEXT
);

-- 实体关系表 (共现关系网络)
CREATE TABLE IF NOT EXISTS entity_relation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id INTEGER NOT NULL,  -- 对应 entity id
    target_entity_id INTEGER NOT NULL,  -- 对应 entity id
    relation_type TEXT DEFAULT 'cooccurrence',
    doc_id INTEGER NOT NULL,            -- 对应 node id
    chunk_id INTEGER,                   -- 对应 chunk id (可选)
    confidence REAL DEFAULT 1.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_entity_id, target_entity_id, doc_id, chunk_id)
);
