#!/usr/bin/env python3
"""
新闻图谱 API 服务（知识库版）
新增：Faiss 向量检索、实体关系查询
"""
import sqlite3
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# 懒加载的知识库组件（首次请求时初始化）
_encoder = None
_faiss_store = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.embedding.encoder import EmbeddingEncoder
        _encoder = EmbeddingEncoder(cache_dir="data/embeddings")
    return _encoder


def _get_faiss_store():
    global _faiss_store
    if _faiss_store is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        sys.path.insert(0, project_root)
        from src.vectorstore.faiss_store import FaissChunkStore
        index_path = os.path.join(project_root, "data", "faiss", "chunk_index")
        _faiss_store = FaissChunkStore(index_path=index_path, dim=384)
    return _faiss_store


DB_PATH = "data/news_graph.db"
RSS_DB_PATH = os.path.expanduser("~/services/rsstt/config/db.sqlite3")
OUTPUT_DIR = "data/output"


def get_db():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(script_dir), DB_PATH)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_rss_db():
    conn = sqlite3.connect(RSS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


class GraphAPIHandler(SimpleHTTPRequestHandler):
    """简单的 REST API 处理器"""
    
    def _get_feed_id(self):
        """从查询参数获取 feed_id"""
        params = parse_qs(urlparse(self.path).query)
        feed_id = params.get("feed_id", [None])[0]
        return int(feed_id) if feed_id else None
    
    def _get_param(self, name, default=None, cast=None):
        """获取查询参数"""
        params = parse_qs(urlparse(self.path).query)
        val = params.get(name, [default])[0]
        if val is None:
            return default
        return cast(val) if cast else val
    
    def do_GET(self):
        path = urlparse(self.path).path
        
        # --- 现有 API（完全不变）---
        if path == "/api/graph":
            self.send_json_response(self.get_graph())
        elif path == "/api/nodes":
            self.send_json_response(self.get_nodes())
        elif path == "/api/edges":
            self.send_json_response(self.get_edges())
        elif path == "/api/stats":
            self.send_json_response(self.get_stats())
        elif path == "/api/feeds":
            self.send_json_response(self.get_feeds())
        elif path == "/api/node":
            params = parse_qs(urlparse(self.path).query)
            node_id = params.get("id", [None])[0]
            if node_id:
                self.send_json_response(self.get_node(int(node_id)))
            else:
                self.send_error(400, "Missing node id")
        # --- 新增知识库 API ---
        elif path == "/api/search":
            self.send_json_response(self.search_chunks())
        elif path == "/api/entities":
            self.send_json_response(self.get_entities())
        elif path == "/api/entity_mentions":
            self.send_json_response(self.get_entity_mentions())
        elif path == "/api/entity_relations":
            self.send_json_response(self.get_entity_relations())
        elif path == "/api/document_chunks":
            self.send_json_response(self.get_document_chunks())
        # --- 静态文件 ---
        elif path == "/" or path == "/index.html" or path == "/visualize.html":
            self.serve_static_file("data/output/visualize.html")
        elif path.startswith("/data/"):
            self.serve_static_file(path)
        else:
            super().do_GET()
    
    # ==================== 现有 API ====================
    
    def get_graph(self):
        feed_id = self._get_feed_id()
        conn = get_db()
        
        if feed_id:
            nodes = []
            cursor = conn.execute("""
                SELECT id, title, link, pub_date, feed_id 
                FROM node WHERE in_graph = 1 AND feed_id = ? ORDER BY pub_date DESC
            """, (feed_id,))
            for row in cursor:
                nodes.append({
                    "id": row[0],
                    "title": row[1],
                    "link": row[2],
                    "pub_date": row[3],
                    "feed_id": row[4]
                })
            
            edges = []
            cursor = conn.execute("""
                SELECT source_id, target_id, weight, feed_id FROM edge WHERE feed_id = ?
            """, (feed_id,))
            for row in cursor:
                edges.append({
                    "source": row[0],
                    "target": row[1],
                    "weight": round(row[2], 4),
                    "feed_id": row[3]
                })
        else:
            nodes = []
            cursor = conn.execute("""
                SELECT id, title, link, pub_date, feed_id 
                FROM node WHERE in_graph = 1 ORDER BY pub_date DESC
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
            cursor = conn.execute("SELECT source_id, target_id, weight, feed_id FROM edge")
            for row in cursor:
                edges.append({
                    "source": row[0],
                    "target": row[1],
                    "weight": round(row[2], 4),
                    "feed_id": row[3]
                })
        
        stats = self.get_stats()
        conn.close()
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": stats
        }
    
    def get_nodes(self):
        feed_id = self._get_feed_id()
        conn = get_db()
        
        if feed_id:
            cursor = conn.execute("""
                SELECT id, title, link, pub_date, feed_id 
                FROM node WHERE in_graph = 1 AND feed_id = ? ORDER BY pub_date DESC
            """, (feed_id,))
        else:
            cursor = conn.execute("""
                SELECT id, title, link, pub_date, feed_id 
                FROM node WHERE in_graph = 1 ORDER BY pub_date DESC
            """)
        
        nodes = []
        for row in cursor:
            nodes.append({
                "id": row[0],
                "title": row[1],
                "link": row[2],
                "pub_date": row[3],
                "feed_id": row[4]
            })
        conn.close()
        return {"nodes": nodes}
    
    def get_edges(self):
        feed_id = self._get_feed_id()
        conn = get_db()
        
        if feed_id:
            cursor = conn.execute("""
                SELECT source_id, target_id, weight, feed_id FROM edge WHERE feed_id = ? LIMIT 1000
            """, (feed_id,))
        else:
            cursor = conn.execute("SELECT source_id, target_id, weight, feed_id FROM edge LIMIT 1000")
        
        edges = []
        for row in cursor:
            edges.append({
                "source": row[0],
                "target": row[1],
                "weight": round(row[2], 4),
                "feed_id": row[3]
            })
        conn.close()
        return {"edges": edges}
    
    def get_feeds(self):
        """从 RSS 数据库获取 feed 列表"""
        conn = get_rss_db()
        cursor = conn.execute("""
            SELECT id, title, link FROM feed WHERE state = 1 ORDER BY id
        """)
        feeds = []
        for row in cursor:
            feeds.append({
                "id": row[0],
                "title": row[1],
                "link": row[2]
            })
        conn.close()
        return {"feeds": feeds}
    
    def get_node(self, node_id):
        conn = get_db()
        cursor = conn.execute("""
            SELECT id, title, link, pub_date, feed_id 
            FROM node WHERE id = ? AND in_graph = 1
        """, (node_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return {"error": "Node not found"}
        
        node = {
            "id": row[0],
            "title": row[1],
            "link": row[2],
            "pub_date": row[3],
            "feed_id": row[4]
        }
        
        # 获取相邻节点（按 feed 过滤，只返回同 feed 的边）
        neighbors = []
        cursor = conn.execute("""
            SELECT e.target_id, e.weight, n.title
            FROM edge e JOIN node n ON e.target_id = n.id
            WHERE e.source_id = ? AND e.feed_id = ?
        """, (node_id, row[4]))
        for row in cursor:
            neighbors.append({
                "node_id": row[0],
                "weight": round(row[1], 4),
                "title": row[2]
            })
        
        cursor = conn.execute("""
            SELECT e.source_id, e.weight, n.title
            FROM edge e JOIN node n ON e.source_id = n.id
            WHERE e.target_id = ? AND e.feed_id = ?
        """, (node_id, row[4]))
        for row in cursor:
            neighbors.append({
                "node_id": row[0],
                "weight": round(row[1], 4),
                "title": row[2]
            })
        
        conn.close()
        
        return {"node": node, "neighbors": neighbors}
    
    def get_stats(self):
        feed_id = self._get_feed_id()
        conn = get_db()
        
        if feed_id:
            cursor = conn.execute("SELECT COUNT(*) FROM node WHERE in_graph = 1 AND feed_id = ?", (feed_id,))
            node_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM edge WHERE feed_id = ?", (feed_id,))
            edge_count = cursor.fetchone()[0]
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM node WHERE in_graph = 1")
            node_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM edge")
            edge_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "nodeCount": node_count,
            "edgeCount": edge_count
        }
    
    # ==================== 新增知识库 API ====================
    
    def search_chunks(self):
        """Faiss 向量检索：根据查询文本搜索最相关的 chunks"""
        query = self._get_param("q", "")
        k = self._get_param("k", 10, int)
        
        if not query:
            return {"error": "Missing query parameter 'q'"}
        
        try:
            encoder = _get_encoder()
            faiss_store = _get_faiss_store()
            
            query_vec = encoder.encode_single(query)
            results = faiss_store.search(query_vec, k=k)
            
            # 补充文档信息
            conn = get_db()
            enriched = []
            for r in results:
                doc_row = conn.execute(
                    "SELECT title, link, pub_date FROM node WHERE id = ?",
                    (r["doc_id"],)
                ).fetchone()
                enriched.append({
                    "score": round(r["score"], 4),
                    "chunk_text": r["text"][:300],
                    "doc_id": r["doc_id"],
                    "doc_title": doc_row[0] if doc_row else "Unknown",
                    "doc_link": doc_row[1] if doc_row else None,
                    "doc_pub_date": doc_row[2] if doc_row else None,
                })
            conn.close()
            
            return {"query": query, "results": enriched}
        except Exception as e:
            return {"error": str(e)}
    
    def get_entities(self):
        """获取实体列表，可按类型过滤"""
        entity_type = self._get_param("type")
        limit = self._get_param("limit", 100, int)
        
        conn = get_db()
        if entity_type:
            cursor = conn.execute("""
                SELECT id, name, type, mention_count FROM entity
                WHERE type = ? ORDER BY mention_count DESC LIMIT ?
            """, (entity_type, limit))
        else:
            cursor = conn.execute("""
                SELECT id, name, type, mention_count FROM entity
                ORDER BY mention_count DESC LIMIT ?
            """, (limit,))
        
        entities = []
        for row in cursor:
            entities.append({
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "mention_count": row[3]
            })
        conn.close()
        return {"entities": entities}
    
    def get_entity_mentions(self):
        """获取某实体在哪些文档中被提及"""
        entity_id = self._get_param("entity_id", cast=int)
        if not entity_id:
            return {"error": "Missing entity_id"}
        
        conn = get_db()
        cursor = conn.execute("""
            SELECT em.doc_id, em.start_pos, em.end_pos, em.text_span,
                   n.title, n.link, n.pub_date
            FROM entity_mention em
            JOIN node n ON em.doc_id = n.id
            WHERE em.entity_id = ?
            ORDER BY n.pub_date DESC
        """, (entity_id,))
        
        mentions = []
        for row in cursor:
            mentions.append({
                "doc_id": row[0],
                "start_pos": row[1],
                "end_pos": row[2],
                "text_span": row[3],
                "doc_title": row[4],
                "doc_link": row[5],
                "doc_pub_date": row[6],
            })
        conn.close()
        return {"entity_id": entity_id, "mentions": mentions}
    
    def get_entity_relations(self):
        """获取某实体的共现关系网络"""
        entity_id = self._get_param("entity_id", cast=int)
        limit = self._get_param("limit", 50, int)
        
        if not entity_id:
            return {"error": "Missing entity_id"}
        
        conn = get_db()
        
        # 获取实体名称
        name_row = conn.execute("SELECT name, type FROM entity WHERE id = ?", (entity_id,)).fetchone()
        if not name_row:
            conn.close()
            return {"error": "Entity not found"}
        
        # 查询关系
        cursor = conn.execute("""
            SELECT er.source_entity_id, er.target_entity_id, er.relation_type,
                   er.confidence, er.doc_id,
                   s.name as source_name, t.name as target_name
            FROM entity_relation er
            JOIN entity s ON er.source_entity_id = s.id
            JOIN entity t ON er.target_entity_id = t.id
            WHERE er.source_entity_id = ? OR er.target_entity_id = ?
            ORDER BY er.confidence DESC
            LIMIT ?
        """, (entity_id, entity_id, limit))
        
        relations = []
        for row in cursor:
            relations.append({
                "source_id": row[0],
                "target_id": row[1],
                "relation_type": row[2],
                "confidence": row[3],
                "doc_id": row[4],
                "source_name": row[5],
                "target_name": row[6],
            })
        conn.close()
        
        return {
            "entity_id": entity_id,
            "entity_name": name_row[0],
            "entity_type": name_row[1],
            "relations": relations
        }
    
    def get_document_chunks(self):
        """获取某文档的所有 chunks"""
        doc_id = self._get_param("doc_id", cast=int)
        if not doc_id:
            return {"error": "Missing doc_id"}
        
        conn = get_db()
        node_row = conn.execute(
            "SELECT title, link, pub_date FROM node WHERE id = ?", (doc_id,)
        ).fetchone()
        
        cursor = conn.execute("""
            SELECT id, text, chunk_index, faiss_id FROM chunk
            WHERE doc_id = ? ORDER BY chunk_index
        """, (doc_id,))
        
        chunks = []
        for row in cursor:
            chunks.append({
                "id": row[0],
                "text": row[1],
                "chunk_index": row[2],
                "faiss_id": row[3],
            })
        conn.close()
        
        return {
            "doc_id": doc_id,
            "doc_title": node_row[0] if node_row else None,
            "doc_link": node_row[1] if node_row else None,
            "doc_pub_date": node_row[2] if node_row else None,
            "chunks": chunks
        }
    
    # ==================== 通用工具 ====================
    
    def send_json_response(self, data):
        response = json.dumps(data, ensure_ascii=False, indent=2)
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response.encode("utf-8"))
    
    def log_message(self, format, *args):
        print(f"[API] {format % args}")
    
    def serve_static_file(self, filepath):
        import os
        from pathlib import Path
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(os.path.dirname(script_dir), filepath)
        
        if os.path.exists(full_path) and os.path.isfile(full_path):
            self.send_response(200)
            
            if filepath.endswith('.html'):
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif filepath.endswith('.json'):
                self.send_header("Content-Type", "application/json")
            elif filepath.endswith('.png'):
                self.send_header("Content-Type", "image/png")
            else:
                self.send_header("Content-Type", "application/octet-stream")
            
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            with open(full_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File not found")


def run_server(port=8082):
    """运行 API 服务器"""
    os.chdir(Path(__file__).parent)
    
    server = HTTPServer(("0.0.0.0", port), GraphAPIHandler)
    print(f"[*] 新闻图谱 API 服务启动: http://localhost:{port}")
    print(f"[*] 端点:")
    print(f"    /api/graph?feed_id=<id>       - 图数据")
    print(f"    /api/nodes?feed_id=<id>       - 节点列表")
    print(f"    /api/edges?feed_id=<id>       - 边列表")
    print(f"    /api/stats?feed_id=<id>       - 统计信息")
    print(f"    /api/feeds                    - RSS 源列表")
    print(f"    /api/node?id=<id>             - 单个节点详情")
    print(f"    /api/search?q=<text>&k=10     - Faiss 向量检索")
    print(f"    /api/entities?type=&limit=    - 实体列表")
    print(f"    /api/entity_mentions?entity_id=<id>  - 实体提及")
    print(f"    /api/entity_relations?entity_id=<id> - 实体关系")
    print(f"    /api/document_chunks?doc_id=<id>     - 文档 chunks")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] 停止服务")
        server.shutdown()


if __name__ == "__main__":
    run_server()
