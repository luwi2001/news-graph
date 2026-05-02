#!/usr/bin/env python3
"""
新闻图谱 API 服务
"""
import sqlite3
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os
from pathlib import Path

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
    
    def do_GET(self):
        path = urlparse(self.path).path
        
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
        elif path == "/" or path == "/index.html" or path == "/visualize.html":
            self.serve_static_file("data/output/visualize.html")
        elif path.startswith("/data/"):
            self.serve_static_file(path)
        else:
            super().do_GET()
    
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
    print(f"    /api/graph?feed_id=<id>   - 图数据（可选按 feed 过滤）")
    print(f"    /api/nodes?feed_id=<id>   - 节点列表")
    print(f"    /api/edges?feed_id=<id>   - 边列表")
    print(f"    /api/stats?feed_id=<id>   - 统计信息")
    print(f"    /api/feeds                - RSS 源列表")
    print(f"    /api/node?id=<id>         - 单个节点详情")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] 停止服务")
        server.shutdown()


if __name__ == "__main__":
    run_server()