#!/usr/bin/env python3
"""
新闻图谱 API 服务（知识库版）
新增：Faiss 向量检索、实体关系查询
"""
import sqlite3
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

import sys
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# 懒加载的知识库组件（首次请求时初始化）
_encoder = None
_faiss_store = None

# 配置：哪些 feed 合并显示为一个板块
FEED_MERGE = {13: 3}  # Reuters World China -> Bloomberg Politics - Asia


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
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_POST(self):
        path = urlparse(self.path).path
        
        if path == "/api/summarize":
            self.send_json_response(self.summarize_highlighted())
        else:
            self.send_error(404, "Not found")
    
    def _read_json_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            return {}
    
    def summarize_highlighted(self):
        """聚合高亮新闻，调用 AI 生成解读"""
        body = self._read_json_body()
        node_ids = body.get("node_ids", [])
        
        if not node_ids:
            return {"error": "No node_ids provided"}
        
        # 获取 feed 映射
        feed_map = {}
        conn_rss = get_rss_db()
        cursor = conn_rss.execute("SELECT id, title FROM feed WHERE state = 1")
        for row in cursor:
            feed_map[row[0]] = row[1]
        
        # 从 news_graph.db 获取节点信息
        conn_graph = get_db()
        placeholders = ','.join('?' * len(node_ids))
        cursor = conn_graph.execute(f"""
            SELECT id, title, link, pub_date, feed_id 
            FROM node 
            WHERE id IN ({placeholders}) AND in_graph = 1
            ORDER BY pub_date ASC
        """, node_ids)
        
        nodes = []
        for row in cursor:
            nodes.append({
                "id": row[0],
                "title": row[1],
                "link": row[2],
                "pub_date": row[3],
                "feed_id": row[4]
            })
        
        if not nodes:
            conn_graph.close()
            conn_rss.close()
            return {"error": "No valid nodes found"}
        
        # 从 RSS db 获取完整内容
        enriched_nodes = []
        for n in nodes:
            cursor = conn_rss.execute("""
                SELECT description FROM entries WHERE link = ?
            """, (n["link"],))
            row = cursor.fetchone()
            description = row[0] if row else ""
            description = re.sub(r'<[^>]+>', ' ', description)
            description = re.sub(r'\s+', ' ', description).strip()
            if len(description) > 2000:
                description = description[:2000] + "..."
            
            enriched_nodes.append({
                "id": n["id"],
                "title": n["title"],
                "link": n["link"],
                "pub_date": n["pub_date"],
                "feed_name": feed_map.get(n["feed_id"], "RSS"),
                "description": description
            })
        
        conn_graph.close()
        conn_rss.close()
        
        # 生成 prompt
        prompt = self._build_summary_prompt(enriched_nodes)
        
        # 调用 AI API（多模型 fallback，返回成本信息）
        ai_result = self._call_ai_api(prompt)
        ai_response = ai_result["content"]
        model_used = ai_result["model"]
        cost = ai_result.get("cost", 0)
        prompt_tokens = ai_result.get("prompt_tokens", 0)
        completion_tokens = ai_result.get("completion_tokens", 0)
        
        # 检查是否成功
        if ai_response.startswith("**"):
            return {
                "fallback": True,
                "error": ai_response,
                "prompt": prompt,
                "node_count": len(enriched_nodes)
            }
        
        return {
            "success": True,
            "summary": ai_response,
            "node_count": len(enriched_nodes),
            "model": model_used,
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    
    def _build_summary_prompt(self, nodes):
        """构建深度 AI 解读 prompt - v3 自由分析版"""
        # 按 feed 分组统计
        feed_counts = {}
        for n in nodes:
            f = n["feed_name"]
            feed_counts[f] = feed_counts.get(f, 0) + 1
        feed_summary = ", ".join([f"{k}({v}篇)" for k, v in feed_counts.items()])
        
        # 构建新闻材料（保留完整描述但控制长度）
        news_text = []
        for i, n in enumerate(nodes, 1):
            date = n["pub_date"][:10] if n["pub_date"] else "未知"
            time_part = n["pub_date"][11:16] if n["pub_date"] and len(n["pub_date"]) > 11 else ""
            time_str = f" {time_part}" if time_part else ""
            desc = n["description"]
            if len(desc) > 900:
                desc = desc[:900] + "..."
            
            news_text.append(f"""【{i}】{date}{time_str} | {n['feed_name']}
标题: {n['title']}
摘要: {desc}
""")
        
        prompt = f"""你是一位顶级地缘政治与产业分析师，拥有20年研究经验，擅长穿透表象挖掘事件的本质逻辑。你的分析风格：锋利、有洞察力、拒绝和稀泥。

## 输入材料
共 {len(nodes)} 篇新闻，来源分布: {feed_summary}

{'\n---\n'.join(news_text)}

---

## 输出要求

### 一、新闻简述（逐篇展开）
按新闻发布时间列举,不要只复述标题。对每条新闻用 **2-3 句话** 展开说明，抓住重点：
- 这条新闻的核心事实是什么？（谁做了什么？数据是多少？）
- **为什么这件事值得关注？** 它打破了什么预期、触发了什么连锁反应？
- 如果后文会用到这条新闻来建立关联，在这里就把关键伏笔点出来——比如具体数据、表态措辞、政策细节。

**要求**：每条都要具体，不要空话。如果信息不足就写"信息有限"。

### 二、定锚
在简述基础上回答：
- 这串新闻本质上反映了一个什么结构性问题？
- 为什么是这个时间点爆发？
- 放在5年或10年的尺度上，这件事意味着什么？

### 三、关联重构（自然讲述）
**不要**用"A导致B""B引发C"这种机械格式。像讲故事一样，自然地指出新闻之间真实存在的关系,考虑时间和逻辑关系。重点讲三类：

**1. 互相印证的**
哪些新闻从不同角度指向了同一个趋势/同一个判断？它们是如何互相加强的？

**2. 有矛盾的**
哪些新闻在关键事实上说法不一致？具体矛盾点是什么？
- 尝试给出合理解释：是不同来源的利益立场不同？是时间差导致信息更新？是有意隐瞒？
- 如果无法判断谁对谁错，明确说"信息冲突，存疑"。

**3. 看似无关实则有关的**
哪些新闻表面上讲的是两件不同的事，但结合某些背景信息后，其实存在隐藏关联？
- 需要什么外部信息才能把这两件事连起来？
- 这个隐藏关联如果成立，意味着什么？

**底线**：如果某两篇新闻之间确实没有可见的联系，直接跳过，不要硬拗。

### 四、外部知识补充
结合你的背景知识，补充报道中未提及但至关重要的上下文：
- 相关国家的历史政策轨迹
- 涉及的产业/技术的全球格局
- 类似事件的历史先例及其后续发展
- 当前宏观环境（利率周期、地缘冲突、能源转型等）如何放大了这件事的影响

### 五、前瞻（带赌注）
给出明确的倾向性判断：
- 最可能的走向是什么？
- 如果我是对的，接下来会先看到什么信号？
- 什么证据会让我立刻推翻这个判断？
- 报道中完全没提但可能爆发的黑天鹅是什么？

## 输出规范
- 语言像给CEO的briefing，不要像给学生讲课
- 每段都要有"所以"——不要只描述现象，要给出判断
- 如果信息不足以支撑某个论点，明确标出"此处信息不足，存疑"
- Markdown 格式
- 不需要控制字数，分析充分即可
"""
        return prompt
    
    def _call_ai_api(self, prompt):
        """调用 OpenRouter API，优先使用非中国模型，支持自动 fallback"""
        import urllib.request
        import urllib.error
        
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return {"error": "OPENROUTER_API_KEY environment variable not set"}
        
        # 模型优先级：DeepSeek 首选 → Mistral Large 次选 → Llama 3.3 第三 → Phi-4 保底
        # deepseek/deepseek-chat-v3-0324 (DeepSeek/中国，推理最强)
        # mistralai/mistral-large-2411 (Mistral/法国，旗舰大模型)
        # meta-llama/llama-3.3-70b-instruct (Meta/美国，新版70B)
        # microsoft/phi-4 (Microsoft/美国，小模型但质量好)
        models = [
            ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3"),
            ("mistralai/mistral-large-2411", "Mistral Large"),
            ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
            ("microsoft/phi-4", "Phi-4")
        ]
        
        last_error = ""
        
        for model_id, model_name in models:
            try:
                payload = json.dumps({
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "你是一位顶级地缘政治与产业分析师，拥有20年研究经验，擅长穿透表象挖掘事件的本质逻辑。你的分析风格：锋利、有洞察力、拒绝和稀泥。"},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 8000,
                    "temperature": 0.7
                }).encode('utf-8')
                
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "http://localhost:8082",
                        "X-Title": "NewsGraph AI Summary"
                    },
                    method="POST"
                )
                
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read().decode('utf-8'))
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        # 获取花费信息
                        usage = result.get("usage", {})
                        cost = result.get("cost", usage.get("cost", 0))
                        if not cost and "cost_details" in result:
                            cost = result["cost_details"].get("upstream_inference_cost", 0)
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        return {
                            "content": content,
                            "model": model_name,
                            "cost": cost,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens
                        }
                    else:
                        error_msg = json.dumps(result, ensure_ascii=False)[:300]
                        last_error = f"{model_name} 响应异常: {error_msg}"
                        continue
                        
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8')
                last_error = f"{model_name} HTTP {e.code}: {body[:300]}"
                # 如果是 4xx 错误（如 401 key 无效），不再尝试其他模型
                if 400 <= e.code < 500 and e.code != 429:
                    break
                continue
            except Exception as e:
                last_error = f"{model_name} 异常: {str(e)[:300]}"
                continue
        
        return {"content": f"**API 调用失败**: {last_error}", "model": ""}
    
    # ==================== 现有 API ====================
    
    def get_graph(self):
        feed_id = self._get_feed_id()
        conn = get_db()
        
        # 计算需要查询的 feed_id 列表（包含被合并的 feed）
        if feed_id:
            target_feed_ids = [feed_id]
            for src, tgt in FEED_MERGE.items():
                if tgt == feed_id:
                    target_feed_ids.append(src)
            placeholders = ','.join('?' * len(target_feed_ids))
        else:
            target_feed_ids = None
            placeholders = None
        
        if feed_id:
            nodes = []
            cursor = conn.execute(f"""
                SELECT id, title, link, pub_date, feed_id 
                FROM node WHERE in_graph = 1 AND feed_id IN ({placeholders}) ORDER BY pub_date DESC
            """, target_feed_ids)
            for row in cursor:
                nodes.append({
                    "id": row[0],
                    "title": row[1],
                    "link": row[2],
                    "pub_date": row[3],
                    "feed_id": row[4]
                })
            
            edges = []
            cursor = conn.execute(f"""
                SELECT source_id, target_id, weight, feed_id FROM edge WHERE feed_id IN ({placeholders})
            """, target_feed_ids)
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
            # 计算需要查询的 feed_id 列表（包含被合并的 feed）
            target_feed_ids = [feed_id]
            for src, tgt in FEED_MERGE.items():
                if tgt == feed_id:
                    target_feed_ids.append(src)
            placeholders = ','.join('?' * len(target_feed_ids))
            cursor = conn.execute(f"""
                SELECT id, title, link, pub_date, feed_id 
                FROM node WHERE in_graph = 1 AND feed_id IN ({placeholders}) ORDER BY pub_date DESC
            """, target_feed_ids)
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
            # 计算需要查询的 feed_id 列表（包含被合并的 feed）
            target_feed_ids = [feed_id]
            for src, tgt in FEED_MERGE.items():
                if tgt == feed_id:
                    target_feed_ids.append(src)
            placeholders = ','.join('?' * len(target_feed_ids))
            cursor = conn.execute(f"""
                SELECT source_id, target_id, weight, feed_id FROM edge WHERE feed_id IN ({placeholders}) LIMIT 1000
            """, target_feed_ids)
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
        """从 RSS 数据库获取 feed 列表，合并配置的 feed"""
        conn = get_rss_db()
        cursor = conn.execute("""
            SELECT id, title, link FROM feed WHERE state = 1 ORDER BY id
        """)
        feeds = []
        merged_target_ids = set(FEED_MERGE.values())
        for row in cursor:
            feed_id = row[0]
            # 被合并的 feed 不单独显示
            if feed_id in FEED_MERGE:
                continue
            # 如果是合并目标，标题加上合并来源
            title = row[1]
            if feed_id in merged_target_ids:
                sources = [f'ID{k}' for k, v in FEED_MERGE.items() if v == feed_id]
                if sources:
                    title = f"{title} (含 {', '.join(sources)})"
            feeds.append({
                "id": feed_id,
                "title": title,
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
        response_bytes = response.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)
    
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
    print(f"    POST /api/summarize           - AI 解读高亮新闻")
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
